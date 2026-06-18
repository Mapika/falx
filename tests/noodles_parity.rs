//! Parity tests: falx's parsed VCF columns must equal what `noodles` (the
//! scalar reference every Arrow-genomics tool wraps) reads from the same bytes —
//! value by value, across real-VCF edge cases (multi-allelic ALT, indels,
//! missing DP, missing/`.` QUAL, varied multi-key INFO). Covers the
//! uncompressed `parse_columns` path (serial + parallel) and, under the `bgzf`
//! feature, the bgzf decompressor and the fused `.vcf.gz` driver.

use std::fmt::Write as _;

use falx::kernels::vcf_typed;
use noodles_vcf::variant::record::info::field::Value;

/// One record reduced to the fields both libraries expose, for exact compare.
#[derive(Debug, PartialEq)]
struct Rec {
    pos: i64,
    refb: Vec<u8>,
    alt: Vec<u8>,
    qual: Option<f64>,
    dp: Option<i64>,
    af_present: bool,
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D) % n
    }
}

/// A VCF whose records exercise the edge cases that distinguish a real parser:
/// SNVs / multi-allelic ALT / indels, integer-or-missing QUAL, and INFO where
/// DP and AF are each sometimes absent and never in a fixed position.
fn make_vcf(records: usize) -> Vec<u8> {
    let mut out = String::new();
    out.push_str("##fileformat=VCFv4.3\n");
    out.push_str("##INFO=<ID=AC,Number=A,Type=Integer,Description=\"\">\n");
    out.push_str("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"\">\n");
    out.push_str("##INFO=<ID=AF,Number=A,Type=Float,Description=\"\">\n");
    out.push_str("##INFO=<ID=MQ,Number=1,Type=Float,Description=\"\">\n");
    out.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");
    let mut rng = Rng(0xC0FF_EE12_3456_789A);
    const BASES: [&str; 4] = ["A", "C", "G", "T"];
    for i in 1..=records as u64 {
        let chrom = 1 + rng.below(22);
        let pos = 1 + rng.below(250_000_000);
        let (refb, alt) = match rng.below(4) {
            0 => ("AG".to_string(), "A".to_string()),  // deletion
            1 => ("A".to_string(), "ATG".to_string()), // insertion
            2 => (
                // multi-allelic
                BASES[rng.below(4) as usize].to_string(),
                format!(
                    "{},{}",
                    BASES[rng.below(4) as usize],
                    BASES[rng.below(4) as usize]
                ),
            ),
            _ => (
                BASES[rng.below(4) as usize].to_string(),
                BASES[rng.below(4) as usize].to_string(),
            ),
        };
        let qual = if rng.below(5) == 0 {
            ".".to_string() // missing
        } else {
            (1 + rng.below(99)).to_string() // integer QUAL → exact as f64 both sides
        };
        // INFO: AC always; DP ~80%; AF ~80%; MQ always — order fixed but DP/AF
        // land at different offsets depending on which are present.
        let mut info = format!("AC={}", 1 + rng.below(3));
        if rng.below(5) != 0 {
            let _ = write!(info, ";DP={}", 1 + rng.below(1000));
        }
        if rng.below(5) != 0 {
            let _ = write!(info, ";AF={:.3}", (rng.below(1000) as f64) / 1000.0);
        }
        let _ = write!(info, ";MQ={}", 30 + rng.below(30));
        let _ = writeln!(
            out,
            "chr{chrom}\t{pos}\trs{i}\t{refb}\t{alt}\t{qual}\tPASS\t{info}"
        );
    }
    out.into_bytes()
}

fn falx_records(data: &[u8]) -> Vec<Rec> {
    let c = vcf_typed::parse_columns(data);
    (0..c.rows)
        .map(|i| Rec {
            pos: c.pos[i],
            refb: c.span(c.reference[i]).to_vec(),
            alt: c.span(c.alternate[i]).to_vec(),
            qual: vcf_typed::bitmap_get(&c.quality_valid, i).then(|| c.quality[i]),
            dp: vcf_typed::bitmap_get(&c.dp_valid, i).then(|| c.dp[i]),
            af_present: vcf_typed::bitmap_get(&c.af_valid, i),
        })
        .collect()
}

fn noodles_records(data: &[u8]) -> Vec<Rec> {
    let mut reader = noodles_vcf::io::Reader::new(std::io::Cursor::new(data));
    let header = reader.read_header().expect("valid VCF header");
    let mut out = Vec::new();
    for rec in reader.records() {
        let rec = rec.expect("valid VCF record");
        let pos = rec
            .variant_start()
            .expect("POS present")
            .expect("valid POS")
            .get() as i64;
        let refb = rec.reference_bases().as_bytes().to_vec();
        let alt = rec.alternate_bases().as_ref().as_bytes().to_vec();
        let qual = rec.quality_score().map(|q| q.expect("valid QUAL") as f64);
        let dp = match rec.info().get(&header, "DP") {
            Some(Ok(Some(Value::Integer(n)))) => Some(n as i64),
            _ => None,
        };
        let af_present = matches!(rec.info().get(&header, "AF"), Some(Ok(Some(_))));
        out.push(Rec {
            pos,
            refb,
            alt,
            qual,
            dp,
            af_present,
        });
    }
    out
}

#[test]
fn vcf_typed_columns_match_noodles() {
    let data = make_vcf(20_000);
    let falx = falx_records(&data);
    let noodles = noodles_records(&data);
    assert_eq!(falx.len(), noodles.len(), "record count");
    assert_eq!(falx, noodles, "parsed columns diverge from noodles");
}

#[test]
fn parallel_columns_match_serial_and_noodles() {
    let data = make_vcf(50_000);
    let c = vcf_typed::parse_columns_par(&data, 8);
    let par: Vec<Rec> = (0..c.rows)
        .map(|i| Rec {
            pos: c.pos[i],
            refb: c.span(c.reference[i]).to_vec(),
            alt: c.span(c.alternate[i]).to_vec(),
            qual: vcf_typed::bitmap_get(&c.quality_valid, i).then(|| c.quality[i]),
            dp: vcf_typed::bitmap_get(&c.dp_valid, i).then(|| c.dp[i]),
            af_present: vcf_typed::bitmap_get(&c.af_valid, i),
        })
        .collect();
    assert_eq!(par, falx_records(&data), "parallel diverges from serial");
    assert_eq!(
        par,
        noodles_records(&data),
        "parallel diverges from noodles"
    );
}

#[cfg(feature = "bgzf")]
mod bgzf_parity {
    use super::*;
    use falx::bgzf;
    use std::io::Write as _;

    fn bgzip(data: &[u8]) -> Vec<u8> {
        let mut w = noodles_bgzf::io::Writer::new(Vec::new());
        w.write_all(data).expect("bgzf write");
        w.finish().expect("bgzf finish")
    }

    #[test]
    fn bgzf_decompress_matches_noodles_and_source() {
        let data = make_vcf(40_000);
        let comp = bgzip(&data);
        let falx = bgzf::decompress(&comp).expect("falx bgzf");
        assert_eq!(falx, data, "falx bgzf decompress != source");
        let par = bgzf::decompress_par(&comp, 8).expect("falx bgzf par");
        assert_eq!(par, data, "falx bgzf parallel != source");
        use std::io::Read as _;
        let mut nood = Vec::new();
        noodles_bgzf::io::Reader::new(std::io::Cursor::new(&comp))
            .read_to_end(&mut nood)
            .expect("noodles bgzf");
        assert_eq!(falx, nood, "falx bgzf != noodles bgzf");
    }

    #[test]
    fn fused_vcf_gz_columns_match_noodles() {
        let data = make_vcf(60_000);
        let comp = bgzip(&data);
        let fused: Vec<Rec> = bgzf::parse_gz_par(&comp, 8, b'\n', falx_records)
            .expect("fusion")
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(
            fused,
            noodles_records(&data),
            "fused .vcf.gz diverges from noodles"
        );
    }
}
