//! End-to-end measurement of the "VCF INFO → typed Arrow sub-columns" wedge.
//!
//! falx locates the INFO field with its SIMD structural kernel and extracts the
//! requested keys (DP:i64, AF:f64) in a single allocation-free pass straight
//! into Arrow columnar buffers (values + validity). The baseline is `noodles`,
//! the scalar Rust VCF library that every Arrow-genomics tool (oxbow, exon,
//! biobear, polars-bio) wraps. noodles' lazy `Info::get` rescans the INFO string
//! from the start for *each* key, so the gap is expected to widen with the
//! number of INFO keys present and the number of keys projected.
//!
//! Run: `cargo run --release --example vcf_info_bench -- [rows] [iters]`

use std::io::Write as _;
use std::time::Instant;

use arrow_array::{Array, Float64Array, Int64Array};
use falx::kernels::vcf;
use noodles_vcf::variant::record::info::field::Value;

/// Deterministic xorshift64* — no external rng dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    fn unit(&mut self) -> f64 {
        (self.below(1_000_000) as f64) / 1_000_000.0
    }
}

/// Twelve INFO keys, GATK/gnomAD-shaped. DP is mid-list and AF near the front,
/// so noodles' per-key rescan walks several fields each lookup.
const INFO_HEADER: &str = "\
##INFO=<ID=AC,Number=A,Type=Integer,Description=\"\">
##INFO=<ID=AF,Number=A,Type=Float,Description=\"\">
##INFO=<ID=AN,Number=1,Type=Integer,Description=\"\">
##INFO=<ID=BaseQRankSum,Number=1,Type=Float,Description=\"\">
##INFO=<ID=ExcessHet,Number=1,Type=Float,Description=\"\">
##INFO=<ID=DP,Number=1,Type=Integer,Description=\"\">
##INFO=<ID=FS,Number=1,Type=Float,Description=\"\">
##INFO=<ID=MQ,Number=1,Type=Float,Description=\"\">
##INFO=<ID=MQRankSum,Number=1,Type=Float,Description=\"\">
##INFO=<ID=QD,Number=1,Type=Float,Description=\"\">
##INFO=<ID=ReadPosRankSum,Number=1,Type=Float,Description=\"\">
##INFO=<ID=SOR,Number=1,Type=Float,Description=\"\">
";

fn make_vcf(rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * 200 + 1024);
    out.extend_from_slice(b"##fileformat=VCFv4.3\n");
    out.extend_from_slice(INFO_HEADER.as_bytes());
    out.extend_from_slice(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    const BASES: &[u8] = b"ACGT";
    for row in 1..=rows as u64 {
        let _ = write!(
            out,
            "chr{}\t{}\trs{row}\t{}\t{}\t{:.1}\tPASS\t",
            1 + rng.below(22),
            1 + rng.below(250_000_000),
            BASES[rng.below(4) as usize] as char,
            BASES[rng.below(4) as usize] as char,
            10.0 + rng.unit() * 89.0,
        );
        // ~5% of rows omit DP, ~5% omit AF — exercises validity bitmaps.
        let drop_dp = rng.below(20) == 0;
        let drop_af = rng.below(20) == 0;
        let _ = write!(out, "AC={};", 1 + rng.below(4));
        if !drop_af {
            let _ = write!(out, "AF={:.4};", rng.unit());
        }
        let _ = write!(out, "AN={};", 2 + rng.below(6));
        let _ = write!(out, "BaseQRankSum={:.3};", rng.unit() * 4.0 - 2.0);
        let _ = write!(out, "ExcessHet={:.4};", rng.unit() * 10.0);
        if !drop_dp {
            let _ = write!(out, "DP={};", 10 + rng.below(500));
        }
        let _ = write!(out, "FS={:.3};", rng.unit() * 30.0);
        let _ = write!(out, "MQ={:.2};", 40.0 + rng.unit() * 20.0);
        let _ = write!(out, "MQRankSum={:.3};", rng.unit() * 4.0 - 2.0);
        let _ = write!(out, "QD={:.2};", rng.unit() * 35.0);
        let _ = write!(out, "ReadPosRankSum={:.3};", rng.unit() * 4.0 - 2.0);
        let _ = writeln!(out, "SOR={:.3}", rng.unit() * 3.0);
    }
    out
}

#[derive(Default)]
struct Columns {
    rows: usize,
    dp: Vec<i64>,
    dp_valid: Vec<bool>,
    af: Vec<f64>,
    af_valid: Vec<bool>,
}

#[inline]
fn parse_i64(b: &[u8]) -> i64 {
    let (neg, body) = match b.first() {
        Some(b'-') => (true, &b[1..]),
        _ => (false, b),
    };
    let mut n: i64 = 0;
    for &c in body {
        if c.is_ascii_digit() {
            n = n * 10 + (c - b'0') as i64;
        }
    }
    if neg { -n } else { n }
}

#[inline]
fn parse_f64(b: &[u8]) -> f64 {
    // SAFETY-free: INFO numeric values are ASCII; fall back to 0.0 on garbage.
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// falx path: SIMD structural parse → INFO field (index 7) → single-pass scan
/// extracting DP and AF together, zero per-record allocation.
fn falx_extract(data: &[u8]) -> Columns {
    let parsed = vcf::parse(data);
    let mut cols = Columns::default();
    for rec in parsed.records() {
        cols.rows += 1;
        let info = rec.field_raw(7).unwrap_or(b"");
        let (mut dp, mut af): (Option<i64>, Option<f64>) = (None, None);
        let mut i = 0usize;
        while i < info.len() {
            let kstart = i;
            while i < info.len() && info[i] != b'=' && info[i] != b';' {
                i += 1;
            }
            let key = &info[kstart..i];
            let mut val: &[u8] = b"";
            if i < info.len() && info[i] == b'=' {
                i += 1;
                let vstart = i;
                while i < info.len() && info[i] != b';' {
                    i += 1;
                }
                val = &info[vstart..i];
            }
            match key {
                b"DP" => dp = Some(parse_i64(val)),
                b"AF" => af = Some(parse_f64(val)),
                _ => {}
            }
            if i < info.len() && info[i] == b';' {
                i += 1;
            }
        }
        cols.dp.push(dp.unwrap_or(0));
        cols.dp_valid.push(dp.is_some());
        cols.af.push(af.unwrap_or(0.0));
        cols.af_valid.push(af.is_some());
    }
    cols
}

/// Baseline: noodles lazy records + `Info::get` per key (the path oxbow/exon/
/// biobear/polars-bio build on).
fn noodles_extract(data: &[u8]) -> Columns {
    let mut reader = noodles_vcf::io::Reader::new(std::io::Cursor::new(data));
    let header = reader.read_header().expect("valid VCF header");
    let mut cols = Columns::default();
    for rec in reader.records() {
        let rec = rec.expect("valid VCF record");
        cols.rows += 1;
        let info = rec.info();
        match info.get(&header, "DP") {
            Some(Ok(Some(Value::Integer(n)))) => {
                cols.dp.push(n as i64);
                cols.dp_valid.push(true);
            }
            _ => {
                cols.dp.push(0);
                cols.dp_valid.push(false);
            }
        }
        // AF is reserved Number=A, so noodles yields an Array for single-ALT
        // records; we only assert AF *presence* against falx (the headline
        // value comparison is DP), so match any present value.
        match info.get(&header, "AF") {
            Some(Ok(Some(_))) => {
                cols.af.push(0.0);
                cols.af_valid.push(true);
            }
            _ => {
                cols.af.push(0.0);
                cols.af_valid.push(false);
            }
        }
    }
    cols
}

/// falx: single SIMD-located INFO pass, counting how many records carry each of
/// `keys` (the projection lookup cost, independent of K beyond the match check).
fn falx_project(data: &[u8], keys: &[&str]) -> Vec<usize> {
    let mut counts = vec![0usize; keys.len()];
    let parsed = vcf::parse(data);
    for rec in parsed.records() {
        let info = rec.field_raw(7).unwrap_or(b"");
        let mut i = 0usize;
        while i < info.len() {
            let kstart = i;
            while i < info.len() && info[i] != b'=' && info[i] != b';' {
                i += 1;
            }
            let key = &info[kstart..i];
            // value (skipped — we measure location/lookup cost here)
            if i < info.len() && info[i] == b'=' {
                i += 1;
                while i < info.len() && info[i] != b';' {
                    i += 1;
                }
            }
            for (j, want) in keys.iter().enumerate() {
                if key == want.as_bytes() {
                    counts[j] += 1;
                    break;
                }
            }
            if i < info.len() && info[i] == b';' {
                i += 1;
            }
        }
    }
    counts
}

/// noodles: one `Info::get` per key per record — each a rescan from the start.
fn noodles_project(data: &[u8], keys: &[&str]) -> Vec<usize> {
    let mut reader = noodles_vcf::io::Reader::new(std::io::Cursor::new(data));
    let header = reader.read_header().expect("valid VCF header");
    let mut counts = vec![0usize; keys.len()];
    for rec in reader.records() {
        let rec = rec.expect("valid VCF record");
        let info = rec.info();
        for (j, key) in keys.iter().enumerate() {
            if matches!(info.get(&header, key), Some(Ok(Some(_)))) {
                counts[j] += 1;
            }
        }
    }
    counts
}

/// Materialize the falx columns into real Arrow arrays (values + validity).
fn to_arrow(cols: &Columns) -> (Int64Array, Float64Array) {
    let dp = Int64Array::from_iter(
        cols.dp
            .iter()
            .zip(&cols.dp_valid)
            .map(|(&v, &ok)| ok.then_some(v)),
    );
    let af = Float64Array::from_iter(
        cols.af
            .iter()
            .zip(&cols.af_valid)
            .map(|(&v, &ok)| ok.then_some(v)),
    );
    (dp, af)
}

fn best_of(iters: usize, mut f: impl FnMut() -> Columns) -> (std::time::Duration, Columns) {
    let mut best = std::time::Duration::MAX;
    let mut last = Columns::default();
    for _ in 0..iters {
        let t = Instant::now();
        let c = f();
        let dt = t.elapsed();
        if dt < best {
            best = dt;
        }
        last = c;
    }
    (best, last)
}

fn throughput(bytes: usize, rows: usize, dt: std::time::Duration) -> (f64, f64) {
    let s = dt.as_secs_f64();
    (
        (bytes as f64) / s / (1024.0 * 1024.0), // MiB/s
        (rows as f64) / s / 1.0e6,              // M records/s
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(800_000);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let data = make_vcf(rows);
    let mb = data.len() as f64 / (1024.0 * 1024.0);
    println!(
        "dataset: {rows} records, {:.1} MiB, 12 INFO keys (DP:i64, AF:f64 projected)\n\
         method: best of {iters} interleaved runs\n",
        mb
    );

    // Warm up + correctness gate before timing.
    let fx = falx_extract(&data);
    let nd = noodles_extract(&data);
    assert_eq!(fx.rows, nd.rows, "record count mismatch");
    assert_eq!(fx.dp, nd.dp, "DP values diverge");
    assert_eq!(fx.dp_valid, nd.dp_valid, "DP validity diverges");
    assert_eq!(fx.af_valid, nd.af_valid, "AF validity diverges");
    let dp_present = fx.dp_valid.iter().filter(|&&b| b).count();
    let af_present = fx.af_valid.iter().filter(|&&b| b).count();
    println!(
        "correctness: DP column identical to noodles ({dp_present}/{rows} present); \
         AF validity identical ({af_present}/{rows} present)\n"
    );

    // Interleave to share cache/clock conditions.
    let mut falx_best = std::time::Duration::MAX;
    let mut nood_best = std::time::Duration::MAX;
    for _ in 0..iters {
        let (df, _) = best_of(1, || falx_extract(&data));
        let (dn, _) = best_of(1, || noodles_extract(&data));
        falx_best = falx_best.min(df);
        nood_best = nood_best.min(dn);
    }

    let (f_mb, f_rec) = throughput(data.len(), rows, falx_best);
    let (n_mb, n_rec) = throughput(data.len(), rows, nood_best);
    println!(
        "falx    : {:>8.2} ms   {f_mb:>8.1} MiB/s   {f_rec:>6.2} M rec/s",
        falx_best.as_secs_f64() * 1e3
    );
    println!(
        "noodles : {:>8.2} ms   {n_mb:>8.1} MiB/s   {n_rec:>6.2} M rec/s",
        nood_best.as_secs_f64() * 1e3
    );
    println!(
        "\nspeedup : {:.2}x  (falx faster, 2 INFO keys → typed Arrow columns)",
        f_mb / n_mb
    );

    // Scaling: noodles' Info::get rescans INFO from the start for each key, so
    // projecting more keys is K passes; falx extracts all K in one pass. This is
    // the wedge — measure how the gap moves with key count.
    const KEYS: &[&str] = &["DP", "AF", "AC", "AN", "MQ", "QD", "FS", "SOR"];
    println!("\nkey-count scaling (falx single-pass vs noodles K rescans):");
    for &k in &[1usize, 2, 4, 8] {
        let keys = &KEYS[..k];
        let cf = falx_project(&data, keys);
        let cn = noodles_project(&data, keys);
        assert_eq!(cf, cn, "present-count mismatch at K={k}");
        let mut fb = std::time::Duration::MAX;
        let mut nb = std::time::Duration::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            std::hint::black_box(falx_project(&data, keys));
            fb = fb.min(t.elapsed());
            let t = Instant::now();
            std::hint::black_box(noodles_project(&data, keys));
            nb = nb.min(t.elapsed());
        }
        let (ff, _) = throughput(data.len(), rows, fb);
        let (nn, _) = throughput(data.len(), rows, nb);
        println!(
            "  K={k}: falx {ff:>7.1} MiB/s | noodles {nn:>7.1} MiB/s | {:.2}x",
            ff / nn
        );
    }

    // Prove the Arrow end-to-end output.
    let (dp_arr, af_arr) = to_arrow(&fx);
    println!(
        "\narrow   : Int64Array len={} nulls={}, Float64Array len={} nulls={}",
        dp_arr.len(),
        dp_arr.null_count(),
        af_arr.len(),
        af_arr.null_count()
    );
}
