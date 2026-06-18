//! Real-data validation of the .vcf.gz → Arrow pipeline against an independent
//! scalar ground truth (not noodles, so the checks don't depend on noodles
//! accepting a real file's quirks). noodles is used only for the bgzf reference
//! and the throughput baseline.
//!
//! Run: `cargo run --release --features bgzf --example validate_vcf_gz -- <file.vcf.gz> [INFO_KEY]`

use std::io::Read as _;
use std::time::Instant;

use falx::bgzf;
use falx::kernels::{vcf, vcf_typed};
use noodles_bgzf as nbgzf;

/// Independent reference: a plain scalar pass over the decompressed VCF.
struct Truth {
    records: usize,
    pos: Vec<i64>,
    key: Vec<Option<i64>>, // the requested INFO key, parsed as integer
}

fn scalar_i64(b: &[u8]) -> Option<i64> {
    let (neg, body) = match b.first() {
        Some(b'-') => (true, &b[1..]),
        _ => (false, b),
    };
    if body.is_empty() || !body.iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut n = 0i64;
    for &c in body {
        n = n.wrapping_mul(10).wrapping_add((c - b'0') as i64);
    }
    Some(if neg { -n } else { n })
}

fn ground_truth(data: &[u8], info_key: &str) -> Truth {
    let mut t = Truth {
        records: 0,
        pos: Vec::new(),
        key: Vec::new(),
    };
    let needle = info_key.as_bytes();
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        t.records += 1;
        let mut fields = line.split(|&b| b == b'\t');
        let _chrom = fields.next();
        t.pos.push(fields.next().and_then(scalar_i64).unwrap_or(0));
        let info = fields.nth(5).unwrap_or(b""); // fields[7] overall (0=CHROM..7=INFO)
        // scan INFO for `key=value`
        let mut found = None;
        for kv in info.split(|&b| b == b';') {
            if let Some(eq) = kv.iter().position(|&b| b == b'=')
                && &kv[..eq] == needle
            {
                found = scalar_i64(&kv[eq + 1..]);
                break;
            }
        }
        t.key.push(found);
    }
    t
}

/// falx generic INFO extraction (structural parse → field 7 → single-pass scan),
/// the same shape the generated kernel emits.
fn falx_info(data: &[u8], info_key: &str) -> Vec<Option<i64>> {
    let needle = info_key.as_bytes();
    let parsed = vcf::parse(data);
    let mut out = Vec::new();
    for rec in parsed.records() {
        let info = rec.field_raw(7).unwrap_or(b"");
        let mut got = None;
        let mut i = 0;
        while i < info.len() {
            let ks = i;
            while i < info.len() && info[i] != b'=' && info[i] != b';' {
                i += 1;
            }
            let key = &info[ks..i];
            let mut val: &[u8] = b"";
            if i < info.len() && info[i] == b'=' {
                i += 1;
                let vs = i;
                while i < info.len() && info[i] != b';' {
                    i += 1;
                }
                val = &info[vs..i];
            }
            if key == needle {
                got = scalar_i64(val);
                break;
            }
            if i < info.len() && info[i] == b';' {
                i += 1;
            }
        }
        out.push(got);
    }
    out
}

/// Fused .vcf.gz → POS column via the library driver (exercises the straddler/
/// line-ownership boundary handling on real bgzf block boundaries).
fn fused_pos(comp: &[u8], threads: usize) -> Vec<i64> {
    bgzf::parse_gz_par(comp, threads, b'\n', |s| vcf_typed::parse_columns(s).pos)
        .expect("fusion")
        .into_iter()
        .flatten()
        .collect()
}

fn noodles_decompress(comp: &[u8]) -> Vec<u8> {
    let mut r = nbgzf::io::Reader::new(std::io::Cursor::new(comp));
    let mut out = Vec::new();
    r.read_to_end(&mut out).expect("noodles bgzf");
    out
}

fn check(label: &str, ok: bool) {
    println!("  [{}] {label}", if ok { "PASS" } else { "FAIL" });
    assert!(ok, "validation failed: {label}");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: validate_vcf_gz <file.vcf.gz> [INFO_KEY]");
    let info_key = args.next().unwrap_or_else(|| "ALLELEID".to_string());
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());

    let comp = std::fs::read(&path).expect("read .vcf.gz");
    println!(
        "file: {path} ({:.1} MiB compressed)\n",
        comp.len() as f64 / 1048576.0
    );

    // 1. bgzf byte-exactness on real htslib framing.
    println!("bgzf (real htslib framing):");
    let serial = bgzf::decompress(&comp).expect("falx bgzf serial");
    let parallel = bgzf::decompress_par(&comp, threads).expect("falx bgzf par");
    let reference = noodles_decompress(&comp);
    check("falx serial == falx parallel", serial == parallel);
    check("falx == noodles bgzf (byte-exact)", serial == reference);
    let raw = serial;
    println!("  decompressed {:.1} MiB\n", raw.len() as f64 / 1048576.0);

    // 2. structural parse + typed columns vs independent scalar ground truth.
    let truth = ground_truth(&raw, &info_key);
    let cols = vcf_typed::parse_columns(&raw);
    println!("structural / typed columns (vs scalar ground truth):");
    check(
        &format!("record count == {} ", truth.records),
        cols.rows == truth.records,
    );
    check("POS column matches", cols.pos == truth.pos);
    let dp_present = (0..cols.rows)
        .filter(|&r| vcf_typed::bitmap_get(&cols.dp_valid, r))
        .count();
    check(
        "DP/AF absent on this file → all null (absent-key path)",
        dp_present == 0,
    );
    println!();

    // 3. INFO extraction on real INFO vs ground truth.
    println!("INFO key '{info_key}' (real multi-key INFO):");
    let fx = falx_info(&raw, &info_key);
    let present = truth.key.iter().filter(|v| v.is_some()).count();
    check(
        &format!(
            "falx INFO scan == ground truth ({present}/{} present)",
            truth.records
        ),
        fx == truth.key,
    );
    println!();

    // 4. Fusion on real bgzf block boundaries (straddler handling).
    println!("fusion (real block boundaries):");
    let fp = fused_pos(&comp, threads);
    check("fused POS == ground truth POS", fp == truth.pos);
    println!();

    // 5. Throughput sanity (best of 3).
    let u = raw.len();
    let mibs = |f: &dyn Fn()| {
        let mut b = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            f();
            b = b.min(t.elapsed().as_secs_f64());
        }
        u as f64 / b / 1048576.0
    };
    let fused = mibs(&|| {
        std::hint::black_box(fused_pos(&comp, threads));
    });
    let nood = mibs(&|| {
        let r = noodles_decompress(&comp);
        std::hint::black_box(ground_truth(&r, &info_key).records);
    });
    println!(
        "throughput (uncompressed MiB/s): fused ~{fused:.0} | noodles-decompress+scan ~{nood:.0}"
    );

    println!("\nALL REAL-DATA CHECKS PASSED ✓");
}
