//! SOTA head-to-head: `.vcf.gz` → typed columns (POS int + ALLELEID int), the
//! exact same task the htslib C harness (`/tmp/hts_extract.c`) runs, on the same
//! real ClinVar file. Emits the same checksum so the two engines can be verified
//! byte-for-byte equivalent.
//!
//! - **falx 1-thread (engine)**: serial bgzf inflate + structural parse + INFO
//!   scan. Apples-to-apples vs htslib's single-threaded `bcf_read` loop.
//! - **falx N-thread (pipeline)**: fused decompress→parse across bgzf block
//!   groups (htslib cannot parallelize the VCF text parse, only decompression).
//!
//! Run: `cargo run --release --features bgzf --example vcf_clinvar_sota -- <file.vcf.gz> [iters] [threads]`

use std::time::{Duration, Instant};

use falx::bgzf;
use falx::kernels::vcf;

/// Parse an ASCII integer (optionally signed); `None` if not a clean integer.
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

#[derive(Default, Clone, Copy)]
struct Sums {
    records: i64,
    present: i64,
    sum_alleleid: i64,
    sum_pos: u64,
}

/// Structural parse of a decompressed VCF buffer → POS + ALLELEID typed columns,
/// reduced to the same checksum the C harness emits. This is the falx kernel work:
/// SIMD structural index → field_raw(1)/field_raw(7) → single-pass INFO scan.
///
/// `early`: stop the INFO scan once ALLELEID is found (how a real columnar
/// extractor behaves). When false, walk the *entire* INFO field every record —
/// a conservative number that removes the lazy-extraction advantage and isolates
/// raw structural-parse throughput (htslib always materializes all INFO fields).
fn extract(data: &[u8], early: bool) -> Sums {
    let needle = b"ALLELEID";
    let parsed = vcf::parse(data);
    let mut s = Sums::default();
    for rec in parsed.records() {
        s.records += 1;
        if let Some(pos) = rec.field_raw(1).and_then(scalar_i64) {
            s.sum_pos = s.sum_pos.wrapping_add(pos as u64);
        }
        let info = rec.field_raw(7).unwrap_or(b"");
        let mut i = 0;
        let mut found = false;
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
            if key == needle && !found {
                if let Some(v) = scalar_i64(val) {
                    s.present += 1;
                    s.sum_alleleid = s.sum_alleleid.wrapping_add(v);
                }
                found = true;
                if early {
                    break;
                }
            }
            if i < info.len() && info[i] == b';' {
                i += 1;
            }
        }
    }
    s
}

fn merge(mut a: Sums, b: Sums) -> Sums {
    a.records += b.records;
    a.present += b.present;
    a.sum_alleleid = a.sum_alleleid.wrapping_add(b.sum_alleleid);
    a.sum_pos = a.sum_pos.wrapping_add(b.sum_pos);
    a
}

/// Single-threaded engine: serial inflate, then a single-pass parse+extract.
fn falx_1t(comp: &[u8], early: bool) -> Sums {
    let buf = bgzf::decompress(comp).expect("bgzf serial");
    extract(&buf, early)
}

/// Fused multi-threaded pipeline: each worker inflates its own bgzf block group
/// and parses it locally; per-worker checksums are merged.
fn falx_nt(comp: &[u8], threads: usize) -> Sums {
    bgzf::parse_gz_par(comp, threads, b'\n', |s| extract(s, true))
        .expect("fusion")
        .into_iter()
        .fold(Sums::default(), merge)
}

fn best(iters: usize, mut f: impl FnMut() -> Sums) -> (Duration, Sums) {
    let mut b = Duration::MAX;
    let mut last = Sums::default();
    for _ in 0..iters {
        let t = Instant::now();
        last = std::hint::black_box(f());
        b = b.min(t.elapsed());
    }
    (b, last)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .expect("usage: vcf_clinvar_sota <file.vcf.gz> [iters] [threads]");
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let threads: usize = a
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()));

    let comp = std::fs::read(&path).expect("read .vcf.gz");
    // Uncompressed size (one serial inflate, untimed) for MiB/s.
    let u = bgzf::decompress(&comp).expect("bgzf").len();
    println!(
        "file: {path}  ({:.1} MiB compressed → {:.1} MiB uncompressed)  | {threads} threads, best of {iters}\n",
        comp.len() as f64 / 1048576.0,
        u as f64 / 1048576.0,
    );

    let (t1, s1) = best(iters, || falx_1t(&comp, true));
    let (tf, _sf) = best(iters, || falx_1t(&comp, false));
    let (tn, sn) = best(iters, || falx_nt(&comp, threads));

    // Same checksum the C harness prints — compare these across engines.
    println!(
        "CHECKSUM records={} present={} sum_alleleid={} sum_pos={}",
        s1.records, s1.present, s1.sum_alleleid, s1.sum_pos
    );
    assert_eq!(s1.records, sn.records, "1t vs Nt record count diverges");
    assert_eq!(
        s1.sum_alleleid, sn.sum_alleleid,
        "1t vs Nt ALLELEID diverges"
    );
    assert_eq!(s1.sum_pos, sn.sum_pos, "1t vs Nt POS diverges");

    let mibs = |dt: Duration| u as f64 / dt.as_secs_f64() / 1048576.0;
    let rps = |dt: Duration| s1.records as f64 / dt.as_secs_f64();
    println!("\n.vcf.gz → POS+ALLELEID typed columns:");
    println!(
        "  falx 1-thread (early-exit) : {:>7.1} MiB/s  {:>10.0} rec/s  ({:.3}s)",
        mibs(t1),
        rps(t1),
        t1.as_secs_f64()
    );
    println!(
        "  falx 1-thread (full INFO)  : {:>7.1} MiB/s  {:>10.0} rec/s  ({:.3}s)",
        mibs(tf),
        rps(tf),
        tf.as_secs_f64()
    );
    println!(
        "  falx {threads}-thread (fused)    : {:>7.1} MiB/s  {:>10.0} rec/s  ({:.3}s)",
        mibs(tn),
        rps(tn),
        tn.as_secs_f64()
    );
}
