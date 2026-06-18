//! Block-parallel bgzf decompression: falx vs noodles-bgzf.
//!
//! Decompression is the wall in front of the SIMD parsers on `.vcf.gz` inputs
//! (single-thread gunzip ~0.3 GiB/s vs a multi-GiB/s parser). bgzf blocks are
//! independent, so this measures how well falx's block-parallel inflate (a
//! boundary scan + scatter into disjoint slots, pure-Rust miniz_oxide core)
//! feeds the parser, against the noodles-bgzf reader the genomics tools use.
//!
//! Run: `cargo run --release --features bgzf --example bgzf_bench -- [uncompressed_MiB] [iters]`
//! (noodles-bgzf is a dev-dependency, so this example always has the baseline.)

use std::io::{Read as _, Write as _};
use std::num::NonZero;
use std::time::{Duration, Instant};

use noodles_bgzf as bgzf;

/// Deterministic xorshift64* — keeps the generated VCF stable run to run.
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
}

/// Generate roughly `target` bytes of realistic VCF text.
fn make_vcf(target: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(target + 4096);
    out.extend_from_slice(b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");
    let mut rng = Rng(0x1234_5678_9abc_def1);
    const BASES: &[u8] = b"ACGT";
    let mut row = 0u64;
    while out.len() < target {
        row += 1;
        let _ = writeln!(
            out,
            "chr{}\t{}\trs{row}\t{}\t{}\t{:.1}\tPASS\tAC={};AF={:.4};AN={};DP={};MQ={:.2};QD={:.2};FS={:.3};SOR={:.3}",
            1 + rng.below(22),
            1 + rng.below(250_000_000),
            BASES[rng.below(4) as usize] as char,
            BASES[rng.below(4) as usize] as char,
            10.0 + (rng.below(9000) as f64) / 100.0,
            1 + rng.below(4),
            (rng.below(10000) as f64) / 10000.0,
            2 + rng.below(6),
            10 + rng.below(500),
            40.0 + (rng.below(2000) as f64) / 100.0,
            (rng.below(3500) as f64) / 100.0,
            (rng.below(30000) as f64) / 1000.0,
            (rng.below(3000) as f64) / 1000.0,
        );
    }
    out
}

/// bgzip `data` with noodles (the format other tools read/write).
fn bgzip(data: &[u8]) -> Vec<u8> {
    let mut w = bgzf::io::Writer::new(Vec::new());
    w.write_all(data).expect("bgzf write");
    w.finish().expect("bgzf finish")
}

fn noodles_decompress(comp: &[u8]) -> Vec<u8> {
    let mut r = bgzf::io::Reader::new(std::io::Cursor::new(comp));
    let mut out = Vec::new();
    r.read_to_end(&mut out).expect("noodles bgzf read");
    out
}

// Takes an owned buffer: the multithreaded reader owns its source across the
// worker threads (`R: 'static`). Callers clone outside the timed region.
fn noodles_decompress_par(comp: Vec<u8>, threads: usize) -> Vec<u8> {
    let workers = NonZero::new(threads).unwrap_or(NonZero::<usize>::MIN);
    let mut r =
        bgzf::io::MultithreadedReader::with_worker_count(workers, std::io::Cursor::new(comp));
    let mut out = Vec::new();
    r.read_to_end(&mut out).expect("noodles bgzf mt read");
    out
}

fn best(iters: usize, mut f: impl FnMut()) -> Duration {
    let mut b = Duration::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed());
    }
    b
}

fn mibs(bytes: usize, dt: Duration) -> f64 {
    (bytes as f64) / dt.as_secs_f64() / (1024.0 * 1024.0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mib: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(192);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());

    let raw = make_vcf(mib * 1024 * 1024);
    let comp = bgzip(&raw);
    println!(
        "uncompressed {:.1} MiB  →  bgzf {:.1} MiB ({:.1}x)  |  {threads} threads, best of {iters}\n",
        raw.len() as f64 / 1048576.0,
        comp.len() as f64 / 1048576.0,
        raw.len() as f64 / comp.len() as f64,
    );

    // Correctness: every path reproduces the original bytes exactly.
    assert_eq!(
        falx::bgzf::decompress(&comp).unwrap(),
        raw,
        "falx serial mismatch"
    );
    assert_eq!(
        falx::bgzf::decompress_par(&comp, threads).unwrap(),
        raw,
        "falx parallel mismatch"
    );
    assert_eq!(noodles_decompress(&comp), raw, "noodles serial mismatch");
    assert_eq!(
        noodles_decompress_par(comp.clone(), threads),
        raw,
        "noodles parallel mismatch"
    );
    println!("correctness: all paths reproduce the input byte-for-byte\n");

    let u = raw.len();
    let fs = best(iters, || {
        std::hint::black_box(falx::bgzf::decompress(&comp).unwrap());
    });
    let fp = best(iters, || {
        std::hint::black_box(falx::bgzf::decompress_par(&comp, threads).unwrap());
    });
    let ns = best(iters, || {
        std::hint::black_box(noodles_decompress(&comp));
    });
    // Clone the compressed buffer outside the timer (the MT reader consumes an
    // owned source), so only the decompression itself is measured.
    let np = {
        let mut b = Duration::MAX;
        for _ in 0..iters {
            let owned = comp.clone();
            let t = Instant::now();
            std::hint::black_box(noodles_decompress_par(owned, threads));
            b = b.min(t.elapsed());
        }
        b
    };

    println!("decompression throughput (uncompressed MiB/s):");
    println!("  falx    serial   : {:>8.1}", mibs(u, fs));
    println!(
        "  falx    parallel : {:>8.1}   ({:.2}x over its serial)",
        mibs(u, fp),
        mibs(u, fp) / mibs(u, fs)
    );
    println!("  noodles serial   : {:>8.1}", mibs(u, ns));
    println!("  noodles parallel : {:>8.1}", mibs(u, np));
    println!(
        "\nfalx parallel vs noodles parallel : {:.2}x | vs noodles serial : {:.2}x",
        mibs(u, fp) / mibs(u, np),
        mibs(u, fp) / mibs(u, ns),
    );
}
