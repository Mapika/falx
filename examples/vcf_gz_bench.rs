//! End-to-end `.vcf.gz` → typed Arrow columns: decompress→parse **fusion** vs a
//! staged pipeline vs the noodles incumbent.
//!
//! - **fused**: each worker inflates its own contiguous bgzf block group, finishes
//!   the record straddling its tail by inflating into the next group's first
//!   block up to a newline, drops its leading partial line (line ownership), and
//!   runs the generated `vcf_typed::parse_columns` on the local buffer. No
//!   full-file decompressed buffer is ever materialized; decompressed bytes are
//!   parsed while hot in cache.
//! - **staged**: `bgzf::decompress_par` into one buffer, then `parse_columns_par`.
//! - **noodles**: the VCF reader over a multithreaded bgzf reader (what oxbow/
//!   exon/biobear/polars-bio do), extracting the same DP column.
//!
//! Run: `cargo run --release --features bgzf --example vcf_gz_bench -- [MiB] [iters] [threads]`

use std::io::Write as _;
use std::num::NonZero;
use std::time::{Duration, Instant};

use falx::bgzf;
use falx::kernels::vcf_typed;
use noodles_bgzf as nbgzf;
use noodles_vcf::variant::record::info::field::Value;

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D)) % n
    }
}

const HEADER: &str = "\
##fileformat=VCFv4.3
##INFO=<ID=DP,Number=1,Type=Integer,Description=\"\">
##INFO=<ID=AF,Number=A,Type=Float,Description=\"\">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
";

fn make_vcf(target: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(target + 4096);
    out.extend_from_slice(HEADER.as_bytes());
    let mut rng = Rng(0x1234_5678_9abc_def1);
    const BASES: &[u8] = b"ACGT";
    let mut row = 0u64;
    while out.len() < target {
        row += 1;
        let drop_dp = rng.below(20) == 0;
        let _ = write!(
            out,
            "chr{}\t{}\trs{row}\t{}\t{}\t{:.1}\tPASS\t",
            1 + rng.below(22),
            1 + rng.below(250_000_000),
            BASES[rng.below(4) as usize] as char,
            BASES[rng.below(4) as usize] as char,
            10.0 + (rng.below(9000) as f64) / 100.0,
        );
        if drop_dp {
            let _ = writeln!(out, "AF={:.4}", (rng.below(10000) as f64) / 10000.0);
        } else {
            let _ = writeln!(
                out,
                "DP={};AF={:.4}",
                10 + rng.below(500),
                (rng.below(10000) as f64) / 10000.0
            );
        }
    }
    out
}

fn bgzip(data: &[u8]) -> Vec<u8> {
    let mut w = nbgzf::io::Writer::new(Vec::new());
    w.write_all(data).expect("bgzf write");
    w.finish().expect("bgzf finish")
}

/// DP column (values + validity) lifted out of the generated columns.
fn dp_of(cols: &vcf_typed::Columns) -> (Vec<i64>, Vec<bool>) {
    let valid = (0..cols.rows)
        .map(|r| vcf_typed::bitmap_get(&cols.dp_valid, r))
        .collect();
    (cols.dp.clone(), valid)
}

/// Staged: decompress everything in parallel, then parse the buffer in parallel.
fn staged(comp: &[u8], threads: usize) -> (Vec<i64>, Vec<bool>) {
    let buf = bgzf::decompress_par(comp, threads).expect("bgzf");
    dp_of(&vcf_typed::parse_columns_par(&buf, threads))
}

/// Fused via the library driver: each worker decompresses its block group and
/// parses it locally; we concatenate the per-worker DP columns.
fn fused(comp: &[u8], threads: usize) -> (Vec<i64>, Vec<bool>) {
    let parts = bgzf::parse_gz_par(comp, threads, b'\n', |s| {
        dp_of(&vcf_typed::parse_columns(s))
    })
    .expect("fusion");
    let mut dp = Vec::new();
    let mut valid = Vec::new();
    for (d, v) in parts {
        dp.extend_from_slice(&d);
        valid.extend_from_slice(&v);
    }
    (dp, valid)
}

/// noodles incumbent: VCF reader over a multithreaded bgzf reader.
fn noodles_gz(comp: Vec<u8>, threads: usize) -> (Vec<i64>, Vec<bool>) {
    let workers = NonZero::new(threads).unwrap_or(NonZero::<usize>::MIN);
    let bgzf_reader =
        nbgzf::io::MultithreadedReader::with_worker_count(workers, std::io::Cursor::new(comp));
    let mut reader = noodles_vcf::io::Reader::new(bgzf_reader);
    let header = reader.read_header().expect("vcf header");
    let mut dp = Vec::new();
    let mut valid = Vec::new();
    for rec in reader.records() {
        let rec = rec.expect("vcf record");
        match rec.info().get(&header, "DP") {
            Some(Ok(Some(Value::Integer(n)))) => {
                dp.push(n as i64);
                valid.push(true);
            }
            _ => {
                dp.push(0);
                valid.push(false);
            }
        }
    }
    (dp, valid)
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

fn main() {
    let mut a = std::env::args().skip(1);
    let mib: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(192);
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let threads: usize = a
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()));

    let raw = make_vcf(mib * 1024 * 1024);
    let comp = bgzip(&raw);
    let u = raw.len();
    println!(
        "uncompressed {:.1} MiB → bgzf {:.1} MiB ({:.1}x) | {threads} threads, best of {iters}\n",
        u as f64 / 1048576.0,
        comp.len() as f64 / 1048576.0,
        u as f64 / comp.len() as f64,
    );

    // Correctness: fused and staged agree with noodles on the whole DP column.
    let nd = noodles_gz(comp.clone(), threads);
    let st = staged(&comp, threads);
    let fz = fused(&comp, threads);
    assert_eq!(st, nd, "staged DP column diverges from noodles");
    assert_eq!(fz, nd, "fused DP column diverges from noodles");
    let present = nd.1.iter().filter(|&&b| b).count();
    println!(
        "correctness: fused == staged == noodles on DP ({} rows, {present} present)\n",
        nd.0.len()
    );

    let mibs = |dt: Duration| u as f64 / dt.as_secs_f64() / 1048576.0;
    let f_t = best(iters, || {
        std::hint::black_box(fused(&comp, threads));
    });
    let s_t = best(iters, || {
        std::hint::black_box(staged(&comp, threads));
    });
    let n_t = {
        let mut b = Duration::MAX;
        for _ in 0..iters {
            let owned = comp.clone();
            let t = Instant::now();
            std::hint::black_box(noodles_gz(owned, threads));
            b = b.min(t.elapsed());
        }
        b
    };

    println!(".vcf.gz → DP/AF typed Arrow columns (uncompressed MiB/s):");
    println!("  fused   : {:>7.1}", mibs(f_t));
    println!("  staged  : {:>7.1}", mibs(s_t));
    println!("  noodles : {:>7.1}", mibs(n_t));
    println!(
        "\nfused vs staged : {:.2}x | fused vs noodles : {:.2}x",
        mibs(f_t) / mibs(s_t),
        mibs(f_t) / mibs(n_t),
    );
}
