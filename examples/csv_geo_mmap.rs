//! Demonstrates `falx::io` (mmap feature): mmap the CSV and parse the mapped
//! bytes directly, so page faults fault in *during* the 24-way parallel parse —
//! overlapping I/O with compute the way DuckDB/polars' mmap readers do. Compared
//! against the naive `std::fs::read` + parse path that loses to them end-to-end.
//!
//! (A prior version also probed `madvise(SEQUENTIAL)` — a no-op here — and
//! `MAP_POPULATE`, which was *slower*: a single-threaded kernel prefault loses to
//! letting the parse threads fault their own regions. So `falx::io::map` is plain.)
//!
//! Run: cargo run --release --features mmap --example csv_geo_mmap -- <file.csv> [iters] [threads]

use std::hint::black_box;
use std::time::{Duration, Instant};

use falx::kernels::csv_geo;

fn agg(cols: &csv_geo::Columns) -> (usize, f64, f64) {
    let valid = cols
        .latitude_valid
        .iter()
        .map(|w| w.count_ones() as usize)
        .sum::<usize>();
    let slat = cols.latitude.iter().copied().sum::<f64>();
    let slon = cols.longitude.iter().copied().sum::<f64>();
    (valid, slat, slon)
}

fn best(iters: usize, mut f: impl FnMut() -> Duration) -> Duration {
    let mut b = Duration::MAX;
    for _ in 0..iters {
        b = b.min(f());
    }
    b
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .expect("usage: csv_geo_mmap <file.csv> [iters] [threads]");
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(9);
    let threads: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(24);

    let nbytes = std::fs::metadata(&path).expect("stat").len();
    let gib = nbytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let (vc, slat, slon) =
        falx::io::with_mapped(&path, |b| agg(&csv_geo::parse_columns(b))).expect("map");
    println!("file: {path}  ({gib:.2} GiB)  | {threads} threads, best of {iters}");
    println!("CHECKSUM count={vc} sum_lat={slat:.2} sum_lon={slon:.2}\n");

    // Reference: naive serial fs::read into a Vec, then parallel parse.
    let read_par = best(iters, || {
        let t = Instant::now();
        let d = std::fs::read(&path).expect("read");
        black_box(agg(&csv_geo::parse_columns_par(&d, threads)));
        t.elapsed()
    });

    // mmap + parallel parse: a fresh map each iter (matches DuckDB/polars
    // reopening per query); faults fault in across the parse threads.
    let mmap_par = best(iters, || {
        let t = Instant::now();
        let m = falx::io::map(&path).expect("map");
        black_box(agg(&csv_geo::parse_columns_par(&m, threads)));
        t.elapsed()
    });

    // Single-core end-to-end via mmap (vs DuckDB/polars 1-thread).
    let mmap_1t = best(iters, || {
        let t = Instant::now();
        let m = falx::io::map(&path).expect("map");
        black_box(agg(&csv_geo::parse_columns(&m)));
        t.elapsed()
    });

    let g = |dt: Duration| gib / dt.as_secs_f64();
    let ms = |dt: Duration| dt.as_secs_f64() * 1000.0;
    println!("end-to-end CSV file → sum(lat,lon)        best(ms)   GiB/s");
    println!(
        "  serial read + parse  ({threads}t)  [ref]    : {:>8.1}  {:>6.2}",
        ms(read_par),
        g(read_par)
    );
    println!(
        "  mmap + parse         ({threads}t)           : {:>8.1}  {:>6.2}",
        ms(mmap_par),
        g(mmap_par)
    );
    println!(
        "  mmap + parse         (1t)            : {:>8.1}  {:>6.2}",
        ms(mmap_1t),
        g(mmap_1t)
    );
}
