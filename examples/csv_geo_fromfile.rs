//! Apples-to-apples vs DuckDB/polars `read_csv`: time the *whole* path from a CSV
//! file on disk (page-cached) to the aggregate count(*)/sum(Latitude)/sum(Longitude),
//! since those engines include the file read. We bracket it three ways:
//!   read-only | parse-only (in-RAM) | read+parse (end-to-end, matches read_csv)
//! falx additionally materializes the City string column (DuckDB/polars-lazy project
//! it away) — so falx is doing strictly more work here.
//!
//! Run: cargo run --release --example csv_geo_fromfile -- <file.csv> [iters] [threads]

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
        .expect("usage: csv_geo_fromfile <file.csv> [iters] [threads]");
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(7);
    let threads: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(24);

    let data = std::fs::read(&path).expect("read csv");
    let gib = data.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    let (vc, slat, slon) = agg(&csv_geo::parse_columns(&data));
    println!(
        "file: {path}  ({:.2} GiB)  | {threads} threads, best of {iters}",
        gib
    );
    println!("CHECKSUM count={vc} sum_lat={slat:.2} sum_lon={slon:.2}\n");

    // read-only (single-threaded std::fs::read from page cache)
    let t_read = best(iters, || {
        let t = Instant::now();
        let v = std::fs::read(&path).expect("read");
        black_box(&v);
        t.elapsed()
    });

    // parse-only (in-RAM), serial + parallel
    let t_p1 = best(iters, || {
        let t = Instant::now();
        black_box(agg(&csv_geo::parse_columns(black_box(&data))));
        t.elapsed()
    });
    let t_pn = best(iters, || {
        let t = Instant::now();
        black_box(agg(&csv_geo::parse_columns_par(black_box(&data), threads)));
        t.elapsed()
    });

    // read+parse end-to-end (matches read_csv), serial + parallel
    let t_rp1 = best(iters, || {
        let t = Instant::now();
        let d = std::fs::read(&path).expect("read");
        black_box(agg(&csv_geo::parse_columns(&d)));
        t.elapsed()
    });
    let t_rpn = best(iters, || {
        let t = Instant::now();
        let d = std::fs::read(&path).expect("read");
        black_box(agg(&csv_geo::parse_columns_par(&d, threads)));
        t.elapsed()
    });

    let g = |dt: Duration| gib / dt.as_secs_f64();
    let ms = |dt: Duration| dt.as_secs_f64() * 1000.0;
    println!("                              best(ms)   GiB/s");
    println!(
        "  read-only (1 thread)      : {:>8.1}  {:>6.2}",
        ms(t_read),
        g(t_read)
    );
    println!(
        "  parse-only      1 thread  : {:>8.1}  {:>6.2}",
        ms(t_p1),
        g(t_p1)
    );
    println!(
        "  parse-only    {threads:>2} threads  : {:>8.1}  {:>6.2}",
        ms(t_pn),
        g(t_pn)
    );
    println!(
        "  read+parse      1 thread  : {:>8.1}  {:>6.2}",
        ms(t_rp1),
        g(t_rp1)
    );
    println!(
        "  read+parse    {threads:>2} threads  : {:>8.1}  {:>6.2}",
        ms(t_rpn),
        g(t_rpn)
    );
}
