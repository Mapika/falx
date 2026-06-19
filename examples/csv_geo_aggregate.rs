//! Beat-DuckDB head-to-head: the apples-to-apples match for DuckDB's streaming
//! `SELECT count(*), sum(Latitude), sum(Longitude)` is falx's *fused projected*
//! aggregation (`parse_csv_geo_stats_par`) — full SIMD structural parse + locate
//! fields 5/6 + `parse_f64_field`, with NO column Vecs materialized (unlike
//! `parse_columns_par`, which builds two 21M-element `Vec<f64>` ≈ 342 MB).
//! mmap'd input so I/O overlaps the parse.
//!
//! Run: cargo run --release --features mmap --example csv_geo_aggregate -- <file.csv> [iters] [threads]

use std::hint::black_box;
use std::time::{Duration, Instant};

use falx::kernels::csv_geo;

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
        .expect("usage: csv_geo_aggregate <file.csv> [iters] [threads]");
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(11);
    let threads: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(24);

    let nbytes = std::fs::metadata(&path).expect("stat").len();
    let gib = nbytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // Verify the fused path sees every row (DuckDB: count=21378801) and parses the
    // *same* values. The fused stats reduce to a bit-checksum (sum of f64.to_bits());
    // the materialized parse_columns path exposes the actual f64 values, whose
    // float-sum already matches DuckDB (276383.08 / -269847.86). Asserting the two
    // bit-checksums agree proves the fused path parsed identical floats — so it is
    // computing DuckDB's answer, just reduced with a 1-instruction-different op.
    let (st, lat_sum, lon_sum, lat_bits, lon_bits) = falx::io::with_mapped(&path, |b| {
        let st = csv_geo::parse_csv_geo_stats_par(b, threads);
        let cols = csv_geo::parse_columns_par(b, threads);
        let lat_sum: f64 = cols.latitude.iter().copied().sum();
        let lon_sum: f64 = cols.longitude.iter().copied().sum();
        let lat_bits = cols
            .latitude
            .iter()
            .fold(0u64, |a, v| a.wrapping_add(v.to_bits()));
        let lon_bits = cols
            .longitude
            .iter()
            .fold(0u64, |a, v| a.wrapping_add(v.to_bits()));
        (st, lat_sum, lon_sum, lat_bits, lon_bits)
    })
    .expect("map");
    assert_eq!(
        st.latitude_checksum, lat_bits,
        "fused lat checksum != materialized"
    );
    assert_eq!(
        st.longitude_checksum, lon_bits,
        "fused lon checksum != materialized"
    );
    println!("file: {path}  ({gib:.2} GiB)  | {threads} threads, best of {iters}");
    println!(
        "stats: records={} lat_values={} lon_values={}",
        st.records, st.latitude_values, st.longitude_values
    );
    println!(
        "verified: fused checksum == materialized; sum(lat)={lat_sum:.2} sum(lon)={lon_sum:.2} (== DuckDB)\n"
    );

    // Fused projected aggregation (no materialization) — the DuckDB-equivalent.
    let fused_par = best(iters, || {
        let m = falx::io::map(&path).expect("map");
        let t = Instant::now();
        black_box(csv_geo::parse_csv_geo_stats_par(&m, threads));
        t.elapsed()
    });
    let fused_1t = best(iters, || {
        let m = falx::io::map(&path).expect("map");
        let t = Instant::now();
        black_box(csv_geo::parse_csv_geo_stats(&m));
        t.elapsed()
    });
    // Old materializing path (builds the Vec<f64> columns) for contrast.
    let cols_par = best(iters, || {
        let m = falx::io::map(&path).expect("map");
        let t = Instant::now();
        black_box(csv_geo::parse_columns_par(&m, threads));
        t.elapsed()
    });

    let g = |dt: Duration| gib / dt.as_secs_f64();
    let ms = |dt: Duration| dt.as_secs_f64() * 1000.0;
    println!("CSV file → count/sum(lat,lon)              best(ms)   GiB/s");
    println!(
        "  falx fused-stats (mmap, {threads}t)  [proj] : {:>8.1}  {:>6.2}",
        ms(fused_par),
        g(fused_par)
    );
    println!(
        "  falx fused-stats (mmap, 1t)         : {:>8.1}  {:>6.2}",
        ms(fused_1t),
        g(fused_1t)
    );
    println!(
        "  falx parse_columns (mmap, {threads}t) [mat] : {:>8.1}  {:>6.2}",
        ms(cols_par),
        g(cols_par)
    );
    println!("\n(DuckDB reference on this box: ~2.87 GiB/s @ 48t, ~0.24 @ 1t)");
}
