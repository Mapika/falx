//! csv_hash typed-column projection: serial vs the region-parallel path.
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: bench_csv_hash_columns <csv-hash file> [threads] [reps]");
        std::process::exit(2)
    });
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let data = std::fs::read(&path).expect("read input");
    let gib = data.len() as f64 / (1024.0 * 1024.0 * 1024.0);

    let want = falx::kernels::csv_hash::parse_columns(&data);
    let got = falx::kernels::csv_hash::parse_columns_par(&data, threads);
    assert_eq!(want.rows, got.rows, "parallel row count matches serial");
    let sum = |c: &falx::kernels::csv_hash::Columns<'_>| -> i64 {
        (0..c.rows)
            .map(|r| c.amount[r])
            .fold(0i64, i64::wrapping_add)
    };
    assert_eq!(sum(&want), sum(&got), "parallel checksum matches serial");
    println!(
        "verified: rows={} amount_checksum={}",
        want.rows,
        sum(&want)
    );

    let bench = |label: &str, f: &mut dyn FnMut() -> usize| {
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            black_box(f());
            let s = t.elapsed().as_secs_f64();
            best = best.min(s);
        }
        println!(
            "  {label:38} {:8.1} ms  {:6.2} GiB/s",
            best * 1e3,
            gib / best
        );
    };
    println!("\ncsv_hash typed columns (key + amount), {gib:.2} GiB, {threads} threads");
    bench("parse_columns (serial)", &mut || {
        falx::kernels::csv_hash::parse_columns(black_box(&data)).rows
    });
    bench("parse_columns_par", &mut || {
        falx::kernels::csv_hash::parse_columns_par(black_box(&data), threads).rows
    });
    bench("parse_columns_chunks_par", &mut || {
        falx::kernels::csv_hash::parse_columns_chunks_par(black_box(&data), threads).len()
    });
}
