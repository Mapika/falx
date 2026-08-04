//! Minimal single-lane driver for `perf record`: runs exactly one falx path
//! in a loop so profiles contain no other contenders.
//!
//! Usage: profile_lane <file> <lane> [threads] [reps]
//! Lanes: geo-serial | geo-par | geo-chunks | geo-stats | geo-stats-par
//!        text-serial | text-par | text-chunks

use std::hint::black_box;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: profile_lane <file> <lane> [threads] [reps]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).expect("read input");
    let lane = args[2].as_str();
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);
    let reps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);

    let gib = data.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        match lane {
            "geo-serial" => {
                black_box(falx::kernels::csv_geo::parse_columns(black_box(&data)));
            }
            "geo-par" => {
                black_box(falx::kernels::csv_geo::parse_columns_par(
                    black_box(&data),
                    threads,
                ));
            }
            "geo-chunks" => {
                black_box(falx::kernels::csv_geo::parse_columns_chunks_par(
                    black_box(&data),
                    threads,
                ));
            }
            "geo-stats" => {
                black_box(falx::kernels::csv_geo::parse_csv_geo_stats(black_box(
                    &data,
                )));
            }
            "geo-stats-par" => {
                black_box(falx::kernels::csv_geo::parse_csv_geo_stats_par(
                    black_box(&data),
                    threads,
                ));
            }
            "text-serial" => {
                black_box(falx::kernels::csv_geo_text::parse_columns(black_box(&data)));
            }
            "text-par" => {
                black_box(falx::kernels::csv_geo_text::parse_columns_par(
                    black_box(&data),
                    threads,
                ));
            }
            "text-chunks" => {
                black_box(falx::kernels::csv_geo_text::parse_columns_chunks_par(
                    black_box(&data),
                    threads,
                ));
            }
            other => {
                eprintln!("unknown lane: {other}");
                std::process::exit(2);
            }
        }
        let s = t.elapsed().as_secs_f64();
        if s < best {
            best = s;
        }
    }
    println!("{lane}: best {:.1} ms  {:.2} GiB/s", best * 1e3, gib / best);
}
