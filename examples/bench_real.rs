use std::hint::black_box;
use std::io;
use std::time::Instant;

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: cargo run --release --example bench_real -- <file.csv>");
        std::process::exit(1);
    }

    let file_path = &args[1];
    let data = std::fs::read(file_path)?;
    let file_size = data.len();

    println!("File: {}", file_path);
    println!(
        "Size: {} bytes ({:.2} MiB)",
        file_size,
        file_size as f64 / (1024.0 * 1024.0)
    );
    println!();

    // Warm up + 5 runs
    const RUNS: usize = 5;
    const WARMUP: usize = 1;

    // ========== falx::kernels::csv::index_structurals ==========
    let mut best_index_ms = f64::INFINITY;
    for _ in 0..WARMUP {
        let mut out = Vec::new();
        let _start = Instant::now();
        falx::kernels::csv::index_structurals(black_box(&data), &mut out);
        let _ = black_box(out);
    }
    for _ in 0..RUNS {
        let mut out = Vec::new();
        let start = Instant::now();
        falx::kernels::csv::index_structurals(black_box(&data), &mut out);
        let _ = black_box(out);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        best_index_ms = best_index_ms.min(elapsed);
    }

    // ========== falx::kernels::csv::parse + iterate ==========
    let mut best_parse_ms = f64::INFINITY;
    let mut falx_field_bytes: u64 = 0;
    for _ in 0..WARMUP {
        let parsed = falx::kernels::csv::parse(black_box(&data));
        let _ = black_box(&parsed);
    }
    for _ in 0..RUNS {
        let start = Instant::now();
        let parsed = falx::kernels::csv::parse(black_box(&data));
        let mut field_bytes = 0u64;
        for record in parsed.records() {
            for field in record.fields() {
                field_bytes += field.len() as u64;
            }
        }
        falx_field_bytes = black_box(field_bytes);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        best_parse_ms = best_parse_ms.min(elapsed);
    }

    // ========== csv crate ==========
    let mut best_csv_crate_ms = f64::INFINITY;
    let mut csv_field_bytes: u64 = 0;
    for _ in 0..WARMUP {
        let reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(black_box(&data[..]));
        let _ = black_box(reader);
    }
    for _ in 0..RUNS {
        let start = Instant::now();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(black_box(&data[..]));
        let mut field_bytes = 0u64;
        for record in reader.byte_records().flatten() {
            for field in record.iter() {
                field_bytes += field.len() as u64;
            }
        }
        csv_field_bytes = black_box(field_bytes);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        best_csv_crate_ms = best_csv_crate_ms.min(elapsed);
    }

    // ========== Print results ==========
    let gib = file_size as f64 / (1024.0 * 1024.0 * 1024.0);
    let index_throughput = gib / (best_index_ms / 1000.0);
    let parse_throughput = gib / (best_parse_ms / 1000.0);
    let csv_throughput = gib / (best_csv_crate_ms / 1000.0);

    let speedup_index = best_csv_crate_ms / best_index_ms;
    let speedup_parse = best_csv_crate_ms / best_parse_ms;

    println!("Benchmark Results (best of {} runs):", RUNS);
    println!();
    println!(
        "falx::kernels::csv::index_structurals:  {:.3} ms / {:.2} GiB/s ({:.2}x vs csv)",
        best_index_ms, index_throughput, speedup_index
    );
    println!(
        "falx::kernels::csv::parse + iterate:    {:.3} ms / {:.2} GiB/s ({:.2}x vs csv)",
        best_parse_ms, parse_throughput, speedup_parse
    );
    println!(
        "csv crate (ReaderBuilder):                {:.3} ms / {:.2} GiB/s",
        best_csv_crate_ms, csv_throughput
    );
    println!();
    println!("Field byte totals:");
    println!("  falx:     {}", falx_field_bytes);
    println!("  csv crate: {}", csv_field_bytes);
    if falx_field_bytes != csv_field_bytes {
        eprintln!(
            "WARNING: Field byte totals differ! falx={}, csv={}",
            falx_field_bytes, csv_field_bytes
        );
    }

    Ok(())
}
