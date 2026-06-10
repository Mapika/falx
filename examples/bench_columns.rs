use std::hint::black_box;
use std::io;
use std::time::Instant;

/// xorshift64* RNG; avoids a dev-dependency for test data generation.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform distribution in [0, 1)
    fn uniform(&mut self) -> f64 {
        let u = self.next();
        (u >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Uniform in [a, b)
    fn range_f64(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.uniform()
    }

    /// Uniform in [0, n)
    fn range_usize(&mut self, n: usize) -> usize {
        ((self.next() as u128) * (n as u128) >> 64) as usize
    }
}

/// Generate synthetic CSV data, worldcitiespop-shaped:
/// Country,City,AccentCity,Region,Population,Latitude,Longitude
fn generate_csv(target_bytes: usize) -> Vec<u8> {
    let mut rng = Rng(0xDEAD_BEEF_CAFE_BABE);
    let mut buf = Vec::new();

    // Two-letter country codes (26 * 26 = 676 choices)
    let countries: Vec<&str> = vec![
        "US", "CA", "GB", "FR", "DE", "IT", "ES", "AU", "JP", "CN",
        "IN", "BR", "MX", "RU", "ZA", "NG", "EG", "TR", "SA", "AE",
        "KR", "SG", "NZ", "AR", "CL", "CO", "PE", "VE", "NL", "BE",
        "CH", "AT", "SE", "NO", "FI", "DK", "PL", "CZ", "HU", "RO",
        "GR", "PT", "IE", "IL", "TH", "MY", "PH", "ID", "VN", "PK",
        "BD", "IR", "IQ", "AF", "KZ", "UZ", "TM", "TJ", "KG", "AZ",
        "AM", "GE", "UA", "BY", "MD", "LT", "LV", "EE", "HR", "BA",
        "RS", "BG", "SK", "SI", "CY", "MT", "LU", "IS", "FO", "GL",
        "PA", "CR", "GT", "HN", "SV", "NI", "BZ", "CU", "DO", "HT",
    ];

    while buf.len() < target_bytes {
        // Country (2 letters)
        let country = countries[rng.range_usize(countries.len())];
        buf.extend_from_slice(country.as_bytes());
        buf.push(b',');

        // City (4-12 lowercase letters)
        let city_len = 4 + rng.range_usize(9);
        for _ in 0..city_len {
            buf.push(b'a' + (rng.range_usize(26) as u8));
        }
        buf.push(b',');

        // AccentCity (same as city)
        let city_len = 4 + rng.range_usize(9);
        for _ in 0..city_len {
            buf.push(b'a' + (rng.range_usize(26) as u8));
        }
        buf.push(b',');

        // Region (2 letters or 2 digits)
        buf.push(b'A' + (rng.range_usize(26) as u8));
        buf.push(b'A' + (rng.range_usize(26) as u8));
        buf.push(b',');

        // Population (sometimes empty, otherwise 0-99999999)
        if rng.next() & 1 != 0 {
            let pop = rng.range_usize(100_000_000);
            let pop_str = format!("{}", pop);
            buf.extend_from_slice(pop_str.as_bytes());
        }
        buf.push(b',');

        // Latitude ([-90, 90], 6 decimals)
        let lat = rng.range_f64(-90.0, 90.0);
        let lat_str = format!("{:.6}", lat);
        buf.extend_from_slice(lat_str.as_bytes());
        buf.push(b',');

        // Longitude ([-180, 180], 6 decimals)
        let lon = rng.range_f64(-180.0, 180.0);
        let lon_str = format!("{:.6}", lon);
        buf.extend_from_slice(lon_str.as_bytes());
        buf.push(b'\n');
    }

    buf
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Generate synthetic data
    let synthetic_size = 64 * 1024 * 1024; // 64 MiB
    let synthetic_data = generate_csv(synthetic_size);

    println!("=== Synthetic Data (64 MiB) ===");
    bench_file(&synthetic_data, "synthetic", false);
    println!();

    // If a file path is provided, read and benchmark it
    if args.len() == 2 {
        match std::fs::read(&args[1]) {
            Ok(real_data) => {
                println!("=== Real File: {} ===", args[1]);
                // Real files (worldcitiespop) have a header line. falx and
                // the csv contender parse it like any row (its cells come
                // out invalid); arrow is told to skip it. Valid counts and
                // sums agree either way.
                bench_file(&real_data, &args[1], true);
                println!();
            }
            Err(e) => {
                eprintln!("Warning: Could not read '{}': {}", args[1], e);
            }
        }
    }

    Ok(())
}

struct BenchResult {
    name: &'static str,
    valid_count: usize,
    sum_lat: f64,
    sum_lon: f64,
    best_ms: f64,
    median_ms: f64,
    throughput: f64,
}

fn bench_file(data: &[u8], label: &str, has_header: bool) {
    let file_size = data.len();
    println!("File: {}", label);
    println!("Size: {} bytes ({:.2} MiB)", file_size, file_size as f64 / (1024.0 * 1024.0));
    println!();

    const WARMUP: usize = 2;
    const RUNS: usize = 9;

    let mut results = Vec::new();

    // ========== falx::kernels::csv_geo::parse_columns (serial) ==========
    {
        let mut times = Vec::new();
        for _ in 0..WARMUP {
            let _cols = falx::kernels::csv_geo::parse_columns(black_box(data));
            let _ = black_box(_cols);
        }
        for _ in 0..RUNS {
            let start = Instant::now();
            let cols = falx::kernels::csv_geo::parse_columns(black_box(data));
            let valid_count = cols.latitude_valid.iter().map(|w| w.count_ones() as usize).sum::<usize>();
            let sum_lat = black_box(cols.latitude.iter().copied().sum::<f64>());
            let sum_lon = black_box(cols.longitude.iter().copied().sum::<f64>());
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            times.push((elapsed, valid_count, sum_lat, sum_lon));
        }
        times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (best_ms, valid_count, sum_lat, sum_lon) = times[0];
        let median_ms = times[RUNS / 2].0;
        let throughput = file_size as f64 / (1024.0 * 1024.0 * 1024.0) / (median_ms / 1000.0);
        results.push(BenchResult {
            name: "falx parse_columns",
            valid_count,
            sum_lat,
            sum_lon,
            best_ms,
            median_ms,
            throughput,
        });
    }

    // ========== falx::kernels::csv_geo::parse_columns_par ==========
    {
        let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        let mut times = Vec::new();
        for _ in 0..WARMUP {
            let _cols = falx::kernels::csv_geo::parse_columns_par(black_box(data), threads);
            let _ = black_box(_cols);
        }
        for _ in 0..RUNS {
            let start = Instant::now();
            let cols = falx::kernels::csv_geo::parse_columns_par(black_box(data), threads);
            let valid_count = cols.latitude_valid.iter().map(|w| w.count_ones() as usize).sum::<usize>();
            let sum_lat = black_box(cols.latitude.iter().copied().sum::<f64>());
            let sum_lon = black_box(cols.longitude.iter().copied().sum::<f64>());
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            times.push((elapsed, valid_count, sum_lat, sum_lon));
        }
        times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (best_ms, valid_count, sum_lat, sum_lon) = times[0];
        let median_ms = times[RUNS / 2].0;
        let throughput = file_size as f64 / (1024.0 * 1024.0 * 1024.0) / (median_ms / 1000.0);
        results.push(BenchResult {
            name: "falx parse_columns_par",
            valid_count,
            sum_lat,
            sum_lon,
            best_ms,
            median_ms,
            throughput,
        });
    }

    // ========== csv crate + str::parse::<f64> ==========
    {
        let mut times = Vec::new();
        for _ in 0..WARMUP {
            let reader = csv::ReaderBuilder::new()
                .has_headers(false)
                .flexible(true)
                .from_reader(black_box(&data[..]));
            let _ = black_box(reader);
        }
        for _ in 0..RUNS {
            let start = Instant::now();
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(false)
                .flexible(true)
                .from_reader(black_box(&data[..]));
            let mut valid_count = 0usize;
            let mut sum_lat = 0.0f64;
            let mut sum_lon = 0.0f64;
            for record in reader.byte_records() {
                if let Ok(rec) = record {
                    if rec.len() > 6 {
                        if let Ok(lat_str) = std::str::from_utf8(&rec[5]) {
                            if let Ok(lat) = lat_str.parse::<f64>() {
                                sum_lat += lat;
                                valid_count += 1;
                            }
                        }
                        if let Ok(lon_str) = std::str::from_utf8(&rec[6]) {
                            if let Ok(lon) = lon_str.parse::<f64>() {
                                sum_lon += lon;
                            }
                        }
                    }
                }
            }
            let valid = black_box(valid_count);
            let lat = black_box(sum_lat);
            let lon = black_box(sum_lon);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            times.push((elapsed, valid, lat, lon));
        }
        times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (best_ms, valid_count, sum_lat, sum_lon) = times[0];
        let median_ms = times[RUNS / 2].0;
        let throughput = file_size as f64 / (1024.0 * 1024.0 * 1024.0) / (median_ms / 1000.0);
        results.push(BenchResult {
            name: "csv crate",
            valid_count,
            sum_lat,
            sum_lon,
            best_ms,
            median_ms,
            throughput,
        });
    }

    // ========== arrow-csv (explicit schema, projected to the two columns) ==========
    {
        use arrow_array::Array;
        use arrow_csv::ReaderBuilder;
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("country", DataType::Utf8, true),
            Field::new("city", DataType::Utf8, true),
            Field::new("accent_city", DataType::Utf8, true),
            Field::new("region", DataType::Utf8, true),
            Field::new("population", DataType::Int64, true),
            Field::new("latitude", DataType::Float64, true),
            Field::new("longitude", DataType::Float64, true),
        ]));

        // Projection is the like-for-like setup: arrow also only
        // materializes the two requested columns.
        let run_arrow = |data: &[u8]| -> Result<(usize, f64, f64), arrow_schema::ArrowError> {
            let reader = ReaderBuilder::new(schema.clone())
                .with_header(has_header)
                .with_batch_size(65536)
                .with_projection(vec![5, 6])
                .build(std::io::Cursor::new(data))?;
            let mut valid_count = 0usize;
            let mut sum_lat = 0.0f64;
            let mut sum_lon = 0.0f64;
            for batch in reader {
                let batch = batch?;
                // The projected batch holds only columns 5 and 6, re-indexed.
                let lat = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .expect("latitude is Float64");
                let lon = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .expect("longitude is Float64");
                valid_count += lat.len() - lat.null_count();
                sum_lat += lat.iter().flatten().sum::<f64>();
                sum_lon += lon.iter().flatten().sum::<f64>();
            }
            Ok((valid_count, sum_lat, sum_lon))
        };

        // The first call doubles as the error probe: arrow is strict about
        // malformed typed cells, so a file it cannot read (or non-UTF-8
        // data in projected columns) skips the contender rather than
        // crashing the benchmark.
        match run_arrow(black_box(data)) {
            Err(e) => eprintln!("arrow-csv skipped on {label}: {e}"),
            Ok(_) => {
                for _ in 1..WARMUP {
                    let _ = black_box(run_arrow(black_box(data)));
                }
                let mut times = Vec::new();
                for _ in 0..RUNS {
                    let start = Instant::now();
                    let out = run_arrow(black_box(data)).expect("arrow read ok after probe");
                    let (valid_count, sum_lat, sum_lon) = black_box(out);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    times.push((elapsed, valid_count, sum_lat, sum_lon));
                }
                times.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let (best_ms, valid_count, sum_lat, sum_lon) = times[0];
                let median_ms = times[RUNS / 2].0;
                let throughput = file_size as f64 / (1024.0 * 1024.0 * 1024.0) / (median_ms / 1000.0);
                results.push(BenchResult {
                    name: "arrow-csv",
                    valid_count,
                    sum_lat,
                    sum_lon,
                    best_ms,
                    median_ms,
                    throughput,
                });
            }
        }
    }

    // ========== Print results ==========
    println!("Benchmark Results ({} runs, median emphasis):", RUNS);
    println!();
    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "Contender", "Best (ms)", "Median (ms)", "Throughput", "Valid Lat", "Sum Lat"
    );
    println!("{}", "=".repeat(100));
    for result in &results {
        println!(
            "{:<25} {:>10.3} {:>10.3} {:>10.2} GiB/s {:>12} {:>12.2}",
            result.name, result.best_ms, result.median_ms, result.throughput, result.valid_count, result.sum_lat
        );
    }
    println!();
    println!("Checksum (sum of longitude values for agreement):");
    for result in &results {
        println!("  {:<25}: {:.6}", result.name, result.sum_lon);
    }
}
