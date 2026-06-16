use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_DATA_DIR: &str = "/mnt/data/falx-bench";
const DEFAULT_RUNS: usize = 3;
const DEFAULT_WARMUP: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Csv,
    CsvGeo,
    Tsv,
    Logfmt,
    Ndjson,
    Vcf,
    Fastq,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Format::Csv => "csv",
            Format::CsvGeo => "csv-geo",
            Format::Tsv => "tsv",
            Format::Logfmt => "logfmt",
            Format::Ndjson => "ndjson",
            Format::Vcf => "vcf",
            Format::Fastq => "fastq",
        }
    }

    fn path(self, data_dir: &std::path::Path) -> PathBuf {
        let ext = match self {
            Format::Csv | Format::CsvGeo => "csv",
            Format::Tsv => "tsv",
            Format::Logfmt => "logfmt",
            Format::Ndjson => "ndjson",
            Format::Vcf => "vcf",
            Format::Fastq => "fastq",
        };
        data_dir.join(format!("{}-1g.{ext}", self.name()))
    }
}

const ALL_FORMATS: &[Format] = &[
    Format::Csv,
    Format::CsvGeo,
    Format::Tsv,
    Format::Logfmt,
    Format::Ndjson,
    Format::Vcf,
    Format::Fastq,
];

#[derive(Debug)]
struct Options {
    data_dir: PathBuf,
    formats: Vec<Format>,
    runs: usize,
    warmup: usize,
    threads: Option<usize>,
    falx_only: bool,
    only_row: Option<String>,
}

#[derive(Clone, Copy)]
struct Measurement {
    best_us: u128,
    median_us: u128,
    work: Work,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Work {
    records: u64,
    primary: u64,
    secondary: u64,
    checksum: u64,
}

impl Work {
    const fn new(records: u64, primary: u64, secondary: u64, checksum: u64) -> Self {
        Self {
            records,
            primary,
            secondary,
            checksum,
        }
    }
}

impl From<u64> for Work {
    fn from(value: u64) -> Self {
        Self::new(value, 0, 0, 0)
    }
}

impl std::fmt::Display for Work {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}/{}/{}",
            self.records, self.primary, self.secondary, self.checksum
        )
    }
}

struct Row {
    label: String,
    measurement: Measurement,
}

fn main() {
    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) if message == "help" => {
            print_usage();
            return;
        }
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            std::process::exit(2);
        }
    };

    println!(
        "falx sustained comparable-library benchmark: data={}, runs={}, warmup={}",
        options.data_dir.display(),
        options.runs,
        options.warmup
    );
    if let Some(threads) = options.threads {
        println!("falx parallel thread override: {threads}");
    }
    if options.falx_only {
        println!("comparables: disabled (--falx-only)");
    } else {
        println!("comparables: {}", comparable_labels().join(", "));
    }

    for format in &options.formats {
        let path = format.path(&options.data_dir);
        let data = std::fs::read(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        match format {
            Format::Csv => bench_csv(&data, &options),
            Format::CsvGeo => bench_csv_geo(&data, &options),
            Format::Tsv => bench_tsv(&data, &options),
            Format::Logfmt => bench_logfmt(&data, &options),
            Format::Ndjson => bench_ndjson(&data, &options),
            Format::Vcf => bench_vcf(&data, &options),
            Format::Fastq => bench_fastq(&data, &options),
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --release --example bench_sustained -- \
         [--data-dir /mnt/data/falx-bench] \
         [--formats all|csv,csv-geo,tsv,logfmt,ndjson,vcf,fastq] \
         [--runs 3] [--warmup 1] [--threads N] [--falx-only] [--only-row SUBSTRING]"
    );
}

fn parse_args<I>(args: I) -> Result<Options, String>
where
    I: IntoIterator<Item = String>,
{
    let mut data_dir = PathBuf::from(DEFAULT_DATA_DIR);
    let mut formats = ALL_FORMATS.to_vec();
    let mut runs = DEFAULT_RUNS;
    let mut warmup = DEFAULT_WARMUP;
    let mut threads = None;
    let mut falx_only = false;
    let mut only_row = None;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => {
                data_dir = PathBuf::from(args.next().ok_or("--data-dir requires a path")?);
            }
            "--formats" => {
                formats = parse_formats(&args.next().ok_or("--formats requires a value")?)?;
            }
            "--runs" => {
                runs = parse_count(&args.next().ok_or("--runs requires a value")?, "--runs")?;
            }
            "--warmup" => {
                warmup = parse_count(&args.next().ok_or("--warmup requires a value")?, "--warmup")?;
            }
            "--threads" => {
                threads = Some(parse_count(
                    &args.next().ok_or("--threads requires a value")?,
                    "--threads",
                )?);
            }
            "--falx-only" => {
                falx_only = true;
            }
            "--only-row" => {
                let filter = args.next().ok_or("--only-row requires a substring")?;
                if filter.is_empty() {
                    return Err("--only-row requires a non-empty substring".into());
                }
                only_row = Some(filter);
            }
            "--help" | "-h" => return Err("help".into()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }

    Ok(Options {
        data_dir,
        formats,
        runs,
        warmup,
        threads,
        falx_only,
        only_row,
    })
}

fn parse_count(value: &str, flag: &str) -> Result<usize, String> {
    let count = value
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be a positive integer"))?;
    if count == 0 {
        return Err(format!("{flag} must be at least 1"));
    }
    Ok(count)
}

fn parse_formats(value: &str) -> Result<Vec<Format>, String> {
    if value.trim().eq_ignore_ascii_case("all") {
        return Ok(ALL_FORMATS.to_vec());
    }

    let mut formats = Vec::new();
    for raw in value.split(',') {
        let format = match raw.trim() {
            "csv" => Format::Csv,
            "csv-geo" | "geo" => Format::CsvGeo,
            "tsv" => Format::Tsv,
            "logfmt" => Format::Logfmt,
            "ndjson" => Format::Ndjson,
            "vcf" => Format::Vcf,
            "fastq" => Format::Fastq,
            other => return Err(format!("unknown format '{other}'")),
        };
        if !formats.contains(&format) {
            formats.push(format);
        }
    }
    if formats.is_empty() {
        return Err("at least one format is required".into());
    }
    Ok(formats)
}

fn measure<W>(options: &Options, mut run: impl FnMut() -> W) -> Measurement
where
    W: Into<Work>,
{
    for _ in 0..options.warmup {
        let _: Work = black_box(run()).into();
    }
    let mut times = Vec::with_capacity(options.runs);
    let mut work = Work::default();
    for _ in 0..options.runs {
        let start = Instant::now();
        work = black_box(run()).into();
        times.push(start.elapsed().as_micros());
    }
    times.sort_unstable();
    Measurement {
        best_us: times[0],
        median_us: times[times.len() / 2],
        work,
    }
}

fn report(format: &str, bytes: usize, rows: &[Row]) {
    if rows.is_empty() {
        println!("\n== {format} skipped by --only-row ==");
        return;
    }
    let gib = bytes as f64 / 1073741824.0;
    println!(
        "\n== {format} ({:.2} GiB input) ==",
        bytes as f64 / 1073741824.0
    );
    println!(
        "{:<42} {:>10} {:>12} {:>10} {:>30}",
        "contender", "best ms", "median ms", "GiB/s", "work"
    );
    for row in rows {
        println!(
            "{:<42} {:>10.2} {:>12.2} {:>10.2} {:>30}",
            row.label,
            row.measurement.best_us as f64 / 1000.0,
            row.measurement.median_us as f64 / 1000.0,
            gib / (row.measurement.best_us as f64 / 1_000_000.0),
            row.measurement.work
        );
    }
}

fn row_enabled(options: &Options, label: &str) -> bool {
    options
        .only_row
        .as_deref()
        .is_none_or(|filter| label.contains(filter))
}

fn push_measured_row<W>(
    rows: &mut Vec<Row>,
    options: &Options,
    label: impl Into<String>,
    run: impl FnMut() -> W,
) where
    W: Into<Work>,
{
    let label = label.into();
    if row_enabled(options, &label) {
        rows.push(Row {
            label,
            measurement: measure(options, run),
        });
    }
}

fn assert_same_work(format: &str, kind: &str, rows: &[Row]) {
    let Some(first) = rows.first() else {
        return;
    };
    for row in rows {
        assert_eq!(
            row.measurement.work, first.measurement.work,
            "{format} {kind}: {} returned {}, expected {}",
            row.label, row.measurement.work, first.measurement.work
        );
    }
}

fn bench_indexer(data: &[u8], options: &Options, f: fn(&[u8], &mut Vec<u32>)) -> Measurement {
    let mut out = Vec::with_capacity(data.len() / 16);
    measure(options, || {
        out.clear();
        f(data, &mut out);
        out.len() as u64
    })
}

fn bench_csv(data: &[u8], options: &Options) {
    let threads = benchmark_threads(options);
    let mut index_rows = vec![
        Row {
            label: "falx index_structurals".into(),
            measurement: bench_indexer(data, options, falx::kernels::csv::index_structurals),
        },
        Row {
            label: format!("falx index_structurals_par x{threads}"),
            measurement: {
                let mut out = Vec::with_capacity(data.len() / 16);
                measure(options, || {
                    out.clear();
                    falx::kernels::csv::index_structurals_par(data, threads, &mut out);
                    out.len() as u64
                })
            },
        },
    ];
    assert_same_work("csv", "index", &index_rows);
    report("CSV structural indexing", data.len(), &index_rows);

    let falx_parse = Row {
        label: "falx field-byte stats".into(),
        measurement: measure(options, || {
            falx::kernels::csv::parse_field_bytes(data).bytes
        }),
    };
    let falx_par = Row {
        label: format!("falx field-byte stats x{threads}"),
        measurement: measure(options, || {
            falx::kernels::csv::parse_field_bytes_par(data, threads).bytes
        }),
    };
    let mut parse_rows = vec![falx_parse, falx_par];
    if !options.falx_only {
        parse_rows.push(Row {
            label: "csv crate byte_records".into(),
            measurement: measure(options, || {
                let mut reader = csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_reader(data);
                let mut total = 0u64;
                for record in reader.byte_records() {
                    for field in record.expect("valid csv").iter() {
                        total += field.len() as u64;
                    }
                }
                total
            }),
        });
    }
    assert_same_work("csv", "field bytes", &parse_rows);
    report("CSV like-for-like field bytes", data.len(), &parse_rows);

    index_rows.clear();
}

fn bench_csv_geo(data: &[u8], options: &Options) {
    let threads = benchmark_threads(options);
    let mut numeric_rows = vec![
        Row {
            label: "falx csv_geo stats lat/lon".into(),
            measurement: measure(options, || falx_csv_geo_numeric_work(data)),
        },
        Row {
            label: format!("falx csv_geo stats lat/lon x{threads}"),
            measurement: measure(options, || {
                falx_csv_geo_numeric_parallel_work(data, threads)
            }),
        },
    ];
    if !options.falx_only {
        numeric_rows.push(Row {
            label: "csv crate projected lat/lon".into(),
            measurement: measure(options, || csv_geo_csv_crate_numeric(data)),
        });
        numeric_rows.push(Row {
            label: "arrow-csv projected lat/lon".into(),
            measurement: measure(options, || csv_geo_arrow_numeric(data)),
        });
    }
    assert_same_work("csv-geo", "numeric valid rows", &numeric_rows);
    report(
        "CSV geo typed numeric projection",
        data.len(),
        &numeric_rows,
    );

    let mut text_rows = vec![
        Row {
            label: "falx csv_geo_text stats city+lat/lon".into(),
            measurement: measure(options, || falx_csv_geo_text_work(data)),
        },
        Row {
            label: format!("falx csv_geo_text stats city+lat/lon x{threads}"),
            measurement: measure(options, || falx_csv_geo_text_parallel_work(data, threads)),
        },
    ];
    if !options.falx_only {
        text_rows.push(Row {
            label: "csv crate projected city+lat/lon".into(),
            measurement: measure(options, || csv_geo_csv_crate_text(data)),
        });
        text_rows.push(Row {
            label: "arrow-csv projected city+lat/lon".into(),
            measurement: measure(options, || csv_geo_arrow_text(data)),
        });
    }
    assert_same_work("csv-geo", "text valid rows", &text_rows);
    report("CSV geo text+numeric projection", data.len(), &text_rows);
}

fn falx_csv_geo_numeric_work(data: &[u8]) -> Work {
    let stats = falx::kernels::csv_geo::parse_csv_geo_stats(data);
    csv_geo_numeric_work(
        stats.latitude_values,
        stats.longitude_values,
        stats.latitude_checksum,
        stats.longitude_checksum,
    )
}

fn falx_csv_geo_numeric_parallel_work(data: &[u8], threads: usize) -> Work {
    let stats = falx::kernels::csv_geo::parse_csv_geo_stats_par(data, threads);
    csv_geo_numeric_work(
        stats.latitude_values,
        stats.longitude_values,
        stats.latitude_checksum,
        stats.longitude_checksum,
    )
}

fn csv_geo_numeric_work(
    latitude_values: u64,
    longitude_values: u64,
    latitude_checksum: u64,
    longitude_checksum: u64,
) -> Work {
    Work::new(
        latitude_values,
        longitude_values,
        0,
        latitude_checksum.wrapping_add(longitude_checksum),
    )
}

fn falx_csv_geo_text_work(data: &[u8]) -> Work {
    let stats = falx::kernels::csv_geo_text::parse_csv_geo_text_stats(data);
    csv_geo_text_stats_work(
        stats.city_values,
        stats.city_bytes,
        stats.city_checksum,
        stats.latitude_values,
        stats.longitude_values,
        stats.latitude_checksum,
        stats.longitude_checksum,
    )
}

fn falx_csv_geo_text_parallel_work(data: &[u8], threads: usize) -> Work {
    let stats = falx::kernels::csv_geo_text::parse_csv_geo_text_stats_par(data, threads);
    csv_geo_text_stats_work(
        stats.city_values,
        stats.city_bytes,
        stats.city_checksum,
        stats.latitude_values,
        stats.longitude_values,
        stats.latitude_checksum,
        stats.longitude_checksum,
    )
}

fn csv_geo_text_stats_work(
    city_values_including_header: u64,
    city_bytes_including_header: u64,
    city_checksum_including_header: u64,
    latitude_values: u64,
    longitude_values: u64,
    latitude_checksum: u64,
    longitude_checksum: u64,
) -> Work {
    let city_checksum =
        city_checksum_including_header.wrapping_sub(csv_geo_checksum_bytes(b"City"));
    Work::new(
        city_values_including_header.saturating_sub(1),
        city_bytes_including_header.saturating_sub(4),
        latitude_values + longitude_values,
        city_checksum
            .wrapping_add(latitude_checksum)
            .wrapping_add(longitude_checksum),
    )
}

fn csv_geo_csv_crate_numeric(data: &[u8]) -> Work {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(data);
    let mut latitude_values = 0u64;
    let mut longitude_values = 0u64;
    let mut latitude_checksum = 0u64;
    let mut longitude_checksum = 0u64;
    for record in reader.byte_records().flatten() {
        if let Some(cell) = record.get(5)
            && let Ok(lat) = std::str::from_utf8(cell).unwrap_or("").parse::<f64>()
        {
            latitude_values += 1;
            latitude_checksum = latitude_checksum.wrapping_add(lat.to_bits());
        }
        if let Some(cell) = record.get(6)
            && let Ok(lon) = std::str::from_utf8(cell).unwrap_or("").parse::<f64>()
        {
            longitude_values += 1;
            longitude_checksum = longitude_checksum.wrapping_add(lon.to_bits());
        }
    }
    csv_geo_numeric_work(
        latitude_values,
        longitude_values,
        latitude_checksum,
        longitude_checksum,
    )
}

fn csv_geo_csv_crate_text(data: &[u8]) -> Work {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(data);
    let mut city_values = 0u64;
    let mut city_bytes = 0u64;
    let mut city_checksum = 0u64;
    let mut latitude_values = 0u64;
    let mut longitude_values = 0u64;
    let mut latitude_checksum = 0u64;
    let mut longitude_checksum = 0u64;
    for record in reader.byte_records().flatten() {
        if let Some(city) = record.get(1) {
            city_values += 1;
            city_bytes += city.len() as u64;
            city_checksum = city_checksum.wrapping_add(csv_geo_checksum_bytes(city));
        }
        if let Some(cell) = record.get(5)
            && let Ok(lat) = std::str::from_utf8(cell).unwrap_or("").parse::<f64>()
        {
            latitude_values += 1;
            latitude_checksum = latitude_checksum.wrapping_add(lat.to_bits());
        }
        if let Some(cell) = record.get(6)
            && let Ok(lon) = std::str::from_utf8(cell).unwrap_or("").parse::<f64>()
        {
            longitude_values += 1;
            longitude_checksum = longitude_checksum.wrapping_add(lon.to_bits());
        }
    }
    Work::new(
        city_values,
        city_bytes,
        latitude_values + longitude_values,
        city_checksum
            .wrapping_add(latitude_checksum)
            .wrapping_add(longitude_checksum),
    )
}

fn csv_geo_arrow_numeric(data: &[u8]) -> Work {
    use arrow_array::Array;
    let schema = csv_geo_schema();
    let reader = arrow_csv::ReaderBuilder::new(schema)
        .with_header(true)
        .with_batch_size(65536)
        .with_projection(vec![5, 6])
        .build(std::io::Cursor::new(data))
        .expect("arrow csv numeric reader");
    let mut latitude_values = 0u64;
    let mut longitude_values = 0u64;
    let mut latitude_checksum = 0u64;
    let mut longitude_checksum = 0u64;
    for batch in reader {
        let batch = batch.expect("arrow csv numeric batch");
        let lat = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .expect("lat f64");
        for value in lat.iter().flatten() {
            latitude_values += 1;
            latitude_checksum = latitude_checksum.wrapping_add(value.to_bits());
        }
        let lon = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .expect("lon f64");
        for value in lon.iter().flatten() {
            longitude_values += 1;
            longitude_checksum = longitude_checksum.wrapping_add(value.to_bits());
        }
    }
    csv_geo_numeric_work(
        latitude_values,
        longitude_values,
        latitude_checksum,
        longitude_checksum,
    )
}

fn csv_geo_arrow_text(data: &[u8]) -> Work {
    use arrow_array::Array;
    let schema = csv_geo_schema();
    let reader = arrow_csv::ReaderBuilder::new(schema)
        .with_header(true)
        .with_batch_size(65536)
        .with_projection(vec![1, 5, 6])
        .build(std::io::Cursor::new(data))
        .expect("arrow csv text reader");
    let mut city_values = 0u64;
    let mut city_bytes = 0u64;
    let mut city_checksum = 0u64;
    let mut latitude_values = 0u64;
    let mut longitude_values = 0u64;
    let mut latitude_checksum = 0u64;
    let mut longitude_checksum = 0u64;
    for batch in reader {
        let batch = batch.expect("arrow csv text batch");
        let city = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .expect("city utf8");
        for maybe_city in city.iter().flatten() {
            city_values += 1;
            city_bytes += maybe_city.len() as u64;
            city_checksum =
                city_checksum.wrapping_add(csv_geo_checksum_bytes(maybe_city.as_bytes()));
        }
        let lat = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .expect("lat f64");
        for value in lat.iter().flatten() {
            latitude_values += 1;
            latitude_checksum = latitude_checksum.wrapping_add(value.to_bits());
        }
        let lon = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .expect("lon f64");
        for value in lon.iter().flatten() {
            longitude_values += 1;
            longitude_checksum = longitude_checksum.wrapping_add(value.to_bits());
        }
    }
    Work::new(
        city_values,
        city_bytes,
        latitude_values + longitude_values,
        city_checksum
            .wrapping_add(latitude_checksum)
            .wrapping_add(longitude_checksum),
    )
}

fn csv_geo_checksum_bytes(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0u64, |acc, byte| acc.wrapping_add(*byte as u64))
}

fn csv_geo_schema() -> std::sync::Arc<arrow_schema::Schema> {
    use arrow_schema::{DataType, Field, Schema};
    std::sync::Arc::new(Schema::new(vec![
        Field::new("country", DataType::Utf8, true),
        Field::new("city", DataType::Utf8, true),
        Field::new("accent_city", DataType::Utf8, true),
        Field::new("region", DataType::Utf8, true),
        Field::new("population", DataType::Int64, true),
        Field::new("latitude", DataType::Float64, true),
        Field::new("longitude", DataType::Float64, true),
    ]))
}

fn bench_tsv(data: &[u8], options: &Options) {
    let threads = benchmark_threads(options);
    let mut rows = vec![
        Row {
            label: "falx field-byte stats".into(),
            measurement: measure(options, || {
                falx::kernels::tsv::parse_field_bytes(data).bytes
            }),
        },
        Row {
            label: format!("falx field-byte stats x{threads}"),
            measurement: measure(options, || {
                falx::kernels::tsv::parse_field_bytes_par(data, threads).bytes
            }),
        },
    ];
    if !options.falx_only {
        rows.push(Row {
            label: "csv crate delimiter=tab".into(),
            measurement: measure(options, || {
                let mut reader = csv::ReaderBuilder::new()
                    .delimiter(b'\t')
                    .has_headers(false)
                    .from_reader(data);
                let mut total = 0u64;
                for record in reader.byte_records() {
                    for field in record.expect("valid tsv").iter() {
                        total += field.len() as u64;
                    }
                }
                total
            }),
        });
    }
    assert_same_work("tsv", "field bytes", &rows);
    report("TSV field bytes", data.len(), &rows);
}

fn bench_logfmt(data: &[u8], options: &Options) {
    let threads = benchmark_threads(options);
    let mut rows = Vec::new();
    push_measured_row(&mut rows, options, "falx logfmt pairs", || {
        falx_logfmt_work(data)
    });
    push_measured_row(
        &mut rows,
        options,
        format!("falx logfmt pairs x{threads}"),
        || falx_logfmt_parallel_work(data, threads),
    );
    if !options.falx_only {
        push_measured_row(&mut rows, options, "logfmt-zerocopy pairs", || {
            logfmt_zerocopy_work(data)
        });
    }
    assert_same_work("logfmt", "pairs", &rows);
    report("logfmt key/value pairs", data.len(), &rows);
}

fn falx_logfmt_work(data: &[u8]) -> Work {
    let stats = falx::kernels::logfmt::parse_logfmt_pairs(data);
    Work::new(
        stats.pairs,
        stats.key_bytes,
        stats.value_bytes,
        stats.checksum,
    )
}

fn falx_logfmt_parallel_work(data: &[u8], threads: usize) -> Work {
    let stats = falx::kernels::logfmt::parse_logfmt_pairs_par(data, threads);
    Work::new(
        stats.pairs,
        stats.key_bytes,
        stats.value_bytes,
        stats.checksum,
    )
}

fn logfmt_zerocopy_work(data: &[u8]) -> Work {
    use logfmt_zerocopy::Logfmt;

    let mut work = Work::default();
    for line in data.split(|&byte| byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        let line = std::str::from_utf8(line).expect("valid UTF-8 logfmt");
        for (key, value) in line.logfmt() {
            let value = unescape_logfmt_value(value.as_bytes());
            add_logfmt_pair(&mut work, key.as_bytes(), &value);
        }
    }
    work
}

fn add_logfmt_pair(work: &mut Work, key: &[u8], value: &[u8]) {
    work.records += 1;
    work.primary += key.len() as u64;
    work.secondary += value.len() as u64;
    work.checksum = logfmt_checksum_bytes(work.checksum, key);
    work.checksum = logfmt_checksum_bytes(work.checksum, value);
}

#[inline]
fn logfmt_checksum_bytes(mut checksum: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        let pairs = (word & 0x00ff_00ff_00ff_00ff) + ((word >> 8) & 0x00ff_00ff_00ff_00ff);
        let quads = (pairs & 0x0000_ffff_0000_ffff) + ((pairs >> 16) & 0x0000_ffff_0000_ffff);
        checksum = checksum.wrapping_add((quads & 0x0000_0000_ffff_ffff) + (quads >> 32));
    }
    for &byte in chunks.remainder() {
        checksum = checksum.wrapping_add(byte as u64);
    }
    checksum
}

fn unescape_logfmt_value(value: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if !value.contains(&b'\\') {
        return std::borrow::Cow::Borrowed(value);
    }

    let mut out = Vec::with_capacity(value.len());
    let mut escaped = false;
    for &byte in value {
        if escaped {
            out.push(byte);
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else {
            out.push(byte);
        }
    }
    if escaped {
        out.push(b'\\');
    }
    std::borrow::Cow::Owned(out)
}

fn bench_ndjson(data: &[u8], options: &Options) {
    let threads = benchmark_threads(options);
    let framing_rows = vec![
        Row {
            label: "falx NDJSON line stats".into(),
            measurement: measure(options, || {
                falx::kernels::ndjson::parse_ndjson_lines(data).records
            }),
        },
        Row {
            label: format!("falx NDJSON line stats x{threads}"),
            measurement: measure(options, || {
                falx::kernels::ndjson::parse_ndjson_lines_par(data, threads).records
            }),
        },
    ];
    assert_same_work("ndjson", "line framing", &framing_rows);
    report("NDJSON line framing", data.len(), &framing_rows);

    if !options.falx_only {
        let mut parser_rows = vec![Row {
            label: "serde_json full DOM parse".into(),
            measurement: measure(options, || {
                let mut docs = 0u64;
                for line in data
                    .split(|&byte| byte == b'\n')
                    .filter(|line| !line.is_empty())
                {
                    serde_json::from_slice::<serde_json::Value>(line).expect("valid json");
                    docs += 1;
                }
                docs
            }),
        }];
        parser_rows.push(Row {
            label: "simd-json full tape parse".into(),
            measurement: bench_simd_json_tape(data, options),
        });
        assert_same_work("ndjson", "document count", &parser_rows);
        report(
            "NDJSON full JSON parser baselines",
            data.len(),
            &parser_rows,
        );
    }
}

fn bench_simd_json_tape(data: &[u8], options: &Options) -> Measurement {
    for _ in 0..options.warmup {
        let mut scratch = data.to_vec();
        let _: Work = black_box(simd_json_docs(&mut scratch)).into();
    }
    let mut times = Vec::with_capacity(options.runs);
    let mut work = Work::default();
    for _ in 0..options.runs {
        let mut scratch = data.to_vec();
        let start = Instant::now();
        work = black_box(simd_json_docs(&mut scratch)).into();
        times.push(start.elapsed().as_micros());
    }
    times.sort_unstable();
    Measurement {
        best_us: times[0],
        median_us: times[times.len() / 2],
        work,
    }
}

fn simd_json_docs(data: &mut [u8]) -> u64 {
    let mut docs = 0u64;
    let mut offset = 0usize;
    while offset < data.len() {
        let end = data[offset..]
            .iter()
            .position(|&byte| byte == b'\n')
            .map_or(data.len(), |position| offset + position);
        if end > offset {
            simd_json::to_tape(&mut data[offset..end]).expect("valid json");
            docs += 1;
        }
        offset = end + 1;
    }
    docs
}

fn bench_vcf(data: &[u8], options: &Options) {
    let threads = benchmark_threads(options);
    let mut rows = vec![
        Row {
            label: "falx vcf_typed projection".into(),
            measurement: measure(options, || falx_vcf_typed_work(data)),
        },
        Row {
            label: format!("falx vcf_typed projection x{threads}"),
            measurement: measure(options, || falx_vcf_typed_parallel_work(data, threads)),
        },
    ];
    if !options.falx_only {
        rows.push(Row {
            label: "noodles-vcf typed records".into(),
            measurement: measure(options, || noodles_vcf_work(data)),
        });
    }
    assert_same_work("vcf", "typed projection", &rows);
    report("VCF typed projection", data.len(), &rows);
}

fn bench_fastq(data: &[u8], options: &Options) {
    let threads = benchmark_threads(options);
    let mut rows = Vec::new();
    push_measured_row(&mut rows, options, "falx generated FASTQ", || {
        generated_fastq_work(data).expect("valid FASTQ for falx")
    });
    push_measured_row(
        &mut rows,
        options,
        format!("falx generated FASTQ par x{threads}"),
        || generated_fastq_parallel_work(data, threads).expect("valid FASTQ for falx"),
    );
    if !options.falx_only {
        push_measured_row(&mut rows, options, "seq_io fastq", || {
            seq_io_fastq_work(data)
        });
        push_measured_row(&mut rows, options, "needletail fastq", || {
            needletail_fastq_work(data)
        });
    }
    assert_same_work("fastq", "records", &rows);
    report("FASTQ records", data.len(), &rows);
}

#[cfg(test)]
fn fastq_sequence_bases_from_newlines(data: &[u8], newlines: &[u32]) -> u64 {
    let mut bases = 0u64;
    let mut start = 0usize;
    for record in newlines.chunks_exact(4) {
        let sequence_end = record[1] as usize;
        let header_end = record[0] as usize;
        bases += sequence_end.saturating_sub(header_end + 1) as u64;
        start = record[3] as usize + 1;
    }
    black_box(start);
    black_box(data.len());
    bases
}

#[cfg(test)]
fn scalar_fastq_sequence_bases(data: &[u8]) -> u64 {
    let mut bases = 0u64;
    let mut line_start = 0usize;
    let mut line_in_record = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        if byte == b'\n' {
            if line_in_record == 1 {
                bases += i.saturating_sub(line_start) as u64;
            }
            line_in_record = (line_in_record + 1) % 4;
            line_start = i + 1;
        }
    }
    bases
}

fn generated_fastq_work(data: &[u8]) -> Result<Work, falx::kernels::fastq::FastqError> {
    falx::kernels::fastq::parse_fastq(data).map(|stats| {
        Work::new(
            stats.records,
            stats.sequence_bytes,
            stats.quality_bytes,
            stats.checksum,
        )
    })
}

fn generated_fastq_parallel_work(
    data: &[u8],
    threads: usize,
) -> Result<Work, falx::kernels::fastq::FastqError> {
    falx::kernels::fastq::parse_fastq_par(data, threads).map(|stats| {
        Work::new(
            stats.records,
            stats.sequence_bytes,
            stats.quality_bytes,
            stats.checksum,
        )
    })
}

fn seq_io_fastq_work(data: &[u8]) -> Work {
    use seq_io::fastq::Record;

    let mut reader = seq_io::fastq::Reader::new(data);
    let mut work = Work::default();
    while let Some(record) = reader.next() {
        let record = record.expect("valid FASTQ record for seq_io");
        add_fastq_record(&mut work, record.seq(), record.qual());
    }
    work
}

fn needletail_fastq_work(data: &[u8]) -> Work {
    use needletail::parser::FastxReader;

    let mut reader = needletail::parser::FastqReader::new(data);
    let mut work = Work::default();
    while let Some(record) = reader.next() {
        let record = record.expect("valid FASTQ record for needletail");
        let seq = record.raw_seq();
        let qual = record.qual().expect("FASTQ quality line");
        add_fastq_record(&mut work, seq, qual);
    }
    work
}

fn falx_vcf_typed_work(data: &[u8]) -> Work {
    let stats = falx::kernels::vcf_typed::parse_vcf_stats(data);
    Work::new(
        stats.records,
        stats.primary_bytes,
        stats.pos_values,
        stats.checksum,
    )
}

fn falx_vcf_typed_parallel_work(data: &[u8], threads: usize) -> Work {
    let stats = falx::kernels::vcf_typed::parse_vcf_stats_par(data, threads);
    Work::new(
        stats.records,
        stats.primary_bytes,
        stats.pos_values,
        stats.checksum,
    )
}

#[cfg(test)]
fn falx_vcf_work_from_columns(cols: &falx::kernels::vcf_typed::Columns<'_>) -> Work {
    let mut work = Work {
        records: cols.rows as u64,
        ..Work::default()
    };

    for row in 0..cols.rows {
        if valid_at(&cols.pos_valid, row) {
            work.secondary += 1;
            work.checksum = vcf_checksum_u64(work.checksum, cols.pos[row] as u64);
        }
        if valid_at(&cols.reference_valid, row) {
            let (start, end) = cols.reference[row];
            let bytes = &cols.data[start as usize..end as usize];
            work.primary += bytes.len() as u64;
            work.checksum = vcf_checksum_bytes(work.checksum, bytes);
        }
        if valid_at(&cols.alternate_valid, row) {
            let (start, end) = cols.alternate[row];
            let bytes = &cols.data[start as usize..end as usize];
            work.primary += bytes.len() as u64;
            work.checksum = vcf_checksum_bytes(work.checksum, bytes);
        }
        if valid_at(&cols.quality_valid, row) {
            work.checksum =
                vcf_checksum_u64(work.checksum, (cols.quality[row] as f32).to_bits() as u64);
        }
    }

    work
}

fn noodles_vcf_work(data: &[u8]) -> Work {
    let mut reader = noodles_vcf::io::Reader::new(std::io::Cursor::new(data));
    reader
        .read_header()
        .expect("valid VCF header for noodles-vcf");

    let mut work = Work::default();
    for record in reader.records() {
        let record = record.expect("valid VCF record for noodles-vcf");
        work.records += 1;

        let pos = record
            .variant_start()
            .expect("VCF record has POS")
            .expect("valid VCF POS");
        work.secondary += 1;
        work.checksum = vcf_checksum_u64(work.checksum, pos.get() as u64);

        let reference = record.reference_bases().as_bytes();
        work.primary += reference.len() as u64;
        work.checksum = vcf_checksum_bytes(work.checksum, reference);

        let alternate = record.alternate_bases();
        let alternate = alternate.as_ref().as_bytes();
        work.primary += alternate.len() as u64;
        work.checksum = vcf_checksum_bytes(work.checksum, alternate);

        if let Some(quality) = record.quality_score() {
            work.checksum = vcf_checksum_u64(
                work.checksum,
                quality.expect("valid VCF QUAL").to_bits() as u64,
            );
        }
    }
    work
}

#[cfg(test)]
fn valid_at(bits: &[u64], row: usize) -> bool {
    bits.get(row / 64)
        .is_some_and(|word| (word & (1u64 << (row % 64))) != 0)
}

#[inline]
fn vcf_checksum_bytes(mut checksum: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        let pairs = (word & 0x00ff_00ff_00ff_00ff) + ((word >> 8) & 0x00ff_00ff_00ff_00ff);
        let quads = (pairs & 0x0000_ffff_0000_ffff) + ((pairs >> 16) & 0x0000_ffff_0000_ffff);
        checksum = checksum.wrapping_add((quads & 0x0000_0000_ffff_ffff) + (quads >> 32));
    }
    for &byte in chunks.remainder() {
        checksum = checksum.wrapping_add(byte as u64);
    }
    checksum
}

fn vcf_checksum_u64(checksum: u64, value: u64) -> u64 {
    vcf_checksum_bytes(checksum, &value.to_le_bytes())
}

fn add_fastq_record(work: &mut Work, seq: &[u8], qual: &[u8]) {
    work.records += 1;
    work.primary += seq.len() as u64;
    work.secondary += qual.len() as u64;
    work.checksum = fastq_checksum_bytes(work.checksum, seq);
    work.checksum = fastq_checksum_bytes(work.checksum, qual);
}

#[inline]
fn fastq_checksum_bytes(mut checksum: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        let pairs = (word & 0x00ff_00ff_00ff_00ff) + ((word >> 8) & 0x00ff_00ff_00ff_00ff);
        let quads = (pairs & 0x0000_ffff_0000_ffff) + ((pairs >> 16) & 0x0000_ffff_0000_ffff);
        checksum = checksum.wrapping_add((quads & 0x0000_0000_ffff_ffff) + (quads >> 32));
    }
    for &byte in chunks.remainder() {
        checksum = checksum.wrapping_add(byte as u64);
    }
    checksum
}

fn comparable_labels() -> &'static [&'static str] {
    &[
        "csv crate byte_records",
        "arrow-csv projected lat/lon",
        "seq_io fastq",
        "needletail fastq",
        "noodles-vcf typed records",
        "logfmt-zerocopy pairs",
        "serde_json full DOM parse",
        "simd-json full tape parse",
    ]
}

fn available_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |threads| threads.get())
}

fn benchmark_threads(options: &Options) -> usize {
    options.threads.unwrap_or_else(available_threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_cover_all_staged_formats() {
        let options = parse_args(std::iter::empty()).unwrap();
        assert_eq!(options.data_dir, std::path::PathBuf::from(DEFAULT_DATA_DIR));
        assert!(options.formats.contains(&Format::Csv));
        assert!(options.formats.contains(&Format::Ndjson));
        assert!(options.formats.contains(&Format::Fastq));
    }

    #[test]
    fn format_list_preserves_requested_order() {
        let options = parse_args(["--formats".to_string(), "ndjson,csv".to_string()]).unwrap();
        assert_eq!(options.formats, vec![Format::Ndjson, Format::Csv]);
    }

    #[test]
    fn thread_override_is_parsed() {
        let options = parse_args(["--threads".to_string(), "24".to_string()]).unwrap();
        assert_eq!(options.threads, Some(24));
    }

    #[test]
    fn falx_only_mode_is_parsed() {
        let options = parse_args(["--falx-only".to_string()]).unwrap();
        assert!(options.falx_only);
    }

    #[test]
    fn only_row_filter_is_parsed() {
        let options = parse_args(["--only-row".to_string(), "x48".to_string()]).unwrap();
        assert_eq!(options.only_row.as_deref(), Some("x48"));
    }

    #[test]
    fn fastq_newline_framing_matches_scalar() {
        let data = b"@r0\nACGT\n+\n!!!!\n@r1\nAC\n+\n!!\n";
        let newlines: Vec<u32> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &byte)| (byte == b'\n').then_some(i as u32))
            .collect();
        assert_eq!(
            fastq_sequence_bases_from_newlines(data, &newlines),
            scalar_fastq_sequence_bases(data)
        );
    }

    #[test]
    fn work_equality_checks_all_counters() {
        let a = Work::new(2, 10, 20, 30);
        let b = Work::new(2, 10, 20, 31);
        assert_ne!(a, b);
    }

    #[test]
    fn generated_fastq_reader_validates_sequence_quality_lengths() {
        let bad = b"@r0\nACGT\n+\n!!!\n";
        assert!(generated_fastq_work(bad).is_err());
    }

    #[test]
    fn generated_fastq_matches_proper_libraries() {
        let data = b"@r0 desc\nACGT\n+\n!!!!\n@r1\nAC\n+\n!!\n";
        let falx = generated_fastq_work(data).unwrap();
        assert_eq!(falx, seq_io_fastq_work(data));
        assert_eq!(falx, needletail_fastq_work(data));
    }

    #[test]
    fn generated_fastq_parallel_matches_serial_and_libraries() {
        let data = b"@r0 desc\nACGT\n+\n!!!!\n@r1\nAC\n+\n!!\n@r2\nNNNNNN\n+meta\n!!!!!!\n";
        let serial = generated_fastq_work(data).unwrap();
        for threads in [1, 2, 3, 8] {
            let parallel = generated_fastq_parallel_work(data, threads).unwrap();
            assert_eq!(serial, parallel, "threads {threads}");
        }
        assert_eq!(serial, seq_io_fastq_work(data));
        assert_eq!(serial, needletail_fastq_work(data));
    }

    #[test]
    fn generated_fastq_parallel_ignores_false_at_boundaries() {
        let mut data = Vec::new();
        let mut sequence = vec![b'A'; 128];
        sequence[32] = b'@';
        let mut quality = vec![b'!'; sequence.len()];
        quality[32] = b'@';
        for record in 0..6000 {
            data.extend_from_slice(format!("@r{record}\n").as_bytes());
            data.extend_from_slice(&sequence);
            data.push(b'\n');
            data.extend_from_slice(b"+\n");
            data.extend_from_slice(&quality);
            data.push(b'\n');
        }

        let serial = generated_fastq_work(&data).unwrap();
        for threads in [2, 7, 31, 48] {
            let parallel = generated_fastq_parallel_work(&data, threads).unwrap();
            assert_eq!(serial, parallel, "threads {threads}");
        }
        assert_eq!(serial, seq_io_fastq_work(&data));
        assert_eq!(serial, needletail_fastq_work(&data));
    }

    #[test]
    fn generated_fastq_rejects_bad_separator() {
        let data = b"@r0\nACGT\n-\n!!!!\n";
        assert!(generated_fastq_work(data).is_err());
        assert!(generated_fastq_parallel_work(data, 4).is_err());
    }

    #[test]
    fn ndjson_line_stats_match_parser_records() {
        let data = b"{\"msg\":\"hello\\nworld\"}\n{\"escaped\":\"quote \\\" ok\"}\n{\"tail\":true}";
        let expected = falx::kernels::ndjson::parse(data).records().count() as u64;
        let stats = falx::kernels::ndjson::parse_ndjson_lines(data);
        assert_eq!(stats.records, expected);

        for threads in [1, 2, 3, 8] {
            assert_eq!(
                stats,
                falx::kernels::ndjson::parse_ndjson_lines_par(data, threads),
                "threads {threads}"
            );
        }
    }

    #[test]
    fn csv_fused_field_bytes_match_cleaned_fields() {
        let data = b"a,\"b,c\",\"d\"\"e\"\nplain,,tail\n";
        let expected = {
            let parsed = falx::kernels::csv::parse(data);
            parsed
                .records()
                .flat_map(|record| record.fields())
                .map(|field| field.len() as u64)
                .sum::<u64>()
        };
        let stats = falx::kernels::csv::parse_field_bytes(data);
        assert_eq!(stats.bytes, expected);

        for threads in [1, 2, 3, 8] {
            assert_eq!(
                stats,
                falx::kernels::csv::parse_field_bytes_par(data, threads),
                "threads {threads}"
            );
        }
    }

    #[test]
    fn tsv_fused_field_bytes_match_raw_fields() {
        let data = b"a\tbb\tccc\n\tleft-empty\tz\nunterminated\trow";
        let expected = {
            let parsed = falx::kernels::tsv::parse(data);
            parsed
                .records()
                .flat_map(|record| record.fields_raw())
                .map(|field| field.len() as u64)
                .sum::<u64>()
        };
        let stats = falx::kernels::tsv::parse_field_bytes(data);
        assert_eq!(stats.bytes, expected);

        for threads in [1, 2, 3, 8] {
            assert_eq!(
                stats,
                falx::kernels::tsv::parse_field_bytes_par(data, threads),
                "threads {threads}"
            );
        }
    }

    #[test]
    fn vcf_falx_typed_matches_noodles() {
        let data = b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t7\trs1\tA\tC\t42.5\tPASS\tDP=10\n";
        assert_eq!(falx_vcf_typed_work(data), noodles_vcf_work(data));
    }

    #[test]
    fn vcf_falx_typed_parallel_matches_serial_columns_and_noodles() {
        let data = b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t7\trs1\tA\tC\t42.5\tPASS\tDP=10\nchr2\t11\trs2\tG\tT\t9.25\tPASS\tDP=3\n";
        let serial = falx_vcf_typed_work(data);
        let cols = falx::kernels::vcf_typed::parse_columns(data);
        assert_eq!(serial, falx_vcf_work_from_columns(&cols));
        for threads in [1, 2, 3, 8] {
            assert_eq!(serial, falx_vcf_typed_parallel_work(data, threads));
        }
        assert_eq!(serial, noodles_vcf_work(data));
    }

    #[test]
    fn csv_geo_fused_numeric_stats_match_materialized_columns() {
        let data = b"Country,City,AccentCity,Region,Population,Latitude,Longitude\nUS,Boston,Boston,MA,675647,42.3601,-71.0589\nCA,Toronto,Toronto,ON,2731571,43.6532,-79.3832\nFR,BadLat,BadLat,IDF,10,not-a-lat,2.3522\n";
        let cols = falx::kernels::csv_geo::parse_columns(data);
        let stats = falx::kernels::csv_geo::parse_csv_geo_stats(data);

        assert_eq!(stats.records, cols.rows as u64);
        assert_eq!(stats.latitude_values, count_bitmap(&cols.latitude_valid));
        assert_eq!(stats.longitude_values, count_bitmap(&cols.longitude_valid));
        assert_eq!(stats.latitude_checksum, f64_column_checksum(&cols.latitude));
        assert_eq!(
            stats.longitude_checksum,
            f64_column_checksum(&cols.longitude)
        );

        for threads in [1, 2, 3, 8] {
            assert_eq!(
                stats,
                falx::kernels::csv_geo::parse_csv_geo_stats_par(data, threads),
                "threads {threads}"
            );
        }
    }

    #[test]
    fn csv_geo_text_fused_stats_match_materialized_columns() {
        let data = b"Country,City,AccentCity,Region,Population,Latitude,Longitude\nUS,\"New \"\"York\"\"\",New York,NY,8804190,40.7128,-74.0060\nUS,,Empty,NA,,0.0,0.0\nCA,Toronto,Toronto,ON,2731571,43.6532,-79.3832\n";
        let cols = falx::kernels::csv_geo_text::parse_columns(data);
        let stats = falx::kernels::csv_geo_text::parse_csv_geo_text_stats(data);

        assert_eq!(stats.records, cols.rows as u64);
        assert_eq!(stats.city_values, count_bitmap(&cols.city_valid));
        assert_eq!(stats.city_bytes, cols.city_data.len() as u64);
        assert_eq!(stats.city_checksum, byte_checksum(&cols.city_data));
        assert_eq!(stats.latitude_values, count_bitmap(&cols.latitude_valid));
        assert_eq!(stats.longitude_values, count_bitmap(&cols.longitude_valid));
        assert_eq!(stats.latitude_checksum, f64_column_checksum(&cols.latitude));
        assert_eq!(
            stats.longitude_checksum,
            f64_column_checksum(&cols.longitude)
        );

        for threads in [1, 2, 3, 8] {
            assert_eq!(
                stats,
                falx::kernels::csv_geo_text::parse_csv_geo_text_stats_par(data, threads),
                "threads {threads}"
            );
        }
    }

    #[test]
    fn logfmt_falx_matches_logfmt_zerocopy() {
        let data = b"level=info msg=\"hello world\" count=42\npath=/tmp ok=true\nmsg=\"with \\\"quoted\\\" part\" raw=abc\n";
        assert_eq!(falx_logfmt_work(data), logfmt_zerocopy_work(data));
    }

    #[test]
    fn logfmt_parallel_matches_serial_and_zerocopy() {
        let data = b"level=info msg=\"hello world\" count=42\npath=/tmp ok=true\nmsg=\"with \\\"quoted\\\" part\" raw=abc\n";
        let serial = falx_logfmt_work(data);
        for threads in [1, 2, 3, 8] {
            assert_eq!(serial, falx_logfmt_parallel_work(data, threads));
        }
        assert_eq!(serial, logfmt_zerocopy_work(data));
    }

    #[test]
    fn proper_library_labels_are_used() {
        let labels = comparable_labels();
        assert!(labels.contains(&"seq_io fastq"));
        assert!(labels.contains(&"needletail fastq"));
        assert!(labels.contains(&"noodles-vcf typed records"));
        assert!(labels.contains(&"logfmt-zerocopy pairs"));
    }

    fn count_bitmap(bitmap: &[u64]) -> u64 {
        bitmap.iter().map(|word| word.count_ones() as u64).sum()
    }

    fn f64_column_checksum(values: &[f64]) -> u64 {
        values
            .iter()
            .fold(0u64, |acc, value| acc.wrapping_add(value.to_bits()))
    }

    fn byte_checksum(bytes: &[u8]) -> u64 {
        bytes
            .iter()
            .fold(0u64, |acc, byte| acc.wrapping_add(*byte as u64))
    }
}
