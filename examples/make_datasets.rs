use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const DEFAULT_OUT: &str = "/mnt/data/falx-bench";
const DEFAULT_SIZE: usize = 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dataset {
    Csv,
    CsvGeo,
    CsvHash,
    Tsv,
    Logfmt,
    Ndjson,
    Vcf,
    Fastq,
}

impl Dataset {
    fn name(self) -> &'static str {
        match self {
            Dataset::Csv => "csv",
            Dataset::CsvGeo => "csv-geo",
            Dataset::CsvHash => "csv-hash",
            Dataset::Tsv => "tsv",
            Dataset::Logfmt => "logfmt",
            Dataset::Ndjson => "ndjson",
            Dataset::Vcf => "vcf",
            Dataset::Fastq => "fastq",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Dataset::Csv | Dataset::CsvGeo | Dataset::CsvHash => "csv",
            Dataset::Tsv => "tsv",
            Dataset::Logfmt => "logfmt",
            Dataset::Ndjson => "ndjson",
            Dataset::Vcf => "vcf",
            Dataset::Fastq => "fastq",
        }
    }
}

const ALL_DATASETS: &[Dataset] = &[
    Dataset::Csv,
    Dataset::CsvGeo,
    Dataset::CsvHash,
    Dataset::Tsv,
    Dataset::Logfmt,
    Dataset::Ndjson,
    Dataset::Vcf,
    Dataset::Fastq,
];

struct Options {
    out_dir: PathBuf,
    size: usize,
    size_label: String,
    formats: Vec<Dataset>,
}

fn main() -> io::Result<()> {
    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) if message == "help" => {
            print_usage();
            return Ok(());
        }
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            std::process::exit(2);
        }
    };

    fs::create_dir_all(&options.out_dir)?;
    for dataset in options.formats {
        let path = options.out_dir.join(format!(
            "{}-{}.{}",
            dataset.name(),
            options.size_label,
            dataset.extension()
        ));
        let bytes = write_dataset(dataset, &path, options.size)?;
        println!(
            "wrote {} ({:.2} GiB)",
            path.display(),
            bytes as f64 / 1073741824.0
        );
    }

    Ok(())
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --release --example make_datasets -- \
         [--out /mnt/data/falx-bench] [--size 1g] \
         [--formats all|csv,csv-geo,csv-hash,tsv,logfmt,ndjson,vcf,fastq]"
    );
}

fn parse_args<I>(args: I) -> Result<Options, String>
where
    I: IntoIterator<Item = String>,
{
    let mut out_dir = PathBuf::from(DEFAULT_OUT);
    let mut size = DEFAULT_SIZE;
    let mut size_label = "1g".to_string();
    let mut formats = ALL_DATASETS.to_vec();

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                let value = args.next().ok_or("--out requires a directory")?;
                out_dir = PathBuf::from(value);
            }
            "--size" => {
                let value = args.next().ok_or("--size requires a value")?;
                size = parse_size(&value)?;
                size_label = value.to_ascii_lowercase();
            }
            "--formats" => {
                let value = args.next().ok_or("--formats requires a value")?;
                formats = parse_formats(&value)?;
            }
            "--help" | "-h" => return Err("help".into()),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }

    Ok(Options {
        out_dir,
        size,
        size_label,
        formats,
    })
}

fn parse_size(value: &str) -> Result<usize, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("size must not be empty".into());
    }
    let last = value.as_bytes()[value.len() - 1].to_ascii_lowercase();
    let (digits, multiplier) = match last {
        b'k' => (&value[..value.len() - 1], 1024usize),
        b'm' => (&value[..value.len() - 1], 1024usize * 1024),
        b'g' => (&value[..value.len() - 1], 1024usize * 1024 * 1024),
        _ => (value, 1usize),
    };
    let base = digits
        .parse::<usize>()
        .map_err(|_| format!("invalid size '{value}'"))?;
    base.checked_mul(multiplier)
        .ok_or_else(|| format!("size '{value}' is too large"))
}

fn parse_formats(value: &str) -> Result<Vec<Dataset>, String> {
    if value.trim().eq_ignore_ascii_case("all") {
        return Ok(ALL_DATASETS.to_vec());
    }

    let mut formats = Vec::new();
    for raw in value.split(',') {
        let dataset = match raw.trim() {
            "csv" => Dataset::Csv,
            "csv-geo" | "geo" => Dataset::CsvGeo,
            "csv-hash" | "csvhash" | "hash" => Dataset::CsvHash,
            "tsv" => Dataset::Tsv,
            "logfmt" => Dataset::Logfmt,
            "ndjson" => Dataset::Ndjson,
            "vcf" => Dataset::Vcf,
            "fastq" => Dataset::Fastq,
            other => return Err(format!("unknown dataset format '{other}'")),
        };
        if !formats.contains(&dataset) {
            formats.push(dataset);
        }
    }
    if formats.is_empty() {
        return Err("at least one dataset format is required".into());
    }
    Ok(formats)
}

fn write_dataset(dataset: Dataset, path: &Path, target: usize) -> io::Result<u64> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut rng = Rng::for_dataset(dataset);
    let mut bytes = 0u64;
    let mut record = Vec::with_capacity(1024);
    let mut row = 0u64;

    while bytes < target as u64 {
        record.clear();
        match dataset {
            Dataset::Csv => csv_record(&mut rng, &mut record),
            Dataset::CsvGeo => csv_geo_record(&mut rng, row, &mut record),
            Dataset::CsvHash => csv_hash_record(&mut rng, row, &mut record),
            Dataset::Tsv => tsv_record(&mut rng, &mut record),
            Dataset::Logfmt => logfmt_record(&mut rng, &mut record),
            Dataset::Ndjson => ndjson_record(&mut rng, &mut record),
            Dataset::Vcf => vcf_record(&mut rng, row, &mut record),
            Dataset::Fastq => fastq_record(&mut rng, row, &mut record),
        }
        writer.write_all(&record)?;
        bytes += record.len() as u64;
        row += 1;
    }
    writer.flush()?;
    Ok(bytes)
}

struct Rng(u64);

impl Rng {
    fn for_dataset(dataset: Dataset) -> Self {
        let seed = match dataset {
            Dataset::Csv => 0x9E37_79B9_7F4A_7C15,
            Dataset::CsvGeo => 0xDEAD_BEEF_CAFE_BABE,
            Dataset::CsvHash => 0xC0FF_EE15_600D_F00D,
            Dataset::Tsv => 0xA076_1D64_78BD_642F,
            Dataset::Logfmt => 0xE703_7ED1_A0B4_28DB,
            Dataset::Ndjson => 0x853C_49E6_748F_EA9B,
            Dataset::Vcf => 0xD1B5_4A32_D192_ED03,
            Dataset::Fastq => 0x0000_0000_5EED_FA57,
        };
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    fn range_f64(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.uniform()
    }

    fn alnum(&mut self, min: usize, max: usize, out: &mut Vec<u8>) {
        let len = min + self.below((max - min) as u64) as usize;
        for _ in 0..len {
            let c = self.below(36);
            out.push(if c < 26 {
                b'a' + c as u8
            } else {
                b'0' + (c - 26) as u8
            });
        }
    }

    fn word(&mut self, min: usize, max: usize, out: &mut Vec<u8>) {
        let len = min + self.below((max - min) as u64) as usize;
        for _ in 0..len {
            out.push(b'a' + self.below(26) as u8);
        }
    }
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(value.to_string().as_bytes());
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(format!("{value:.6}").as_bytes());
}

fn csv_record(rng: &mut Rng, out: &mut Vec<u8>) {
    for field in 0..8 {
        if field > 0 {
            out.push(b',');
        }
        match rng.below(10) {
            0 => {
                out.push(b'"');
                if rng.below(2) == 0 {
                    out.extend_from_slice(b"a,b");
                }
                if rng.below(2) == 0 {
                    out.extend_from_slice(b"x\ny");
                }
                if rng.below(2) == 0 {
                    out.extend_from_slice(b"q\"\"q");
                }
                rng.alnum(1, 6, out);
                out.push(b'"');
            }
            1 | 2 => push_u64(out, rng.below(1_000_000_000)),
            _ => rng.alnum(3, 13, out),
        }
    }
    out.push(b'\n');
}

/// CSV with `#` comments and quotes — the dialect that exercises the `Regions`
/// resolver. A clustered header plus ~2% interspersed comment lines (carrying
/// commas, quotes, and `#` that the resolver must treat as inert), and a quoted
/// body with embedded commas and doubled quotes. No embedded newlines, so a `#`
/// at column 0 is an unambiguous comment — letting the `csv` crate's
/// `comment(Some(b'#'))` reader produce byte-identical fields for a fair row.
fn csv_hash_record(rng: &mut Rng, row: u64, out: &mut Vec<u8>) {
    if row < 8 || row.is_multiple_of(50) {
        out.extend_from_slice(b"# ");
        rng.word(3, 10, out);
        out.extend_from_slice(b", \"quoted, noise\" #section ");
        rng.alnum(2, 8, out);
        out.push(b'\n');
        return;
    }
    for field in 0..8 {
        if field > 0 {
            out.push(b',');
        }
        match rng.below(10) {
            0 => {
                out.push(b'"');
                if rng.below(2) == 0 {
                    out.extend_from_slice(b"a,b");
                }
                if rng.below(2) == 0 {
                    out.extend_from_slice(b"q\"\"q");
                }
                rng.alnum(1, 6, out);
                out.push(b'"');
            }
            1 | 2 => push_u64(out, rng.below(1_000_000_000)),
            _ => rng.alnum(3, 13, out),
        }
    }
    out.push(b'\n');
}

fn csv_geo_record(rng: &mut Rng, row: u64, out: &mut Vec<u8>) {
    if row == 0 {
        out.extend_from_slice(b"Country,City,AccentCity,Region,Population,Latitude,Longitude\n");
        return;
    }

    const COUNTRIES: &[&[u8]] = &[
        b"US", b"CA", b"GB", b"FR", b"DE", b"IT", b"ES", b"AU", b"JP", b"CN", b"IN", b"BR", b"MX",
        b"RU", b"ZA", b"NG", b"EG", b"TR", b"SA", b"AE", b"KR", b"SG", b"NZ", b"AR", b"CL", b"CO",
        b"PE", b"VE", b"NL", b"BE",
    ];
    out.extend_from_slice(COUNTRIES[rng.below(COUNTRIES.len() as u64) as usize]);
    out.push(b',');
    rng.word(4, 13, out);
    out.push(b',');
    rng.word(4, 13, out);
    out.push(b',');
    out.push(b'A' + rng.below(26) as u8);
    out.push(b'A' + rng.below(26) as u8);
    out.push(b',');
    if rng.below(2) == 0 {
        push_u64(out, rng.below(100_000_000));
    }
    out.push(b',');
    push_f64(out, rng.range_f64(-90.0, 90.0));
    out.push(b',');
    push_f64(out, rng.range_f64(-180.0, 180.0));
    out.push(b'\n');
}

fn tsv_record(rng: &mut Rng, out: &mut Vec<u8>) {
    for field in 0..8 {
        if field > 0 {
            out.push(b'\t');
        }
        rng.alnum(3, 13, out);
    }
    out.push(b'\n');
}

fn logfmt_record(rng: &mut Rng, out: &mut Vec<u8>) {
    const LEVELS: &[&[u8]] = &[b"info", b"warn", b"error", b"debug"];
    out.extend_from_slice(b"ts=2026-06-10T12:00:00Z level=");
    out.extend_from_slice(LEVELS[rng.below(LEVELS.len() as u64) as usize]);
    out.extend_from_slice(b" msg=\"");
    rng.alnum(4, 10, out);
    match rng.below(4) {
        0 => out.extend_from_slice(b" with \\\"quoted\\\" part"),
        1 => out.extend_from_slice(b" path C:\\\\temp"),
        _ => out.extend_from_slice(b" plain text here"),
    }
    out.extend_from_slice(b"\" dur=");
    rng.alnum(2, 4, out);
    for _ in 0..rng.below(4) {
        out.push(b' ');
        rng.alnum(3, 8, out);
        out.push(b'=');
        rng.alnum(1, 10, out);
    }
    out.push(b'\n');
}

fn ndjson_record(rng: &mut Rng, out: &mut Vec<u8>) {
    out.extend_from_slice(b"{\"id\":");
    push_u64(out, rng.below(1_000_000_000));
    out.extend_from_slice(b",\"name\":\"");
    rng.alnum(4, 12, out);
    match rng.below(4) {
        0 => out.extend_from_slice(b" \\\"nick\\\""),
        1 => out.extend_from_slice(b"\\\\share"),
        _ => {}
    }
    out.extend_from_slice(b"\",\"tags\":[\"");
    rng.alnum(2, 6, out);
    out.extend_from_slice(b"\",\"");
    rng.alnum(2, 6, out);
    out.extend_from_slice(b"\"],\"nested\":{\"score\":");
    push_u64(out, rng.below(100_000));
    out.extend_from_slice(b",\"ok\":true},\"note\":\"");
    rng.alnum(8, 24, out);
    out.extend_from_slice(b"\"}\n");
}

fn vcf_record(rng: &mut Rng, row: u64, out: &mut Vec<u8>) {
    if row == 0 {
        out.extend_from_slice(b"##fileformat=VCFv4.3\n");
        out.extend_from_slice(b"##source=falx-make-datasets\n");
        out.extend_from_slice(b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");
        return;
    }
    out.extend_from_slice(b"chr");
    push_u64(out, 1 + rng.below(22));
    out.push(b'\t');
    push_u64(out, 1 + rng.below(250_000_000));
    out.extend_from_slice(b"\trs");
    push_u64(out, row);
    out.push(b'\t');
    const BASES: &[u8] = b"ACGT";
    let reference = BASES[rng.below(4) as usize];
    let mut alternate = BASES[rng.below(4) as usize];
    if alternate == reference {
        alternate = BASES[((rng.below(3) + 1) % 4) as usize];
    }
    out.push(reference);
    out.push(b'\t');
    out.push(alternate);
    out.push(b'\t');
    push_f64(out, rng.range_f64(10.0, 99.0));
    out.extend_from_slice(b"\tPASS\tDP=");
    push_u64(out, 10 + rng.below(500));
    out.extend_from_slice(b";AF=");
    push_f64(out, rng.range_f64(0.0, 1.0));
    out.push(b'\n');
}

fn fastq_record(rng: &mut Rng, row: u64, out: &mut Vec<u8>) {
    const BASES: &[u8] = b"ACGT";
    const QUALS: &[u8] = b"!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJ";
    let len = 80 + rng.below(80) as usize;
    out.extend_from_slice(b"@read");
    push_u64(out, row);
    out.extend_from_slice(b" flowcell:1:lane:2\n");
    for _ in 0..len {
        out.push(BASES[rng.below(4) as usize]);
    }
    out.extend_from_slice(b"\n+\n");
    for _ in 0..len {
        out.push(QUALS[rng.below(QUALS.len() as u64) as usize]);
    }
    out.push(b'\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_size_accepts_binary_units() {
        assert_eq!(parse_size("64m").unwrap(), 64 * 1024 * 1024);
        assert_eq!(parse_size("2g").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("123").unwrap(), 123);
    }

    #[test]
    fn parse_formats_accepts_all_or_csv_list() {
        assert!(parse_formats("all").unwrap().contains(&Dataset::Csv));
        assert_eq!(
            parse_formats("csv,ndjson,fastq").unwrap(),
            vec![Dataset::Csv, Dataset::Ndjson, Dataset::Fastq]
        );
    }

    #[test]
    fn csv_geo_dataset_starts_with_header() {
        let path =
            std::env::temp_dir().join(format!("falx-csv-geo-{}-{}.csv", std::process::id(), 128));
        let _ = std::fs::remove_file(&path);
        write_dataset(Dataset::CsvGeo, &path, 128).unwrap();
        let data = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(
            data.starts_with(b"Country,City,AccentCity,Region,Population,Latitude,Longitude\n")
        );
    }
}
