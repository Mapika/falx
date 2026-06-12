//! Throughput benchmark for structural indexing of backslash-escape-heavy data
//! across the three escape-handling format kernels: ndjson, logfmt, and json.
//!
//! For each format: ~64 MiB of synthetic data with frequent backslash escapes
//! (roughly 1 in 4 string characters is an escape sequence), 1 warmup + 9 timed
//! runs per kernel. Best run and median are reported in GiB/s, along with
//! structural count to verify consistency across runs.

use std::hint::black_box;
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn alnum(&mut self, len_range: std::ops::Range<u64>) -> Vec<u8> {
        let len = len_range.start + self.below(len_range.end - len_range.start);
        (0..len)
            .map(|_| {
                let c = self.below(36);
                if c < 26 {
                    b'a' + c as u8
                } else {
                    b'0' + (c - 26) as u8
                }
            })
            .collect()
    }
}

const TARGET_BYTES: usize = 64 * 1024 * 1024;

/// Generate ndjson data with escape-heavy strings:
/// {"key":"alnum...","msg":"text with \\ and \\\\ escapes","n":12345}\n
fn generate_ndjson_escape_heavy(target: usize) -> Vec<u8> {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut data = Vec::with_capacity(target + 1024);
    while data.len() < target {
        data.extend_from_slice(b"{\"key\":\"");
        data.extend_from_slice(&rng.alnum(5..12));
        data.extend_from_slice(b"\",\"msg\":\"");

        // Escape-heavy message: roughly 1 in 4 chars is an escape sequence.
        let msg_len = 20 + rng.below(40);
        for _ in 0..msg_len {
            match rng.below(4) {
                0 => data.extend_from_slice(b"\\\""), // \" escape
                1 => data.extend_from_slice(b"\\\\"), // \\ escape
                _ => {
                    let c = rng.alnum(1..2);
                    data.extend_from_slice(&c);
                }
            }
        }

        data.extend_from_slice(b"\",\"n\":");
        data.extend_from_slice(&rng.below(100000).to_string().into_bytes());
        data.extend_from_slice(b"}\n");
    }
    data
}

/// Generate logfmt data with escape-heavy quoted values:
/// ts=12345 level=info msg="quoted value with \\" escapes" key=value\n
fn generate_logfmt_escape_heavy(target: usize) -> Vec<u8> {
    let mut rng = Rng(0xE703_7ED1_A0B4_28DB);
    let mut data = Vec::with_capacity(target + 1024);
    let levels: [&[u8]; 4] = [b"info", b"warn", b"error", b"debug"];
    while data.len() < target {
        data.extend_from_slice(b"ts=");
        data.extend_from_slice(&rng.below(1000000).to_string().into_bytes());
        data.extend_from_slice(b" level=");
        data.extend_from_slice(levels[rng.below(4) as usize]);
        data.extend_from_slice(b" msg=\"");

        // Escape-heavy quoted value: roughly 1 in 4 chars is an escape.
        let val_len = 25 + rng.below(50);
        for _ in 0..val_len {
            match rng.below(4) {
                0 => data.extend_from_slice(b"\\\""), // \" escape
                1 => data.extend_from_slice(b"\\\\"), // \\ escape
                _ => {
                    let c = rng.alnum(1..2);
                    data.extend_from_slice(&c);
                }
            }
        }

        data.extend_from_slice(b"\" key=");
        data.extend_from_slice(&rng.alnum(3..8));
        data.push(b'\n');
    }
    data
}

/// Generate json array with escape-heavy object values:
/// [{"key":"...","val":"text with \\ and \\\\ escapes"},...]
fn generate_json_escape_heavy(target: usize) -> Vec<u8> {
    let mut rng = Rng(0x853C_49E6_748F_EA9B);
    let mut data = Vec::with_capacity(target + 1024);
    data.push(b'[');
    let mut first = true;
    while data.len() < target {
        if !first {
            data.push(b',');
        }
        first = false;

        data.extend_from_slice(b"{\"key\":\"");
        data.extend_from_slice(&rng.alnum(4..10));
        data.extend_from_slice(b"\",\"val\":\"");

        // Escape-heavy value: roughly 1 in 4 chars is an escape.
        let val_len = 30 + rng.below(60);
        for _ in 0..val_len {
            match rng.below(4) {
                0 => data.extend_from_slice(b"\\\""), // \" escape
                1 => data.extend_from_slice(b"\\\\"), // \\ escape
                _ => {
                    let c = rng.alnum(1..2);
                    data.extend_from_slice(&c);
                }
            }
        }

        data.extend_from_slice(b"\"}");
    }
    data.push(b']');
    data
}

struct Measurement {
    best_us: u128,
    median_us: u128,
    work: usize,
}

/// 1 warmup + 9 timed runs; returns a work count so results stay live.
fn measure(mut run: impl FnMut() -> usize) -> Measurement {
    let work = black_box(run());
    let mut times: Vec<u128> = (0..9)
        .map(|_| {
            let start = Instant::now();
            black_box(run());
            start.elapsed().as_micros()
        })
        .collect();
    times.sort_unstable();
    Measurement {
        best_us: times[0],
        median_us: times[times.len() / 2],
        work,
    }
}

fn measure_indexer(data: &[u8], f: fn(&[u8], &mut Vec<u32>)) -> Measurement {
    let mut out: Vec<u32> = Vec::with_capacity(data.len() / 4);
    measure(move || {
        out.clear();
        f(data, &mut out);
        out.len()
    })
}

struct Row {
    label: &'static str,
    m: Measurement,
}

fn report(format: &str, bytes: usize, rows: &[Row]) {
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "\n== {format} ({:.0} MiB) ==",
        bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "{:<20} {:>10} {:>12} {:>10}",
        "kernel", "best (ms)", "median (ms)", "GiB/s"
    );
    for row in rows {
        println!(
            "{:<20} {:>10.2} {:>12.2} {:>10.2}",
            row.label,
            row.m.best_us as f64 / 1000.0,
            row.m.median_us as f64 / 1000.0,
            gib / (row.m.best_us as f64 / 1_000_000.0),
        );
    }
}

/// Cross-check that the indexers found identical structural counts.
fn check_counts(format: &str, rows: &[Row]) {
    let first = rows[0].m.work;
    for row in rows {
        assert_eq!(
            row.m.work, first,
            "{format}: {} found {} items, expected {first}",
            row.label, row.m.work
        );
    }
    println!("   [{format}: counts agree, {first} structural positions]");
}

fn main() {
    println!(
        "Structural indexing benchmark: escape-heavy data across ndjson, logfmt, json kernels\n"
    );

    // NDJSON benchmark
    let ndjson_data = generate_ndjson_escape_heavy(TARGET_BYTES);
    let ndjson_rows = vec![Row {
        label: "ndjson",
        m: measure_indexer(&ndjson_data, falx::kernels::ndjson::index_structurals),
    }];
    report("ndjson", ndjson_data.len(), &ndjson_rows);
    check_counts("ndjson", &ndjson_rows);

    // Logfmt benchmark
    let logfmt_data = generate_logfmt_escape_heavy(TARGET_BYTES);
    let logfmt_rows = vec![Row {
        label: "logfmt",
        m: measure_indexer(&logfmt_data, falx::kernels::logfmt::index_structurals),
    }];
    report("logfmt", logfmt_data.len(), &logfmt_rows);
    check_counts("logfmt", &logfmt_rows);

    // JSON benchmark
    let json_data = generate_json_escape_heavy(TARGET_BYTES);
    let json_rows = vec![Row {
        label: "json",
        m: measure_indexer(&json_data, falx::kernels::json::index_structurals),
    }];
    report("json", json_data.len(), &json_rows);
    check_counts("json", &json_rows);
}
