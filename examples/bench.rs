//! Throughput benchmark across all generated format kernels.
//!
//! For each format: ~64 MiB of synthetic data, 1 warmup + 7 timed runs per
//! parser, best run reported (median printed for noise visibility). The
//! generated AVX2 kernel is compared against the generated portable
//! fallback, the hand-written M0 kernel (CSV only — the codegen fidelity
//! check), and an ecosystem baseline where a fair one exists (csv crate for
//! CSV, serde_json line-parsing for NDJSON). Baselines do more work than
//! structural indexing (they materialize values); the comparison shows the
//! headroom indexing creates, not a like-for-like parse.

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
                if c < 26 { b'a' + c as u8 } else { b'0' + (c - 26) as u8 }
            })
            .collect()
    }
}

const TARGET_BYTES: usize = 64 * 1024 * 1024;

fn generate_csv(target: usize) -> Vec<u8> {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut data = Vec::with_capacity(target + 1024);
    while data.len() < target {
        for field in 0..8 {
            if field > 0 {
                data.push(b',');
            }
            match rng.below(10) {
                0 => {
                    // Quoted field with embedded structure.
                    data.push(b'"');
                    if rng.below(2) == 0 {
                        data.extend_from_slice(b"a,b");
                    }
                    if rng.below(2) == 0 {
                        data.extend_from_slice(b"x\ny");
                    }
                    if rng.below(2) == 0 {
                        data.extend_from_slice(b"q\"\"q"); // escaped quote
                    }
                    data.extend_from_slice(&rng.alnum(1..6));
                    data.push(b'"');
                }
                1 | 2 => {
                    let digits = 1 + rng.below(9);
                    for _ in 0..digits {
                        data.push(b'0' + rng.below(10) as u8);
                    }
                }
                _ => data.extend_from_slice(&rng.alnum(3..13)),
            }
        }
        data.push(b'\n');
    }
    data
}

fn generate_tsv(target: usize) -> Vec<u8> {
    let mut rng = Rng(0xA076_1D64_78BD_642F);
    let mut data = Vec::with_capacity(target + 1024);
    while data.len() < target {
        for field in 0..8 {
            if field > 0 {
                data.push(b'\t');
            }
            data.extend_from_slice(&rng.alnum(3..13));
        }
        data.push(b'\n');
    }
    data
}

fn generate_logfmt(target: usize) -> Vec<u8> {
    let mut rng = Rng(0xE703_7ED1_A0B4_28DB);
    let mut data = Vec::with_capacity(target + 1024);
    let levels: [&[u8]; 4] = [b"info", b"warn", b"error", b"debug"];
    while data.len() < target {
        data.extend_from_slice(b"ts=2026-06-10T12:00:00Z level=");
        data.extend_from_slice(levels[rng.below(4) as usize]);
        data.extend_from_slice(b" msg=\"");
        data.extend_from_slice(&rng.alnum(4..10));
        match rng.below(4) {
            0 => data.extend_from_slice(b" with \\\"quoted\\\" part"),
            1 => data.extend_from_slice(b" path C:\\\\temp"),
            _ => data.extend_from_slice(b" plain text here"),
        }
        data.extend_from_slice(b"\" dur=");
        data.extend_from_slice(&rng.alnum(2..4));
        for _ in 0..rng.below(4) {
            data.push(b' ');
            data.extend_from_slice(&rng.alnum(3..8));
            data.push(b'=');
            data.extend_from_slice(&rng.alnum(1..10));
        }
        data.push(b'\n');
    }
    data
}

fn generate_ndjson(target: usize) -> Vec<u8> {
    let mut rng = Rng(0x853C_49E6_748F_EA9B);
    let mut data = Vec::with_capacity(target + 1024);
    while data.len() < target {
        data.extend_from_slice(b"{\"id\":");
        data.extend_from_slice(&rng.below(1_000_000).to_string().into_bytes());
        data.extend_from_slice(b",\"name\":\"");
        data.extend_from_slice(&rng.alnum(4..12));
        match rng.below(4) {
            0 => data.extend_from_slice(b" \\\"nick\\\""),
            1 => data.extend_from_slice(b"\\\\share"),
            _ => {}
        }
        data.extend_from_slice(b"\",\"tags\":[\"");
        data.extend_from_slice(&rng.alnum(2..6));
        data.extend_from_slice(b"\",\"");
        data.extend_from_slice(&rng.alnum(2..6));
        data.extend_from_slice(b"\"],\"note\":\"");
        data.extend_from_slice(&rng.alnum(8..24));
        data.extend_from_slice(b"\"}\n");
    }
    data
}

struct Measurement {
    best_us: u128,
    median_us: u128,
    work: usize,
}

/// 1 warmup + 7 timed runs; `run` returns a work count so results stay live.
fn measure(mut run: impl FnMut() -> usize) -> Measurement {
    let work = black_box(run());
    let mut times: Vec<u128> = (0..7)
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

fn report(format: &str, bytes: usize, baseline_label: &str, rows: &[Row]) {
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let baseline = rows
        .iter()
        .find(|r| r.label == baseline_label)
        .expect("baseline row present");
    println!(
        "\n== {format} ({:.0} MiB) ==",
        bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "{:<28} {:>10} {:>12} {:>10} {:>9}",
        "parser", "best (ms)", "median (ms)", "GiB/s", "speedup"
    );
    for row in rows {
        println!(
            "{:<28} {:>10.2} {:>12.2} {:>10.2} {:>8.2}x",
            row.label,
            row.m.best_us as f64 / 1000.0,
            row.m.median_us as f64 / 1000.0,
            gib / (row.m.best_us as f64 / 1_000_000.0),
            baseline.m.best_us as f64 / row.m.best_us as f64,
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
    // ---- CSV: generated vs hand-written vs scalar vs csv crate ----
    let data = generate_csv(TARGET_BYTES);
    let mut rows = vec![
        Row {
            label: "generated kernel (AVX2)",
            m: measure_indexer(&data, falx::kernels::csv::index_structurals),
        },
        Row {
            label: "hand-written M0 kernel",
            m: measure_indexer(&data, falx::index_structurals),
        },
        Row {
            label: "generated fallback (scalar)",
            m: measure_indexer(&data, falx::kernels::csv::fallback::index_structurals),
        },
    ];
    rows.push(Row {
        label: "generated parallel x16",
        m: {
            let mut out: Vec<u32> = Vec::with_capacity(data.len() / 4);
            let data = &data;
            measure(move || {
                out.clear();
                falx::kernels::csv::index_structurals_par(data, 16, &mut out);
                out.len()
            })
        },
    });
    check_counts("csv", &rows);
    rows.push(Row {
        label: "csv crate (full parse)",
        m: measure(|| {
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(&data[..]);
            let mut fields = 0usize;
            for record in reader.byte_records() {
                fields += record.expect("valid csv").len();
            }
            fields
        }),
    });
    report("CSV", data.len(), "csv crate (full parse)", &rows);

    // ---- CSV field iteration: like-for-like (spans + quote/escape
    // cleaning vs the csv crate's materialized records) ----
    let falx_fields = measure(|| {
        let parsed = falx::kernels::csv::parse(&data);
        let mut total = 0usize;
        for record in parsed.records() {
            for field in record.fields() {
                total += field.len();
            }
        }
        total
    });
    // Steady state: recycle the tape buffers across runs, as a per-batch
    // caller would. Fresh ~40 MB tapes cost soft page faults every parse.
    let mut recycled = Some(falx::kernels::csv::parse(&data));
    let falx_fields_warm = measure(|| {
        let parsed = falx::kernels::csv::parse_into(&data, recycled.take().expect("recycled"));
        let mut total = 0usize;
        for record in parsed.records() {
            for field in record.fields() {
                total += field.len();
            }
        }
        recycled = Some(parsed);
        total
    });
    assert_eq!(falx_fields.work, falx_fields_warm.work);
    let csv_fields = measure(|| {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(&data[..]);
        let mut total = 0usize;
        for record in reader.byte_records() {
            for field in record.expect("valid csv").iter() {
                total += field.len();
            }
        }
        total
    });
    assert_eq!(
        falx_fields.work, csv_fields.work,
        "falx and csv crate disagree on total field bytes"
    );
    println!(
        "   [csv fields: byte totals agree, {} bytes across all fields]",
        falx_fields.work
    );
    // Parallel field iteration: chunk the record tape across threads
    // (records_range gives O(1) disjoint chunks).
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let parallel_fields = measure(|| {
        let parsed = falx::kernels::csv::parse_par(&data, threads);
        let n = parsed.terminated_record_count();
        let chunk = n.div_ceil(threads).max(1);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let parsed = &parsed;
                    s.spawn(move || {
                        let range = (t * chunk).min(n)..((t + 1) * chunk).min(n);
                        let mut total = 0usize;
                        for record in parsed.records_range(range) {
                            for field in record.fields() {
                                total += field.len();
                            }
                        }
                        total
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("thread ok"))
                .sum()
        })
    });
    assert_eq!(
        parallel_fields.work, csv_fields.work,
        "parallel chunked iteration disagrees on total field bytes"
    );

    // Streaming: 64 KiB feeds through the incremental parser.
    let streaming_fields = measure(|| {
        let mut parser = falx::kernels::csv::stream();
        let mut total = 0usize;
        for chunk in data.chunks(64 * 1024) {
            parser.feed(chunk, |record| {
                for field in record.fields() {
                    total += field.len();
                }
            });
        }
        parser.finish(|record| {
            for field in record.fields() {
                total += field.len();
            }
        });
        total
    });
    assert_eq!(
        streaming_fields.work, csv_fields.work,
        "streaming disagrees on total field bytes"
    );

    let parallel_label: &'static str =
        Box::leak(format!("falx parallel parse+fields x{threads}").into_boxed_str());
    let rows = vec![
        Row {
            label: "falx parse+fields",
            m: falx_fields,
        },
        Row {
            label: "falx parse_into+fields (recycled tape)",
            m: falx_fields_warm,
        },
        Row {
            label: parallel_label,
            m: parallel_fields,
        },
        Row {
            label: "falx streaming (64 KiB feeds)",
            m: streaming_fields,
        },
        Row {
            label: "csv crate byte_records",
            m: csv_fields,
        },
    ];
    report(
        "CSV field iteration (like-for-like)",
        data.len(),
        "csv crate byte_records",
        &rows,
    );

    // ---- TSV ----
    let data = generate_tsv(TARGET_BYTES);
    let rows = vec![
        Row {
            label: "generated kernel (AVX2)",
            m: measure_indexer(&data, falx::kernels::tsv::index_structurals),
        },
        Row {
            label: "generated fallback (scalar)",
            m: measure_indexer(&data, falx::kernels::tsv::fallback::index_structurals),
        },
    ];
    let rows = {
        let mut rows = rows;
        rows.push(Row {
            label: "generated parallel x16",
            m: {
                let mut out: Vec<u32> = Vec::with_capacity(data.len() / 4);
                let data = &data;
                measure(move || {
                    out.clear();
                    falx::kernels::tsv::index_structurals_par(data, 16, &mut out);
                    out.len()
                })
            },
        });
        rows
    };
    check_counts("tsv", &rows);
    report("TSV", data.len(), "generated fallback (scalar)", &rows);

    // ---- logfmt ----
    let data = generate_logfmt(TARGET_BYTES);
    let rows = vec![
        Row {
            label: "generated kernel (AVX2)",
            m: measure_indexer(&data, falx::kernels::logfmt::index_structurals),
        },
        Row {
            label: "generated fallback (scalar)",
            m: measure_indexer(&data, falx::kernels::logfmt::fallback::index_structurals),
        },
    ];
    check_counts("logfmt", &rows);
    report("logfmt", data.len(), "generated fallback (scalar)", &rows);

    // ---- NDJSON: framing kernel vs serde_json full parse ----
    let data = generate_ndjson(TARGET_BYTES);
    let mut rows = vec![
        Row {
            label: "generated kernel (AVX2)",
            m: measure_indexer(&data, falx::kernels::ndjson::index_structurals),
        },
        Row {
            label: "generated fallback (scalar)",
            m: measure_indexer(&data, falx::kernels::ndjson::fallback::index_structurals),
        },
    ];
    check_counts("ndjson", &rows);
    rows.push(Row {
        label: "serde_json (full parse)",
        m: measure(|| {
            let mut count = 0;
            for line in data.split(|&b| b == b'\n')
                .filter(|line| !line.is_empty())
            {
                serde_json::from_slice::<serde_json::Value>(line).expect("valid json");
                count += 1;
            }
            count
        }),
    });
    // simd-json (Rust port of simdjson, whose techniques falx generates
    // from) builds a full tape per document, where falx only frames records
    // — so this comparison shows what skipping in-document parsing buys.
    // to_tape mutates its input, so each run parses a fresh copy; the
    // copy itself happens outside the timed region.
    rows.push(Row {
        label: "simd-json (tape)",
        m: {
            let mut scratch = data.clone();
            let mut times: Vec<u128> = Vec::new();
            let mut work = 0usize;
            for run in 0..8 {
                scratch.copy_from_slice(&data);
                let mut docs = 0usize;
                let start = Instant::now();
                let mut offset = 0usize;
                while offset < scratch.len() {
                    let end = scratch[offset..]
                        .iter()
                        .position(|&b| b == b'\n')
                        .map_or(scratch.len(), |p| offset + p);
                    if end > offset {
                        simd_json::to_tape(&mut scratch[offset..end]).expect("valid json");
                        docs += 1;
                    }
                    offset = end + 1;
                }
                let elapsed = start.elapsed().as_micros();
                black_box(docs);
                if run > 0 {
                    times.push(elapsed);
                }
                work = docs;
            }
            times.sort_unstable();
            Measurement {
                best_us: times[0],
                median_us: times[times.len() / 2],
                work,
            }
        },
    });
    report("NDJSON", data.len(), "serde_json (full parse)", &rows);
}
