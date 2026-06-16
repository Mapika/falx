//! Stage-by-stage cost breakdown of the CSV parse pipeline, for optimization
//! work. Usage: `cargo run --release --example stages -- [file.csv]`
//! (no argument: 64 MiB synthetic data).

use std::hint::black_box;
use std::time::Instant;

fn best_of<T>(n: usize, mut run: impl FnMut() -> T) -> (f64, T) {
    let value = run();
    let mut best = f64::MAX;
    for _ in 0..n {
        let start = Instant::now();
        black_box(run());
        best = best.min(start.elapsed().as_secs_f64());
    }
    (best, value)
}

fn main() {
    let data = match std::env::args().nth(1) {
        Some(path) => std::fs::read(&path).expect("readable input file"),
        None => synthetic(64 * 1024 * 1024),
    };
    let gib = data.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("input: {:.1} MiB", data.len() as f64 / (1024.0 * 1024.0));

    let mut out: Vec<u32> = Vec::with_capacity(data.len() / 8);
    let (t_index, n) = best_of(5, || {
        out.clear();
        falx::kernels::csv::index_structurals(&data, &mut out);
        out.len()
    });

    let (t_records, recs) = best_of(5, || falx::kernels::csv::parse(&data).records().count());

    let (t_raw, raw_bytes) = best_of(5, || {
        let parsed = falx::kernels::csv::parse(&data);
        let mut total = 0usize;
        for record in parsed.records() {
            for i in 0..record.field_count() {
                total += record.field_raw(i).expect("in range").len();
            }
        }
        total
    });

    let (t_clean, clean_bytes) = best_of(5, || {
        let parsed = falx::kernels::csv::parse(&data);
        let mut total = 0usize;
        for record in parsed.records() {
            for field in record.fields() {
                total += field.len();
            }
        }
        total
    });

    println!(
        "structurals: {n}, records: {recs}, raw bytes: {raw_bytes}, clean bytes: {clean_bytes}"
    );
    println!();
    println!("stage                          total ms     GiB/s    delta ms (stage cost)");
    let rows = [
        ("index only", t_index, t_index),
        ("+ records iteration", t_records, t_records - t_index),
        ("+ field_raw spans", t_raw, t_raw - t_records),
        ("+ clean (quotes/escapes)", t_clean, t_clean - t_raw),
    ];
    for (label, total, delta) in rows {
        println!(
            "{label:<30} {:>9.2} {:>9.2} {:>11.2}",
            total * 1000.0,
            gib / total,
            delta * 1000.0
        );
    }
}

fn synthetic(target: usize) -> Vec<u8> {
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut data = Vec::with_capacity(target + 1024);
    while data.len() < target {
        for field in 0..8 {
            if field > 0 {
                data.push(b',');
            }
            if next() % 10 == 0 {
                data.extend_from_slice(b"\"qu,oted\"\"x\"");
            } else {
                let len = 3 + next() % 10;
                for _ in 0..len {
                    data.push(b'a' + (next() % 26) as u8);
                }
            }
        }
        data.push(b'\n');
    }
    data
}
