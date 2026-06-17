//! Isolates the scalar region-resolver tax: csv vs csv_hash index_structurals
//! on the SAME bytes. Same SIMD compares; the only delta is that csv_hash
//! routes quote/comment context through resolve_regions while csv uses
//! branchless PrefixXor parity. csv is the unchanged-kernel control. Three
//! comment distributions: none (fast path B every block), clustered header
//! (realistic), interleaved (every other line a comment -> walk fallback).
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench(label: &str, data: &[u8], runs: usize) {
    let gib = data.len() as f64 / 1073741824.0;
    let mut csv_g = Vec::new();
    let mut hash_g = Vec::new();
    let mut out = Vec::with_capacity(data.len() / 8);
    falx::kernels::csv::index_structurals(data, &mut out);
    let csv_n = out.len();
    out.clear();
    falx::kernels::csv_hash::index_structurals(data, &mut out);
    let hash_n = out.len();
    for _ in 0..runs {
        out.clear();
        let t = Instant::now();
        falx::kernels::csv::index_structurals(data, &mut out);
        csv_g.push(gib / t.elapsed().as_secs_f64());
        std::hint::black_box(&out);
        out.clear();
        let t = Instant::now();
        falx::kernels::csv_hash::index_structurals(data, &mut out);
        hash_g.push(gib / t.elapsed().as_secs_f64());
        std::hint::black_box(&out);
    }
    let c = median(csv_g);
    let h = median(hash_g);
    println!(
        "{:<28} csv {:>5.2}  csv_hash {:>5.2} GiB/s  tax {:.2}x  (csv structs {}, hash structs {})",
        label, c, h, c / h, csv_n, hash_n
    );
}

/// Insert `#`-comment lines into `base` per `every` data lines (0 = none).
fn with_comments(base: &[u8], every: usize) -> Vec<u8> {
    if every == 0 {
        return base.to_vec();
    }
    let mut out = Vec::with_capacity(base.len() + base.len() / 8);
    let mut line = 0usize;
    for chunk in base.split_inclusive(|&b| b == b'\n') {
        out.extend_from_slice(chunk);
        line += 1;
        if line.is_multiple_of(every) {
            out.extend_from_slice(b"# comment line, with \"a\" quote and, commas\n");
        }
    }
    out
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/data/falx-bench/csv-1g.csv".to_string());
    let data = std::fs::read(&path).expect("read data");
    // Cap working sets to ~256 MiB so the synthetic variants stay in budget.
    let cap = (256 * 1024 * 1024).min(data.len());
    // snap to a line boundary
    let mut end = cap;
    while end < data.len() && data[end - 1] != b'\n' {
        end += 1;
    }
    let base = &data[..end];
    let runs = 9;
    println!("base: {} ({:.0} MiB quoted CSV body)\n", path, base.len() as f64 / 1048576.0);

    bench("none (fast path B)", base, runs);
    let header = with_comments(base, 0); // then prepend a header block
    let mut clustered = Vec::with_capacity(header.len() + 4096);
    for _ in 0..200 {
        clustered.extend_from_slice(b"# clustered header comment with \"quotes\", commas\n");
    }
    clustered.extend_from_slice(base);
    bench("clustered header (realistic)", &clustered, runs);

    let inter = with_comments(base, 4);
    bench("every 4th line (heavy)", &inter, runs);

    let inter2 = with_comments(base, 1);
    bench("every line (pathological)", &inter2, runs);
}
