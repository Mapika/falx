//! Parallel NDJSON integer-sum (the json_parity task: sum every integer-typed
//! value, generically). NDJSON is newline-delimited so it shards trivially: split
//! the buffer into `threads` line-aligned chunks, each thread runs the flat
//! scalar scan, sum the partials. Same slice json_parity uses (first 256 MiB to a
//! newline) by default, or the whole file with arg `full`.
//!
//! Run: cargo run --release --example json_sum_par -- <file.ndjson> [iters] [threads] [full]

use std::hint::black_box;
use std::time::{Duration, Instant};

use falx::kernels::json;
use falx::kernels::json::{Nested, parse_nested, parse_nested_into};
use falx::kernels::ndjson;

#[inline]
fn trim_ws(d: &[u8], mut s: usize, mut e: usize) -> (usize, usize) {
    while s < e && matches!(d[s], b' ' | b'\t' | b'\n' | b'\r') {
        s += 1;
    }
    while e > s && matches!(d[e - 1], b' ' | b'\t' | b'\n' | b'\r') {
        e -= 1;
    }
    (s, e)
}

/// Fused integer-sum: one whole-chunk SIMD `index_structurals` pass, then a single
/// gap-walk summing integer scalars — no per-line `parse_nested` tape, no bracket
/// matching. (Same scalar spans `Nested::scalars()` yields, which only uses the
/// structural *positions*, not the hierarchy.)
fn sum_chunk_fused(data: &[u8], pos: &mut Vec<u32>) -> i64 {
    pos.clear();
    json::index_structurals(data, pos);
    let mut sum = 0i64;
    let mut cursor = 0usize;
    for &p in pos.iter() {
        let p = p as usize;
        let (s, e) = trim_ws(data, cursor, p);
        cursor = p + 1;
        if s < e
            && let Some(n) = parse_int(&data[s..e])
        {
            sum = sum.wrapping_add(n);
        }
    }
    let (s, e) = trim_ws(data, cursor, data.len());
    if s < e
        && let Some(n) = parse_int(&data[s..e])
    {
        sum = sum.wrapping_add(n);
    }
    sum
}

#[inline]
fn parse_int(b: &[u8]) -> Option<i64> {
    let (neg, digits) = match b.first()? {
        b'-' => (true, &b[1..]),
        _ => (false, b),
    };
    if digits.is_empty() || !digits[0].is_ascii_digit() {
        return None;
    }
    let mut v: i64 = 0;
    for &c in digits {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v.wrapping_mul(10).wrapping_add((c - b'0') as i64);
    }
    Some(if neg { -v } else { v })
}

/// Flat scalar scan of one chunk (whole NDJSON lines), recycling the tape buffer.
fn sum_chunk(data: &[u8]) -> i64 {
    let mut nested: Nested = parse_nested(b"");
    let mut sum = 0i64;
    for line in data.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        nested = parse_nested_into(line, nested);
        for scalar in nested.scalars() {
            if let Some(n) = parse_int(scalar) {
                sum = sum.wrapping_add(n);
            }
        }
    }
    sum
}

/// Split `[0,len)` into `threads` ranges, each extended forward to just past a
/// newline so every line belongs to exactly one chunk.
fn line_chunks(data: &[u8], threads: usize) -> Vec<(usize, usize)> {
    let n = data.len();
    let mut bounds = vec![0usize];
    for t in 1..threads {
        let mut b = (t * n) / threads;
        while b < n && data[b - 1] != b'\n' {
            b += 1;
        }
        bounds.push(b.min(n));
    }
    bounds.push(n);
    bounds.dedup();
    bounds.windows(2).map(|w| (w[0], w[1])).collect()
}

fn sum_par(data: &[u8], threads: usize) -> i64 {
    if threads <= 1 {
        return sum_chunk(data);
    }
    let chunks = line_chunks(data, threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|&(a, b)| s.spawn(move || sum_chunk(&data[a..b])))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn sum_par_fused(data: &[u8], threads: usize) -> i64 {
    if threads <= 1 {
        return sum_chunk_fused(data, &mut Vec::new());
    }
    let chunks = line_chunks(data, threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|&(a, b)| s.spawn(move || sum_chunk_fused(&data[a..b], &mut Vec::new())))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

/// Fused parallel sum where each worker allocates its OWN pre-sized position
/// buffer (local NUMA first-touch + no reallocation). Pre-allocating on the main
/// thread and reusing is actually *slower* here — Sapphire Rapids sub-NUMA makes
/// a main-thread buffer remote to the worker core.
fn sum_par_fused_local(data: &[u8], chunks: &[(usize, usize)]) -> i64 {
    std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|&(a, b)| {
                s.spawn(move || {
                    let mut buf = Vec::with_capacity((b - a) / 3 + 64);
                    sum_chunk_fused(&data[a..b], &mut buf)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn best(iters: usize, mut f: impl FnMut() -> Duration) -> Duration {
    let mut b = Duration::MAX;
    for _ in 0..iters {
        b = b.min(f());
    }
    b
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .expect("usage: json_sum_par <file.ndjson> [iters] [threads] [full]");
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(7);
    let threads: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(24);
    let full = a.next().as_deref() == Some("full");

    let raw = std::fs::read(&path).expect("read");
    // Match json_parity: first 256 MiB extended to a newline (unless `full`).
    let end = if full {
        raw.len()
    } else {
        let cap = (256 * 1024 * 1024).min(raw.len());
        let mut e = cap;
        while e < raw.len() && raw[e - 1] != b'\n' {
            e += 1;
        }
        e
    };
    let data = &raw[..end];
    let gib = data.len() as f64 / (1024.0 * 1024.0 * 1024.0);

    let s1 = sum_chunk(data);
    let sn = sum_par(data, threads);
    let sf1 = sum_chunk_fused(data, &mut Vec::new());
    let sfn = sum_par_fused(data, threads);
    let schema = ndjson::parse_ndjson_id_score(data);
    let schema_par = ndjson::parse_ndjson_id_score_par(data, threads);
    assert_eq!(s1, sn, "parallel sum diverges from serial");
    assert_eq!(s1, sf1, "fused sum diverges from flat");
    assert_eq!(s1, sfn, "fused parallel sum diverges");
    assert_eq!(schema.sum, s1, "schema-aware id+score sum diverges");
    assert_eq!(schema_par, schema, "schema-aware parallel sum diverges");
    println!("file: {path}  ({gib:.3} GiB)  best of {iters}\ninteger sum = {s1}\n");

    let t1 = best(iters, || {
        let t = Instant::now();
        black_box(sum_chunk(data));
        t.elapsed()
    });
    let tn = best(iters, || {
        let t = Instant::now();
        black_box(sum_par(data, threads));
        t.elapsed()
    });
    let tf1 = best(iters, || {
        let mut pos = Vec::new();
        let t = Instant::now();
        black_box(sum_chunk_fused(data, &mut pos));
        t.elapsed()
    });
    // Worker-local pre-sized position buffers (local NUMA + no realloc).
    let chunks = line_chunks(data, threads);
    assert_eq!(
        sum_par_fused_local(data, &chunks),
        s1,
        "local fused sum diverges"
    );
    let tfn = best(iters, || {
        let t = Instant::now();
        black_box(sum_par_fused_local(data, &chunks));
        t.elapsed()
    });
    let tid = best(iters, || {
        let t = Instant::now();
        black_box(ndjson::parse_ndjson_id_score(data));
        t.elapsed()
    });
    let tidn = best(iters, || {
        let t = Instant::now();
        black_box(ndjson::parse_ndjson_id_score_par(data, threads));
        t.elapsed()
    });
    let g = |dt: Duration| gib / dt.as_secs_f64();
    let ms = |dt: Duration| dt.as_secs_f64() * 1000.0;
    println!("NDJSON → sum(all integers)         best(ms)   GiB/s");
    println!(
        "  falx flat (tape)   1 thread    : {:>8.1}  {:>6.2}",
        ms(t1),
        g(t1)
    );
    println!(
        "  falx flat (tape)  {threads:>2} threads    : {:>8.1}  {:>6.2}",
        ms(tn),
        g(tn)
    );
    println!(
        "  falx fused (no tape) 1 thread  : {:>8.1}  {:>6.2}",
        ms(tf1),
        g(tf1)
    );
    println!(
        "  falx fused (no tape){threads:>2} threads  : {:>8.1}  {:>6.2}",
        ms(tfn),
        g(tfn)
    );
    println!(
        "  falx id+score schema 1 thread  : {:>8.1}  {:>6.2}",
        ms(tid),
        g(tid)
    );
    println!(
        "  falx id+score schema{threads:>2} threads  : {:>8.1}  {:>6.2}",
        ms(tidn),
        g(tidn)
    );
    println!(
        "\nfused vs flat: {:.2}x (1t) / {:.2}x ({threads}t) | fused scaling: {:.1}x",
        g(tf1) / g(t1),
        g(tfn) / g(tn),
        g(tfn) / g(tf1)
    );
    println!(
        "schema id+score vs generic fused: {:.2}x (1t) / {:.2}x ({threads}t)",
        g(tid) / g(tf1),
        g(tidn) / g(tfn)
    );
}
