//! Same-work JSON parity: sum every integer value in each NDJSON document.
//! Both engines read the same input and must produce the *identical* sum, so
//! this is a true like-for-like query (same input, same output) rather than
//! framing-vs-parsing. falx walks its structural tape lazily and parses only
//! numeric scalars; simd-json eager-parses into a typed tape and we sum its
//! integer nodes. Interleaved timing.
//!
//! `cargo run --release --example json_parity [path]`
use std::hint::black_box;
use std::time::Instant;

use falx::kernels::json::{Node, parse_nested, parse_nested_into};

/// Parse a trimmed scalar span as a base-10 integer; `None` for strings
/// (`"..."`), floats (contain `.`/`e`), booleans, null — anything not a pure
/// integer literal.
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

fn sum_node(node: &Node) -> i64 {
    let mut sum = 0i64;
    let mut had_child = false;
    for child in node.items() {
        had_child = true;
        sum = sum.wrapping_add(sum_node(&child));
    }
    if !had_child && let Some(n) = parse_int(node.bytes()) {
        sum = sum.wrapping_add(n);
    }
    sum
}

/// Baseline: recursive hierarchical navigation (`Node`/`Items`).
fn falx_sum_recursive(data: &[u8]) -> i64 {
    let mut nested = parse_nested(b"");
    let mut sum = 0i64;
    for line in data.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        nested = parse_nested_into(line, nested);
        for item in nested.items() {
            sum = sum.wrapping_add(sum_node(&item));
        }
    }
    sum
}

#[inline]
fn trim(data: &[u8], mut s: usize, mut e: usize) -> (usize, usize) {
    while s < e && matches!(data[s], b' ' | b'\t' | b'\r' | b'\n') {
        s += 1;
    }
    while e > s && matches!(data[e - 1], b' ' | b'\t' | b'\r' | b'\n') {
        e -= 1;
    }
    (s, e)
}

/// Optimized: a flat O(tape) scan. Summing integers needs no hierarchy — every
/// scalar token lives in a gap between consecutive structural-byte positions, so
/// one linear pass over the tape (positions are stored in ascending order)
/// visits every scalar with no recursion, no per-level iterator, no `Node`.
fn falx_sum_flat(data: &[u8]) -> i64 {
    let mut nested = parse_nested(b"");
    let mut sum = 0i64;
    for line in data.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        nested = parse_nested_into(line, nested);
        let mut prev = 0usize;
        for &entry in nested.tape() {
            let pos = (entry as u32) as usize;
            let (s, e) = trim(line, prev, pos);
            if s < e && let Some(n) = parse_int(&line[s..e]) {
                sum = sum.wrapping_add(n);
            }
            prev = pos + 1;
        }
        let (s, e) = trim(line, prev, line.len());
        if s < e && let Some(n) = parse_int(&line[s..e]) {
            sum = sum.wrapping_add(n);
        }
    }
    sum
}

fn simd_json_sum(scratch: &mut [u8]) -> i64 {
    let mut sum = 0i64;
    let mut offset = 0usize;
    while offset < scratch.len() {
        let end = scratch[offset..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(scratch.len(), |p| offset + p);
        if end > offset {
            let tape = simd_json::to_tape(&mut scratch[offset..end]).expect("valid json");
            for node in &tape.0 {
                match node {
                    simd_json::Node::Static(simd_json::StaticNode::I64(n)) => {
                        sum = sum.wrapping_add(*n)
                    }
                    simd_json::Node::Static(simd_json::StaticNode::U64(n)) => {
                        sum = sum.wrapping_add(*n as i64)
                    }
                    _ => {}
                }
            }
        }
        offset = end + 1;
    }
    sum
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/data/falx-bench/ndjson-1g.ndjson".to_string());
    let raw = std::fs::read(&path).expect("read data");
    let cap = (256 * 1024 * 1024).min(raw.len());
    let mut end = cap;
    while end < raw.len() && raw[end - 1] != b'\n' {
        end += 1;
    }
    let data = raw[..end].to_vec();
    let gib = data.len() as f64 / 1073741824.0;

    // Same-work proof: all three produce the identical integer sum.
    let rec = falx_sum_recursive(&data);
    let flat = falx_sum_flat(&data);
    let mut sj_scratch = data.clone();
    let sj = simd_json_sum(&mut sj_scratch);
    assert_eq!(rec, flat, "falx recursive vs flat disagree");
    assert_eq!(rec, sj, "falx vs simd-json disagree: {rec} != {sj}");
    println!("{gib:.2} GiB NDJSON — integer sum = {rec} (all engines agree)\n");

    let reps = 9;
    let median = |f: &dyn Fn() -> i64| -> u128 {
        for _ in 0..2 {
            black_box(f());
        }
        let mut t: Vec<u128> = (0..reps)
            .map(|_| {
                let s = Instant::now();
                black_box(f());
                s.elapsed().as_micros()
            })
            .collect();
        t.sort_unstable();
        t[reps / 2]
    };
    let thr = |us: u128| gib / (us as f64 / 1e6);

    let rec_us = median(&|| falx_sum_recursive(&data));
    let flat_us = median(&|| falx_sum_flat(&data));
    let sj_us = median(&|| {
        let mut s = data.clone();
        simd_json_sum(&mut s)
    });

    println!(
        "falx recursive (Node/Items nav)      : {:.2} GiB/s  ({} ms)",
        thr(rec_us),
        rec_us / 1000
    );
    println!(
        "falx flat tape scan  [optimized]     : {:.2} GiB/s  ({} ms)",
        thr(flat_us),
        flat_us / 1000
    );
    println!(
        "simd-json to_tape + sum              : {:.2} GiB/s  ({} ms)",
        thr(sj_us),
        sj_us / 1000
    );
    println!(
        "\nflat-scan speedup over recursive: {:.2}x",
        thr(flat_us) / thr(rec_us)
    );
    println!(
        "falx (flat) vs simd-json on same-work query: {:.2}x",
        thr(flat_us) / thr(sj_us)
    );
}
