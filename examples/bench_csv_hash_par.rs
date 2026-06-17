//! Parallel parse for csv_hash (CSV + `#` comments + quotes): serial parse vs
//! parse_par across thread counts. csv_hash was the lone serial-only kernel;
//! this measures the new speculative 3-state parallel path. Records are
//! byte-identical to serial (tests/comments.rs::parallel_parse_matches_serial).
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/data/falx-bench/csv-1g.csv".to_string());
    let raw = std::fs::read(&path).expect("read data");
    // ~512 MiB quote-dense body, line-snapped, with a clustered comment header
    // (the realistic csv_hash shape).
    let cap = (512 * 1024 * 1024).min(raw.len());
    let mut end = cap;
    while end < raw.len() && raw[end - 1] != b'\n' {
        end += 1;
    }
    let mut data = Vec::with_capacity(end + 8192);
    for _ in 0..256 {
        data.extend_from_slice(b"# clustered header comment with \"quotes\", commas\n");
    }
    data.extend_from_slice(&raw[..end]);
    let gib = data.len() as f64 / 1073741824.0;
    let runs = 7;

    // Correctness sanity: record counts agree (cheap, on real data).
    let serial_recs = falx::kernels::csv_hash::parse(&data).terminated_record_count();
    let par_recs = falx::kernels::csv_hash::parse_par(&data, 24).terminated_record_count();
    assert_eq!(serial_recs, par_recs, "parse_par record count must match serial");

    println!("csv_hash parse_par: {:.2} GiB (quote-dense body + clustered # header)\n", gib);

    // Headline: interleave serial vs par@24 (fresh tape) vs par_into@24
    // (recycled tape) so they share run conditions. parse_par_into hands the
    // previous Parsed back, so its master tape is already paged in.
    let (mut serial, mut par24, mut into24) = (Vec::new(), Vec::new(), Vec::new());
    let mut recycle = falx::kernels::csv_hash::parse(&data); // seed the recycle buffer
    for _ in 0..15 {
        let t = Instant::now();
        let p = falx::kernels::csv_hash::parse(&data);
        serial.push(gib / t.elapsed().as_secs_f64());
        std::hint::black_box(p.terminated_record_count());

        let t = Instant::now();
        let p = falx::kernels::csv_hash::parse_par(&data, 24);
        par24.push(gib / t.elapsed().as_secs_f64());
        std::hint::black_box(p.terminated_record_count());

        let t = Instant::now();
        let p = falx::kernels::csv_hash::parse_par_into(&data, 24, recycle);
        into24.push(gib / t.elapsed().as_secs_f64());
        std::hint::black_box(p.terminated_record_count());
        recycle = p; // hand back for the next iteration
    }
    let (s, p24, i24) = (median(serial), median(par24), median(into24));
    println!("INTERLEAVED A/B (median of 15):");
    println!("  serial parse           {:>6.2} GiB/s   1.00x", s);
    println!("  parse_par x24          {:>6.2} GiB/s   {:.2}x", p24, p24 / s);
    println!("  parse_par_into x24     {:>6.2} GiB/s   {:.2}x  (recycled tape)\n", i24, i24 / s);

    // Indicative scaling curve (single pass each, noisier; HT >24 hurts).
    println!("scaling sweep (indicative):");
    for threads in [2usize, 4, 8, 16, 24, 48] {
        let mut g = Vec::new();
        for _ in 0..runs {
            let t = Instant::now();
            let p = falx::kernels::csv_hash::parse_par(&data, threads);
            g.push(gib / t.elapsed().as_secs_f64());
            std::hint::black_box(p.terminated_record_count());
        }
        let m = median(g);
        println!("  parse_par x{:<4} {:>6.2} GiB/s   {:.2}x", threads, m, m / s);
    }
}
