//! Framing-driven bgzf decompression vs the hand-written path, and the
//! end-to-end compressed-container pipeline (locate -> inflate -> parse).
use std::hint::black_box;
use std::time::Instant;

fn build(chunks: &[&[u8]]) -> Vec<u8> {
    use std::io::Write;
    let mut out = Vec::new();
    for c in chunks {
        let mut comp =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        comp.write_all(c).unwrap();
        let d = comp.finish().unwrap();
        let bl = 12 + 6 + d.len() + 8;
        let mut h = [0u8; 18];
        h[0] = 0x1f;
        h[1] = 0x8b;
        h[2] = 8;
        h[3] = 0x04;
        h[9] = 0xff;
        h[10] = 6;
        h[12] = b'B';
        h[13] = b'C';
        h[14] = 2;
        let bs = (bl - 1) as u16;
        h[16] = bs as u8;
        h[17] = (bs >> 8) as u8;
        out.extend_from_slice(&h);
        out.extend_from_slice(&d);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(c.len() as u32).to_le_bytes());
    }
    out
}

fn main() {
    let mib: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let threads: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    // ~64 KiB uncompressed CSV blocks, split on record boundaries.
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut raw = 0usize;
    let mut row = 0u64;
    while raw < mib * 1024 * 1024 {
        let mut c = Vec::with_capacity(65536);
        while c.len() < 60000 {
            c.extend_from_slice(
                format!(
                    "cc,city{row},accent,region,999,{:.6},-1.500000\n",
                    row as f64 / 8.0
                )
                .as_bytes(),
            );
            row += 1;
        }
        raw += c.len();
        chunks.push(c);
    }
    let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
    let data = build(&refs);
    let gib = raw as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "{:.2} GiB uncompressed -> {:.2} GiB bgzf ({} blocks), {threads} threads\n",
        gib,
        data.len() as f64 / (1024.0 * 1024.0 * 1024.0),
        chunks.len()
    );
    let framing = falx::framing::bgzf_framing();

    // NOTE ON ORDERING: the two whole-stream decompression rows each allocate
    // and touch a fresh output buffer the size of the uncompressed stream, and
    // whichever runs *second* measures ~5% slower purely from that allocation
    // churn. Swapping their order swaps which one looks slower, so read them as
    // equivalent — the framing-driven path costs nothing measurable over the
    // hand-written one. The fused row is the meaningful comparison: it never
    // materializes a whole-stream buffer at all.
    let bench = |label: &str, f: &dyn Fn() -> u64| {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            black_box(f());
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "  {label:44} {:8.1} ms {:7.2} GiB/s",
            best * 1e3,
            gib / best
        );
    };

    bench("scan_frames only (sequential chain)", &|| {
        falx::framing::scan(&framing, &data).unwrap().len() as u64
    });
    bench("bgzf::decompress_framed_par (framing-driven)", &|| {
        falx::bgzf::decompress_framed_par(&data, &framing, threads)
            .unwrap()
            .len() as u64
    });
    bench("bgzf::decompress_par (hand-written)", &|| {
        falx::bgzf::decompress_par(&data, threads).unwrap().len() as u64
    });
    bench("framed inflate + parse (fused, no full buffer)", &|| {
        let s = falx::bgzf::parse_framed_par(
            &data,
            &framing,
            threads,
            || 0u64,
            |acc, _i, blk| {
                *acc += falx::kernels::csv_geo::parse_csv_geo_stats(blk).records;
            },
        )
        .unwrap();
        s.iter().sum()
    });

    // Correctness: both routes agree.
    let a = falx::bgzf::decompress_par(&data, threads).unwrap();
    let b = falx::bgzf::decompress_framed_par(&data, &framing, threads).unwrap();
    assert_eq!(a, b, "framing-driven output differs from hand-written");
    let rows: u64 = falx::bgzf::parse_framed_par(
        &data,
        &framing,
        threads,
        || 0u64,
        |acc, _i, blk| {
            *acc += falx::kernels::csv_geo::parse_csv_geo_stats(blk).records;
        },
    )
    .unwrap()
    .iter()
    .sum();
    let direct = falx::kernels::csv_geo::parse_csv_geo_stats(&a).records;
    assert_eq!(
        rows, direct,
        "fused parse disagrees with decompress-then-parse"
    );
    println!("\nverified: framing-driven == hand-written (byte-identical); fused rows = {rows}");
}
