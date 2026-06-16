//! FASTQ framing on falx's generated newline indexer.
//!
//! FASTQ (the dominant sequencing-read format) packs each read into 4 lines:
//! `@header`, sequence, `+`, quality. The quality line may contain `@` and
//! `+`, so a reader cannot split on a sigil — it must find line boundaries
//! and group them by 4. falx generates the SIMD newline scan (and parallelizes
//! it); grouping is the thin layer below. Run:
//!     cargo run --release --example fastq
use falx::kernels::lines;
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
    fn b(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn generate_fastq(target: usize) -> Vec<u8> {
    let mut r = Rng(0x5EED_FA57);
    let mut d = Vec::with_capacity(target + 1024);
    let bases = b"ACGT";
    // Quality alphabet includes '@' and '+' on purpose: that is exactly why
    // FASTQ framing must count lines instead of splitting on a record sigil.
    let quals = b"!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJ";
    let mut id = 0u64;
    while d.len() < target {
        let len = 80 + r.b(80) as usize;
        d.extend_from_slice(b"@read");
        d.extend_from_slice(id.to_string().as_bytes());
        d.extend_from_slice(b" flowcell:1:lane:2\n");
        for _ in 0..len {
            d.push(bases[r.b(4) as usize]);
        }
        d.extend_from_slice(b"\n+\n");
        for _ in 0..len {
            d.push(quals[r.b(quals.len() as u64) as usize]);
        }
        d.push(b'\n');
        id += 1;
    }
    d
}

/// One FASTQ read: four byte-slices into the original buffer, zero-copy.
struct Read<'a> {
    header: &'a [u8],
    seq: &'a [u8],
    plus: &'a [u8],
    qual: &'a [u8],
}

/// Frame reads from sorted newline positions: every four line boundaries are
/// one read. Calls `f` per read; no allocation, no copy.
fn for_each_read(data: &[u8], nls: &[u32], mut f: impl FnMut(Read<'_>)) {
    let mut start = 0usize;
    for rec in nls.chunks_exact(4) {
        let (a, b, c, d) = (
            rec[0] as usize,
            rec[1] as usize,
            rec[2] as usize,
            rec[3] as usize,
        );
        f(Read {
            header: &data[start..a],
            seq: &data[a + 1..b],
            plus: &data[b + 1..c],
            qual: &data[c + 1..d],
        });
        start = d + 1;
    }
}

/// Naive scalar reader: walk every byte, count newlines, group by 4.
fn scalar_total_bases(data: &[u8]) -> usize {
    let (mut total, mut line_start, mut line_in_rec) = (0usize, 0usize, 0u32);
    for (i, &b) in data.iter().enumerate() {
        if b == b'\n' {
            if line_in_rec == 1 {
                total += i - line_start;
            } // sequence line
            line_in_rec = (line_in_rec + 1) % 4;
            line_start = i + 1;
        }
    }
    total
}

fn best(reps: usize, mut f: impl FnMut() -> usize) -> (f64, usize) {
    let w = black_box(f());
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            black_box(f());
            s.elapsed().as_secs_f64()
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], w)
}

fn main() {
    let data = generate_fastq(64 * 1024 * 1024);
    let gib = data.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());

    // Total sequence bases via falx (parallel newline index + group by 4).
    let falx_total = |nls: &[u32]| {
        let mut bases = 0usize;
        for_each_read(&data, nls, |r| bases += r.seq.len());
        bases
    };
    let (t_idx, n) = best(7, || {
        let mut nls = Vec::with_capacity(data.len() / 32);
        lines::index_structurals_par(&data, threads, &mut nls);
        nls.len()
    });
    let (t_full, falx_bases) = best(7, || {
        let mut nls = Vec::with_capacity(data.len() / 32);
        lines::index_structurals_par(&data, threads, &mut nls);
        falx_total(&nls)
    });
    let (t_scalar, scalar_bases) = best(5, || scalar_total_bases(&data));

    // One validation pass exercising all four framed fields: every read must be
    // `@`-headed, `+`-separated, with quality as long as its sequence.
    let mut nls = Vec::with_capacity(data.len() / 32);
    lines::index_structurals_par(&data, threads, &mut nls);
    let mut malformed = 0usize;
    for_each_read(&data, &nls, |r| {
        if r.header.first() != Some(&b'@')
            || r.plus.first() != Some(&b'+')
            || r.qual.len() != r.seq.len()
        {
            malformed += 1;
        }
    });

    println!(
        "FASTQ {:.0} MiB, {} reads, {malformed} malformed\n",
        data.len() as f64 / (1024.0 * 1024.0),
        n / 4
    );
    println!(
        "falx newline index (par x{threads}):   {:.2} GiB/s",
        gib / t_idx
    );
    println!(
        "falx frame + base count (par x{threads}): {:.2} GiB/s",
        gib / t_full
    );
    println!(
        "scalar reader (baseline):          {:.2} GiB/s",
        gib / t_scalar
    );
    println!(
        "\nsequence bytes agree: {} ({} vs {})",
        falx_bases == scalar_bases,
        falx_bases,
        scalar_bases
    );
    println!(
        "speedup vs scalar: {:.1}x",
        (gib / t_full) / (gib / t_scalar)
    );
}
