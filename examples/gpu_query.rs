//! On-device pipeline: a GPU CSV **query engine**.
//!
//! `SELECT count(*), sum(col_k) WHERE col_k > T` runs entirely on the GPU —
//! structural (newline) index, then a per-row navigate→parse→filter→aggregate —
//! and returns 16 bytes. The point: the heavy intermediate (record boundaries,
//! parsed values) never crosses PCIe. We pay one upload of the raw bytes and
//! download only the answer, so end-to-end becomes *upload*-bound while the GPU's
//! parse+aggregate compute hides underneath — and beats the compute-bound CPU
//! even after paying that upload. (Shrink the upload too — GPU-resident
//! decompression — and the wall is gone entirely.)
//!
//! Run: `cargo run --release --features gpu --example gpu_query`

use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const SRC: &str = r#"
#define FULL 0xffffffffu

// ── stage 1: warp-cooperative newline index (kept on device) ──────────────
extern "C" __global__ void nl_count(
    const unsigned char* data, unsigned long long n, unsigned long long warp_bytes,
    unsigned int* counts
) {
    unsigned long long warp = ((unsigned long long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned long long lo = warp*warp_bytes; if (lo >= n) return;
    unsigned long long hi = lo + warp_bytes; if (hi > n) hi = n;
    unsigned int c = 0;
    for (unsigned long long b = lo; b < hi; b += 32) {
        unsigned long long i = b + lane;
        c += __popc(__ballot_sync(FULL, (i < hi) && data[i] == '\n'));
    }
    if (lane == 0) counts[warp] = c;
}
extern "C" __global__ void nl_scatter(
    const unsigned char* data, unsigned long long n, unsigned long long warp_bytes,
    const unsigned int* base, unsigned int* nl
) {
    unsigned long long warp = ((unsigned long long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned long long lo = warp*warp_bytes; if (lo >= n) return;
    unsigned long long hi = lo + warp_bytes; if (hi > n) hi = n;
    unsigned int w = base[warp];
    for (unsigned long long b = lo; b < hi; b += 32) {
        unsigned long long i = b + lane;
        bool s = (i < hi) && data[i] == '\n';
        unsigned mask = __ballot_sync(FULL, s);
        if (s) nl[w + __popc(mask & ((1u << lane) - 1u))] = (unsigned int)i;
        w += __popc(mask);
    }
}

// ── stage 2: one thread per row → navigate to col, parse int, filter, sum ──
extern "C" __global__ void query(
    const unsigned char* data, const unsigned int* nl, unsigned int num_rows,
    unsigned int col, long long threshold,
    unsigned long long* out_count, unsigned long long* out_sum
) {
    unsigned int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= num_rows) return;
    unsigned int i  = (r == 0) ? 0u : nl[r-1] + 1u;   // row start
    unsigned int hi = nl[r];                          // row end (the '\n')
    unsigned int c = 0;
    while (i < hi && c < col) { c += (data[i] == ','); ++i; }
    long long v = 0; bool any = false;
    while (i < hi && data[i] >= '0' && data[i] <= '9') { v = v*10 + (data[i]-'0'); ++i; any = true; }
    if (any && v > threshold) {
        atomicAdd(out_count, 1ULL);
        atomicAdd(out_sum, (unsigned long long)v);
    }
}
"#;

fn make_csv(rows: usize, cols: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * 7);
    let mut x = 0x1234_5678_9abc_def1u64;
    for _ in 0..rows {
        for c in 0..cols {
            if c > 0 {
                out.push(b',');
            }
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let mut v = x % 100_000;
            let s = out.len();
            loop {
                out.push(b'0' + (v % 10) as u8);
                v /= 10;
                if v == 0 {
                    break;
                }
            }
            out[s..].reverse();
        }
        out.push(b'\n');
    }
    out
}

/// CPU reference: count + sum of column `col` where it exceeds `threshold`.
fn cpu_query(data: &[u8], col: usize, threshold: i64) -> (u64, u64) {
    let (mut count, mut sum) = (0u64, 0u64);
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        if let Some(field) = line.split(|&b| b == b',').nth(col) {
            let v: i64 = field
                .iter()
                .filter(|b| b.is_ascii_digit())
                .fold(0i64, |a, &b| a * 10 + (b - b'0') as i64);
            if v > threshold {
                count += 1;
                sum += v as u64;
            }
        }
    }
    (count, sum)
}

fn prefix_sum(counts: &[u32]) -> (Vec<u32>, usize) {
    let mut base = Vec::with_capacity(counts.len());
    let mut acc = 0usize;
    for &c in counts {
        base.push(acc as u32);
        acc += c as usize;
    }
    (base, acc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (col, threshold) = (3usize, 50_000i64);
    let data = make_csv(8_000_000, 8); // ~300 MiB
    let n = data.len();
    println!(
        "data: {:.1} MiB | query: count(*), sum(col{col}) WHERE col{col} > {threshold}\n",
        n as f64 / 1048576.0
    );

    // ── CPU ────────────────────────────────────────────────────────────────
    let t = Instant::now();
    let cpu = cpu_query(&data, col, threshold);
    let cpu_dt = t.elapsed().as_secs_f64();
    println!(
        "CPU scalar   : count={} sum={} | {:.2} GiB/s",
        cpu.0,
        cpu.1,
        n as f64 / cpu_dt / 1073741824.0
    );

    // ── GPU ──────────────────────────────────────────────────────────────--
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let m = ctx.load_module(compile_ptx(SRC)?)?;
    let (nl_count, nl_scatter, query) = (
        m.load_function("nl_count").unwrap(),
        m.load_function("nl_scatter").unwrap(),
        m.load_function("query").unwrap(),
    );

    let warp_bytes = 4096u64;
    let num_warps = (n as u64).div_ceil(warp_bytes);
    let n_u = n as u64;

    let run = || -> Result<((u64, u64), f64, f64), cudarc::driver::DriverError> {
        let t_all = Instant::now();
        let d_data = stream.clone_htod(&data)?; // the one PCIe upload
        let mut d_counts = stream.alloc_zeros::<u32>(num_warps as usize)?;
        stream.synchronize()?;
        let t_gpu = Instant::now();

        let cfg_w = LaunchConfig::for_num_elems((num_warps * 32) as u32);
        let mut b = stream.launch_builder(&nl_count);
        b.arg(&d_data);
        b.arg(&n_u);
        b.arg(&warp_bytes);
        b.arg(&mut d_counts);
        unsafe { b.launch(cfg_w) }?;

        let counts = stream.clone_dtoh(&d_counts)?; // small (per-warp)
        let (base, num_rows) = prefix_sum(&counts);
        let d_base = stream.clone_htod(&base)?;
        let mut d_nl: CudaSlice<u32> = stream.alloc_zeros::<u32>(num_rows)?;
        let mut b = stream.launch_builder(&nl_scatter);
        b.arg(&d_data);
        b.arg(&n_u);
        b.arg(&warp_bytes);
        b.arg(&d_base);
        b.arg(&mut d_nl);
        unsafe { b.launch(cfg_w) }?;

        let mut d_count = stream.alloc_zeros::<u64>(1)?;
        let mut d_sum = stream.alloc_zeros::<u64>(1)?;
        let nr = num_rows as u32;
        let col_u = col as u32;
        let thr = threshold;
        let cfg_r = LaunchConfig::for_num_elems(nr);
        let mut b = stream.launch_builder(&query);
        b.arg(&d_data);
        b.arg(&d_nl);
        b.arg(&nr);
        b.arg(&col_u);
        b.arg(&thr);
        b.arg(&mut d_count);
        b.arg(&mut d_sum);
        unsafe { b.launch(cfg_r) }?;
        stream.synchronize()?;
        let gpu_compute = t_gpu.elapsed().as_secs_f64();

        let count = stream.clone_dtoh(&d_count)?[0]; // 16 bytes total back
        let sum = stream.clone_dtoh(&d_sum)?[0];
        Ok(((count, sum), gpu_compute, t_all.elapsed().as_secs_f64()))
    };

    let _ = run()?; // warm up
    let (mut res, mut best_c, mut best_e) = ((0, 0), f64::MAX, f64::MAX);
    for _ in 0..5 {
        let (r, c, e) = run()?;
        res = r;
        best_c = best_c.min(c);
        best_e = best_e.min(e);
    }
    assert_eq!(res, cpu, "GPU query result diverges from CPU");
    println!(
        "GPU on-device: count={} sum={} | {:.2} GiB/s end-to-end (incl upload), {:.1} GiB/s compute",
        res.0,
        res.1,
        n as f64 / best_e / 1073741824.0,
        n as f64 / best_c / 1073741824.0
    );
    println!(
        "\ncorrectness: GPU == CPU ✓\nend-to-end speedup over CPU: {:.1}x  (downloaded 16 bytes, not the parse)",
        (n as f64 / best_e) / (n as f64 / cpu_dt)
    );
    Ok(())
}
