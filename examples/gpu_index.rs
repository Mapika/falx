//! Groundbreaking swing: falx emits a **GPU structural-index kernel** for a
//! format, NVRTC-compiles it, runs it on the GPU, and produces the exact same
//! ordered structural positions as the CPU SIMD kernel — at GPU throughput.
//!
//! The structural byte set is baked into the generated CUDA-C (specialized per
//! format, just like the CPU codegen). Ordered output uses a two-pass
//! count → exclusive-prefix-sum → scatter compaction (no atomics, no CUB), so
//! the result is byte-for-byte identical to `csv::index_structurals` on
//! quote-free data (full quote-region masking via a GPU prefix-XOR scan is the
//! next step).
//!
//! Run: `cargo run --release --features gpu --example gpu_index`

use std::time::Instant;

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

/// Emit a format-specialized CUDA-C structural indexer. `structural` is the
/// format's structural byte set (e.g. CSV = `[b',', b'\n']`); membership is
/// baked into `is_structural`, mirroring how the CPU codegen specializes.
fn emit_cuda(structural: &[u8]) -> String {
    let pred = structural
        .iter()
        .map(|b| format!("c == {b}"))
        .collect::<Vec<_>>()
        .join(" || ");
    // Warp-cooperative: each warp owns a contiguous `warp_bytes` segment and
    // sweeps it 32 bytes at a time, lane L reading byte base+L (coalesced).
    // `__ballot_sync` turns the warp's 32 membership bits into a mask; `__popc`
    // counts them and a lane's masked-popcount is its ordered write rank.
    format!(
        r#"
#define FULL 0xffffffffu
__device__ __forceinline__ bool is_structural(unsigned char c) {{ return {pred}; }}

extern "C" __global__ void count_seg(
    const unsigned char* data, unsigned long long n, unsigned long long warp_bytes,
    unsigned int* counts
) {{
    unsigned long long gtid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long warp = gtid >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned long long lo = warp * warp_bytes;
    if (lo >= n) return;
    unsigned long long hi = lo + warp_bytes; if (hi > n) hi = n;
    unsigned int c = 0;
    for (unsigned long long b = lo; b < hi; b += 32) {{
        unsigned long long i = b + lane;
        bool s = (i < hi) && is_structural(data[i]);
        c += __popc(__ballot_sync(FULL, s));
    }}
    if (lane == 0) counts[warp] = c;
}}

extern "C" __global__ void scatter_seg(
    const unsigned char* data, unsigned long long n, unsigned long long warp_bytes,
    const unsigned int* base, unsigned int* out
) {{
    unsigned long long gtid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long warp = gtid >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned long long lo = warp * warp_bytes;
    if (lo >= n) return;
    unsigned long long hi = lo + warp_bytes; if (hi > n) hi = n;
    unsigned int w = base[warp];
    for (unsigned long long b = lo; b < hi; b += 32) {{
        unsigned long long i = b + lane;
        bool s = (i < hi) && is_structural(data[i]);
        unsigned mask = __ballot_sync(FULL, s);
        unsigned rank = __popc(mask & ((1u << lane) - 1u));
        if (s) out[w + rank] = (unsigned int)i;
        w += __popc(mask);
    }}
}}
"#
    )
}

/// Quote-free CSV-ish data: with no quotes, every `,`/`\n` is structural, so the
/// CPU quote-aware kernel and the GPU all-positions kernel must agree exactly.
fn make_csv(rows: usize, cols: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * 8);
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

fn exclusive_prefix_sum(counts: &[u32]) -> (Vec<u32>, usize) {
    let mut base = Vec::with_capacity(counts.len());
    let mut acc = 0usize;
    for &c in counts {
        base.push(acc as u32);
        acc += c as usize;
    }
    (base, acc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let structural = [b',', b'\n'];
    let data = make_csv(4_000_000, 8); // ~150 MiB
    let n = data.len();
    println!(
        "data: {:.1} MiB, structural set {:?}\n",
        n as f64 / 1048576.0,
        structural.map(|b| b as char)
    );

    // ── CPU reference: the real falx SIMD kernel ───────────────────────────
    let mut cpu = Vec::new();
    let t = Instant::now();
    falx::kernels::csv::index_structurals(&data, &mut cpu);
    let cpu_dt = t.elapsed();
    let cpu_gibs = n as f64 / cpu_dt.as_secs_f64() / 1073741824.0;
    println!(
        "CPU (falx AVX-512/AVX2 SIMD): {} positions, {cpu_gibs:.1} GiB/s",
        cpu.len()
    );

    // ── GPU: emit → NVRTC → run ────────────────────────────────────────────
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let src = emit_cuda(&structural);
    let tc = Instant::now();
    let module = ctx.load_module(compile_ptx(&src)?)?;
    let count_f = module.load_function("count_seg").unwrap();
    let scatter_f = module.load_function("scatter_seg").unwrap();
    println!(
        "GPU: NVRTC-compiled the emitted kernel in {:.0} ms",
        tc.elapsed().as_secs_f64() * 1e3
    );

    let warp_bytes = 4096u64; // bytes per warp
    let num_warps = (n as u64).div_ceil(warp_bytes);
    let threads_total = num_warps * 32;
    let n_u = n as u64;

    let run = || -> Result<(Vec<u32>, f64, f64), cudarc::driver::DriverError> {
        let t_all = Instant::now();
        let d_data = stream.clone_htod(&data)?;
        let mut d_counts = stream.alloc_zeros::<u32>(num_warps as usize)?;
        stream.synchronize()?;
        let t_k = Instant::now();
        let cfg = LaunchConfig::for_num_elems(threads_total as u32);
        {
            let mut b = stream.launch_builder(&count_f);
            b.arg(&d_data);
            b.arg(&n_u);
            b.arg(&warp_bytes);
            b.arg(&mut d_counts);
            unsafe { b.launch(cfg) }?;
        }
        let counts = stream.clone_dtoh(&d_counts)?;
        let (base, total) = exclusive_prefix_sum(&counts);
        let d_base = stream.clone_htod(&base)?;
        let mut d_out = stream.alloc_zeros::<u32>(total)?;
        {
            let mut b = stream.launch_builder(&scatter_f);
            b.arg(&d_data);
            b.arg(&n_u);
            b.arg(&warp_bytes);
            b.arg(&d_base);
            b.arg(&mut d_out);
            unsafe { b.launch(cfg) }?;
        }
        stream.synchronize()?;
        let compute = t_k.elapsed().as_secs_f64();
        let out = stream.clone_dtoh(&d_out)?;
        Ok((out, compute, t_all.elapsed().as_secs_f64()))
    };

    let _ = run()?; // warm up
    let mut best_compute = f64::MAX;
    let mut best_e2e = f64::MAX;
    let mut gpu_out = Vec::new();
    for _ in 0..5 {
        let (out, compute, e2e) = run()?;
        best_compute = best_compute.min(compute);
        best_e2e = best_e2e.min(e2e);
        gpu_out = out;
    }

    // ── Correctness: byte-for-byte vs the CPU SIMD kernel ──────────────────
    assert_eq!(gpu_out.len(), cpu.len(), "GPU position count != CPU");
    assert!(gpu_out == cpu, "GPU positions diverge from CPU SIMD kernel");
    println!("\ncorrectness: GPU structural index == CPU SIMD index, byte-for-byte ✓");

    let comp_gibs = n as f64 / best_compute / 1073741824.0;
    let e2e_gibs = n as f64 / best_e2e / 1073741824.0;
    println!(
        "\nGPU kernels (compute only)      : {comp_gibs:7.1} GiB/s   ({:.2}x CPU)",
        comp_gibs / cpu_gibs
    );
    println!("GPU end-to-end (incl H2D+D2H)   : {e2e_gibs:7.1} GiB/s   (PCIe-bound)");
    Ok(())
}
