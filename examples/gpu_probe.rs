//! De-risk the GPU stack end to end: NVRTC-compile a CUDA-C kernel from a
//! string, launch it on the GPU, copy results back, verify on host. The kernel
//! is the embarrassingly-parallel core of structural parsing — classify each
//! byte against a delimiter — so a pass here means the falx → GPU pipeline is
//! real.
//!
//! Run: `cargo run --release --features gpu --example gpu_probe`

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const KERNEL: &str = r#"
extern "C" __global__ void classify(const unsigned char* data, int n, int delim, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        flags[i] = (data[i] == (unsigned char)delim) ? 1 : 0;
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    println!("GPU context on device 0 ✓");
    let stream = ctx.default_stream();

    // Runtime-compile CUDA C → PTX, the way falx will emit kernels.
    let t = std::time::Instant::now();
    let ptx = compile_ptx(KERNEL)?;
    println!(
        "NVRTC compiled CUDA-C → PTX in {:.1} ms ✓",
        t.elapsed().as_secs_f64() * 1e3
    );
    let module = ctx.load_module(ptx)?;
    let f = module.load_function("classify")?;

    // ~256 MiB of comma/letter bytes.
    let n = 256 * 1024 * 1024usize;
    let data: Vec<u8> = (0..n)
        .map(|i| if i % 7 == 0 { b',' } else { b'a' })
        .collect();
    let expected = data.iter().filter(|&&b| b == b',').count();

    let d_data = stream.clone_htod(&data)?;
    let mut d_flags = stream.alloc_zeros::<i32>(n)?;
    let n_i = n as i32;
    let delim = i32::from(b',');

    let cfg = LaunchConfig::for_num_elems(n as u32);
    let warm = {
        let mut b = stream.launch_builder(&f);
        b.arg(&d_data);
        b.arg(&n_i);
        b.arg(&delim);
        b.arg(&mut d_flags);
        unsafe { b.launch(cfg) }?;
        stream.synchronize()?;
        // timed pass
        let t = std::time::Instant::now();
        let mut b = stream.launch_builder(&f);
        b.arg(&d_data);
        b.arg(&n_i);
        b.arg(&delim);
        b.arg(&mut d_flags);
        unsafe { b.launch(cfg) }?;
        stream.synchronize()?;
        t.elapsed()
    };

    let flags = stream.clone_dtoh(&d_flags)?;
    let got = flags.iter().filter(|&&x| x == 1).count();
    assert_eq!(got, expected, "GPU classification disagrees with host");

    let gibs = n as f64 / warm.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    println!(
        "classified {} MiB on GPU: {got} delimiters (host expected {expected}) ✓",
        n / 1048576
    );
    println!("kernel throughput: {gibs:.1} GiB/s (compute only, excludes H2D copy)");
    println!("\nGPU stack works end to end — falx can emit + run kernels on this device.");
    Ok(())
}
