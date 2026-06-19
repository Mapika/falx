//! GPU bgzf decompression via **nvCOMP's batched DEFLATE** — the fast,
//! parallel-within-block decompressor that breaks the one-thread-per-block wall
//! of the hand-rolled puff port. Each bgzf block's raw-deflate payload is one
//! batch chunk; nvCOMP inflates them all on the GPU at once. Validated
//! per-block against the CPU decompressor, then timed.
//!
//! libnvcomp is dlopen'd at runtime (path via $NVCOMP_LIB, default = the
//! `nvidia-nvcomp-cu12` pip layout). Run:
//!   `NVCOMP_LIB=/path/to/libnvcomp.so.5 cargo run --release --features "gpu bgzf" --example gpu_nvcomp`

use std::ffi::c_void;
use std::time::Instant;

use cudarc::driver::{CudaContext, DevicePtr};
use libloading::Library;

#[repr(C)]
#[derive(Clone, Copy)]
struct Opts {
    backend: i32,
    sort_before_hw: i32,
    reserved: [u8; 56],
}
impl Opts {
    fn default_backend() -> Self {
        Opts {
            backend: 0,
            sort_before_hw: 0,
            reserved: [0; 56],
        } // BACKEND_DEFAULT
    }
}

#[repr(C)]
#[derive(Default, Debug)]
struct AlignReq {
    input: usize,
    output: usize,
    temp: usize,
}

type CuStream = *mut c_void;
type FnAligns = unsafe extern "C" fn(Opts, *mut AlignReq) -> i32;
type FnTemp = unsafe extern "C" fn(usize, usize, Opts, *mut usize, usize) -> i32;
type FnDecomp = unsafe extern "C" fn(
    *const *const c_void, // compressed chunk ptrs (device)
    *const usize,         // compressed chunk bytes (device)
    *const usize,         // uncompressed buffer bytes (device)
    *mut usize,           // actual uncompressed bytes out (device)
    usize,                // num chunks
    *mut c_void,          // temp (device)
    usize,                // temp bytes
    *const *mut c_void,   // uncompressed chunk ptrs (device)
    Opts,
    *mut i32, // statuses (device)
    CuStream,
) -> i32;

fn round_up(x: usize, a: usize) -> usize {
    if a <= 1 { x } else { x.div_ceil(a) * a }
}

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

fn bgzip(data: &[u8]) -> Vec<u8> {
    use std::io::Write as _;
    let mut w = noodles_bgzf::io::Writer::new(Vec::new());
    w.write_all(data).expect("bgzf write");
    w.finish().expect("bgzf finish")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib_path = std::env::var("NVCOMP_LIB").unwrap_or_else(|_| {
        // default: the nvidia-nvcomp-cu12 pip layout under the venv
        let home = std::env::var("HOME").unwrap_or_default();
        let _ = home;
        "/tmp/falxpy/lib/python3.10/site-packages/nvidia/libnvcomp/lib64/libnvcomp.so.5".into()
    });
    let lib = unsafe { Library::new(&lib_path) }
        .map_err(|e| format!("dlopen {lib_path}: {e} (pip install nvidia-nvcomp-cu12)"))?;
    let f_aligns: libloading::Symbol<FnAligns> =
        unsafe { lib.get(b"nvcompBatchedDeflateDecompressGetRequiredAlignments")? };
    let f_temp: libloading::Symbol<FnTemp> =
        unsafe { lib.get(b"nvcompBatchedDeflateDecompressGetTempSizeAsync")? };
    let f_decomp: libloading::Symbol<FnDecomp> =
        unsafe { lib.get(b"nvcompBatchedDeflateDecompressAsync")? };
    println!("nvcomp loaded from {lib_path}");

    let raw = make_csv(8_000_000, 8);
    let comp = bgzip(&raw);
    let reference = falx::bgzf::decompress(&comp)?; // CPU truth
    let blocks = falx::bgzf::scan(&comp)?;
    let opts = Opts::default_backend();

    let mut align = AlignReq::default();
    let st = unsafe { (f_aligns)(opts, &mut align) };
    assert_eq!(st, 0, "GetRequiredAlignments status {st}");
    println!(
        "{} bgzf blocks, {:.1} MiB raw / {:.1} MiB bgzf | nvcomp align: in={} out={} temp={}",
        blocks.len(),
        raw.len() as f64 / 1048576.0,
        comp.len() as f64 / 1048576.0,
        align.input,
        align.output,
        align.temp
    );

    // Lay compressed payloads and output buffers at aligned offsets.
    let (mut comp_buf, mut coff, mut clen) = (Vec::<u8>::new(), Vec::new(), Vec::new());
    let (mut ooff, mut olen) = (Vec::new(), Vec::new());
    let mut out_total = 0usize;
    let mut max_chunk = 0usize;
    for b in &blocks {
        let p = &comp[b.payload.clone()];
        comp_buf.resize(round_up(comp_buf.len(), align.input.max(1)), 0);
        coff.push(comp_buf.len());
        clen.push(p.len());
        comp_buf.extend_from_slice(p);
        out_total = round_up(out_total, align.output.max(1));
        ooff.push(out_total);
        olen.push(b.isize);
        out_total += b.isize;
        max_chunk = max_chunk.max(b.isize);
    }
    let nchunks = blocks.len();

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // temp size (host call)
    let mut temp_bytes = 0usize;
    let st = unsafe { (f_temp)(nchunks, max_chunk, opts, &mut temp_bytes, out_total) };
    assert_eq!(st, 0, "GetTempSize status {st}");

    let run = || -> Result<(Vec<u8>, f64, f64), Box<dyn std::error::Error>> {
        let t_all = Instant::now();
        let d_comp = stream.clone_htod(&comp_buf)?; // upload compressed only
        let d_out = stream.alloc_zeros::<u8>(out_total)?;
        let d_temp = stream.alloc_zeros::<u8>(temp_bytes.max(1))?;
        let d_clen = stream.clone_htod(&clen)?;
        let d_olen = stream.clone_htod(&olen)?;
        let d_actual = stream.alloc_zeros::<usize>(nchunks)?;
        let d_status = stream.alloc_zeros::<i32>(nchunks)?;
        // device base addresses → per-chunk pointer arrays
        let (comp_base, _g1) = d_comp.device_ptr(&stream);
        let (out_base, _g2) = d_out.device_ptr(&stream);
        let cptrs: Vec<u64> = coff.iter().map(|&o| comp_base + o as u64).collect();
        let optrs: Vec<u64> = ooff.iter().map(|&o| out_base + o as u64).collect();
        let d_cptrs = stream.clone_htod(&cptrs)?;
        let d_optrs = stream.clone_htod(&optrs)?;
        stream.synchronize()?;
        let t_gpu = Instant::now();

        let (cp, _g3) = d_cptrs.device_ptr(&stream);
        let (op, _g4) = d_optrs.device_ptr(&stream);
        let (cl, _g5) = d_clen.device_ptr(&stream);
        let (ol, _g6) = d_olen.device_ptr(&stream);
        let (ac, _g7) = d_actual.device_ptr(&stream);
        let (tp, _g8) = d_temp.device_ptr(&stream);
        let (sp, _g9) = d_status.device_ptr(&stream);
        let st = unsafe {
            (f_decomp)(
                cp as *const *const c_void,
                cl as *const usize,
                ol as *const usize,
                ac as *mut usize,
                nchunks,
                tp as *mut c_void,
                temp_bytes,
                op as *const *mut c_void,
                opts,
                sp as *mut i32,
                stream.cu_stream() as CuStream,
            )
        };
        stream.synchronize()?;
        if st != 0 {
            return Err(format!("DecompressAsync status {st}").into());
        }
        let gpu = t_gpu.elapsed().as_secs_f64();
        let statuses = stream.clone_dtoh(&d_status)?;
        if let Some(i) = statuses.iter().position(|&s| s != 0) {
            return Err(format!("chunk {i} nvcomp status {}", statuses[i]).into());
        }
        let out = stream.clone_dtoh(&d_out)?;
        Ok((out, gpu, t_all.elapsed().as_secs_f64()))
    };

    let (out, _, _) = run()?;
    // validate per block against the CPU reference (output is aligned, maybe padded)
    let mut ref_off = 0usize;
    for (i, b) in blocks.iter().enumerate() {
        let g = &out[ooff[i]..ooff[i] + olen[i]];
        let r = &reference[ref_off..ref_off + b.isize];
        assert!(g == r, "block {i} nvcomp output != CPU");
        ref_off += b.isize;
    }
    println!("\ncorrectness: all {nchunks} blocks nvcomp-decompressed == CPU, byte-for-byte ✓");

    let (mut bc, mut be) = (f64::MAX, f64::MAX);
    for _ in 0..5 {
        let (_, c, e) = run()?;
        bc = bc.min(c);
        be = be.min(e);
    }
    let u = raw.len() as f64;
    println!(
        "\nnvcomp GPU decompress:\n  compute only : {:.1} GiB/s\n  end-to-end   : {:.1} GiB/s (upload compressed + decompress + download)",
        u / bc / 1073741824.0,
        u / be / 1073741824.0
    );
    println!("\n(vs the puff one-thread-per-block port: ~2.2 GiB/s compute)");
    Ok(())
}
