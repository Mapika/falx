//! The full GPU-resident genomics pipeline: `.vcf.gz` → answer, never leaving
//! the device. Upload the *compressed* bytes once, then on the GPU — (1) inflate
//! every bgzf block in parallel (one thread per independent block, a CUDA-C port
//! of Mark Adler's `puff.c`), (2) structural (newline) index, (3) per-row
//! navigate → parse → filter → aggregate query — and download 16 bytes. The
//! decompressed text, record boundaries, and parsed values never cross PCIe, so
//! the upload is just the compressed size and the CPU's decompression wall is gone.
//!
//! Run: `cargo run --release --features "gpu bgzf" --example gpu_pipeline`

use std::io::Write as _;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

/// (count, sum), per-block inflate status, GPU-compute seconds, end-to-end seconds.
type RunOut = ((u64, u64), Vec<i32>, f64, f64);

const SRC: &str = r#"
// ───────────────────────── DEFLATE inflate (puff.c port) ─────────────────────
__device__ static const short LENS[29] =
 {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
__device__ static const short LEXT[29] =
 {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
__device__ static const short DISTS[30] =
 {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
__device__ static const short DEXT[30] =
 {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
__device__ static const short ORDER[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};

struct st {
    const unsigned char* in; unsigned inlen; unsigned incnt;
    unsigned char* out; unsigned outlen; unsigned outcnt;
    int bitbuf, bitcnt;
};

__device__ int gbits(st* s, int need) {
    int val = s->bitbuf;
    while (s->bitcnt < need) {
        if (s->incnt >= s->inlen) return -1;
        val |= (int)s->in[s->incnt++] << s->bitcnt;
        s->bitcnt += 8;
    }
    s->bitbuf = (int)((unsigned)val >> need);
    s->bitcnt -= need;
    return val & ((1 << need) - 1);
}

__device__ int gdecode(st* s, const short* count, const short* symbol) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= 15; len++) {
        code |= gbits(s, 1);
        int cnt = count[len];
        if (code - first < cnt) return symbol[index + (code - first)];
        index += cnt; first += cnt; first <<= 1; code <<= 1;
    }
    return -10;
}

__device__ int gconstruct(short* count, short* symbol, const short* length, int n) {
    short offs[16];
    for (int len = 0; len <= 15; len++) count[len] = 0;
    for (int sy = 0; sy < n; sy++) count[length[sy]]++;
    if (count[0] == n) return 0;
    int left = 1;
    for (int len = 1; len <= 15; len++) { left <<= 1; left -= count[len]; if (left < 0) return left; }
    offs[1] = 0;
    for (int len = 1; len < 15; len++) offs[len + 1] = offs[len] + count[len];
    for (int sy = 0; sy < n; sy++) if (length[sy] != 0) symbol[offs[length[sy]]++] = (short)sy;
    return left;
}

__device__ int gcodes(st* s, const short* lc, const short* ls, const short* dc, const short* ds) {
    int symbol;
    do {
        symbol = gdecode(s, lc, ls);
        if (symbol < 0) return symbol;
        if (symbol < 256) {
            if (s->outcnt < s->outlen) s->out[s->outcnt] = (unsigned char)symbol;
            s->outcnt++;
        } else if (symbol > 256) {
            symbol -= 257; if (symbol >= 29) return -10;
            int len = LENS[symbol] + gbits(s, LEXT[symbol]);
            symbol = gdecode(s, dc, ds); if (symbol < 0) return symbol;
            unsigned dist = DISTS[symbol] + gbits(s, DEXT[symbol]);
            if (dist > s->outcnt) return -11;
            while (len--) {
                if (s->outcnt < s->outlen) s->out[s->outcnt] = s->out[s->outcnt - dist];
                s->outcnt++;
            }
        }
    } while (symbol != 256);
    return 0;
}

__device__ int gfixed(st* s, short* lc, short* ls, short* dc, short* ds) {
    short lengths[288]; int sy;
    for (sy = 0; sy < 144; sy++) lengths[sy] = 8;
    for (; sy < 256; sy++) lengths[sy] = 9;
    for (; sy < 280; sy++) lengths[sy] = 7;
    for (; sy < 288; sy++) lengths[sy] = 8;
    gconstruct(lc, ls, lengths, 288);
    for (sy = 0; sy < 30; sy++) lengths[sy] = 5;
    gconstruct(dc, ds, lengths, 30);
    return gcodes(s, lc, ls, dc, ds);
}

__device__ int gdynamic(st* s, short* lc, short* ls, short* dc, short* ds) {
    short lengths[320];
    int nlen = gbits(s, 5) + 257, ndist = gbits(s, 5) + 1, ncode = gbits(s, 4) + 4;
    if (nlen > 286 || ndist > 30) return -3;
    int index;
    for (index = 0; index < ncode; index++) lengths[ORDER[index]] = (short)gbits(s, 3);
    for (; index < 19; index++) lengths[ORDER[index]] = 0;
    if (gconstruct(lc, ls, lengths, 19) != 0) return -4;
    index = 0;
    while (index < nlen + ndist) {
        int symbol = gdecode(s, lc, ls);
        if (symbol < 0) return symbol;
        if (symbol < 16) { lengths[index++] = (short)symbol; }
        else {
            int len = 0;
            if (symbol == 16) { if (index == 0) return -5; len = lengths[index - 1]; symbol = 3 + gbits(s, 2); }
            else if (symbol == 17) { symbol = 3 + gbits(s, 3); }
            else { symbol = 11 + gbits(s, 7); }
            if (index + symbol > nlen + ndist) return -6;
            while (symbol--) lengths[index++] = (short)len;
        }
    }
    if (lengths[256] == 0) return -9;
    int err = gconstruct(lc, ls, lengths, nlen);
    if (err < 0 || (err > 0 && nlen != lc[0] + lc[1])) return -7;
    err = gconstruct(dc, ds, lengths + nlen, ndist);
    if (err < 0 || (err > 0 && ndist != dc[0] + dc[1])) return -8;
    return gcodes(s, lc, ls, dc, ds);
}

__device__ int gpu_inflate(const unsigned char* in, unsigned inlen, unsigned char* out, unsigned outlen) {
    st s; s.in = in; s.inlen = inlen; s.incnt = 0; s.out = out; s.outlen = outlen;
    s.outcnt = 0; s.bitbuf = 0; s.bitcnt = 0;
    short lc[16], ls[288], dc[16], ds[30];
    int last, type, err = 0;
    do {
        last = gbits(&s, 1);
        type = gbits(&s, 2);
        if (type == 0) {
            s.bitbuf = 0; s.bitcnt = 0;
            if (s.incnt + 4 > s.inlen) return -2;
            unsigned len = (unsigned)s.in[s.incnt] | ((unsigned)s.in[s.incnt + 1] << 8);
            s.incnt += 4;
            if (s.incnt + len > s.inlen) return -2;
            for (unsigned k = 0; k < len; k++) {
                if (s.outcnt < s.outlen) s.out[s.outcnt] = s.in[s.incnt];
                s.outcnt++; s.incnt++;
            }
        } else if (type == 1) { err = gfixed(&s, lc, ls, dc, ds); }
        else if (type == 2) { err = gdynamic(&s, lc, ls, dc, ds); }
        else return -1;
        if (err != 0) return err;
    } while (!last);
    return (int)s.outcnt;
}

extern "C" __global__ void inflate_blocks(
    const unsigned char* comp, const unsigned int* poff, const unsigned int* plen,
    const unsigned int* ooff, const unsigned int* olen, unsigned int nblocks,
    unsigned char* out, int* status
) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    status[b] = gpu_inflate(comp + poff[b], plen[b], out + ooff[b], olen[b]);
}

// ───────────────────────── newline index + query ────────────────────────────
#define FULL 0xffffffffu
extern "C" __global__ void nl_count(const unsigned char* d, unsigned long long n, unsigned long long wb, unsigned int* counts) {
    unsigned long long warp = ((unsigned long long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned long long lo = warp*wb; if (lo >= n) return;
    unsigned long long hi = lo + wb; if (hi > n) hi = n;
    unsigned int c = 0;
    for (unsigned long long b = lo; b < hi; b += 32)
        c += __popc(__ballot_sync(FULL, (b + lane < hi) && d[b + lane] == '\n'));
    if (lane == 0) counts[warp] = c;
}
extern "C" __global__ void nl_scatter(const unsigned char* d, unsigned long long n, unsigned long long wb, const unsigned int* base, unsigned int* nl) {
    unsigned long long warp = ((unsigned long long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    unsigned lane = threadIdx.x & 31u;
    unsigned long long lo = warp*wb; if (lo >= n) return;
    unsigned long long hi = lo + wb; if (hi > n) hi = n;
    unsigned int w = base[warp];
    for (unsigned long long b = lo; b < hi; b += 32) {
        bool s = (b + lane < hi) && d[b + lane] == '\n';
        unsigned mask = __ballot_sync(FULL, s);
        if (s) nl[w + __popc(mask & ((1u << lane) - 1u))] = (unsigned int)(b + lane);
        w += __popc(mask);
    }
}
extern "C" __global__ void query(
    const unsigned char* d, const unsigned int* nl, unsigned int num_rows,
    unsigned int col, long long thr, unsigned long long* oc, unsigned long long* os
) {
    unsigned r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= num_rows) return;
    unsigned i = (r == 0) ? 0u : nl[r-1] + 1u, hi = nl[r], c = 0;
    while (i < hi && c < col) { c += (d[i] == ','); ++i; }
    long long v = 0; bool any = false;
    while (i < hi && d[i] >= '0' && d[i] <= '9') { v = v*10 + (d[i]-'0'); ++i; any = true; }
    if (any && v > thr) { atomicAdd(oc, 1ULL); atomicAdd(os, (unsigned long long)v); }
}
"#;

fn make_vcf_like_csv(rows: usize, cols: usize) -> Vec<u8> {
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
    let mut w = noodles_bgzf::io::Writer::new(Vec::new());
    w.write_all(data).expect("bgzf write");
    w.finish().expect("bgzf finish")
}

fn cpu_query(data: &[u8], col: usize, thr: i64) -> (u64, u64) {
    let (mut count, mut sum) = (0u64, 0u64);
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        if let Some(f) = line.split(|&b| b == b',').nth(col) {
            let v = f
                .iter()
                .filter(|b| b.is_ascii_digit())
                .fold(0i64, |a, &b| a * 10 + (b - b'0') as i64);
            if v > thr {
                count += 1;
                sum += v as u64;
            }
        }
    }
    (count, sum)
}

fn prefix_sum(c: &[u32]) -> (Vec<u32>, usize) {
    let mut base = Vec::with_capacity(c.len());
    let mut acc = 0usize;
    for &x in c {
        base.push(acc as u32);
        acc += x as usize;
    }
    (base, acc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (col, thr) = (3usize, 50_000i64);
    let raw = make_vcf_like_csv(8_000_000, 8);
    let comp = bgzip(&raw);
    let (n, cn) = (raw.len(), comp.len());
    println!(
        "raw {:.1} MiB → bgzf {:.1} MiB ({:.1}x) | query: count,sum(col{col}) WHERE col{col}>{thr}\n",
        n as f64 / 1048576.0,
        cn as f64 / 1048576.0,
        n as f64 / cn as f64
    );

    // bgzf block table (host scan — cheap, reads block headers only).
    let blocks = falx::bgzf::scan(&comp)?;
    let mut poff = Vec::new();
    let mut plen = Vec::new();
    let mut ooff = Vec::new();
    let mut olen = Vec::new();
    let mut acc = 0u32;
    for b in &blocks {
        poff.push(b.payload.start as u32);
        plen.push((b.payload.end - b.payload.start) as u32);
        ooff.push(acc);
        olen.push(b.isize as u32);
        acc += b.isize as u32;
    }
    let nblocks = blocks.len() as u32;
    println!(
        "{nblocks} bgzf blocks, {:.1} MiB decompressed",
        acc as f64 / 1048576.0
    );

    // ── CPU full pipeline: bgzf decompress + parse + query ──────────────────
    let t = Instant::now();
    let dec = falx::bgzf::decompress_par(
        &comp,
        std::thread::available_parallelism().map_or(8, |x| x.get()),
    )?;
    let cpu_ref = cpu_query(&dec, col, thr);
    let cpu_dt = t.elapsed().as_secs_f64();
    println!(
        "\nCPU (.vcf.gz→decompress_par→parse→query): count={} sum={} | {:.2} GiB/s",
        cpu_ref.0,
        cpu_ref.1,
        n as f64 / cpu_dt / 1073741824.0
    );

    // ── GPU full pipeline, all on-device ────────────────────────────────────
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let m = ctx.load_module(compile_ptx(SRC)?)?;
    let inflate = m.load_function("inflate_blocks").unwrap();
    let (nl_count, nl_scatter, query) = (
        m.load_function("nl_count").unwrap(),
        m.load_function("nl_scatter").unwrap(),
        m.load_function("query").unwrap(),
    );

    let total = acc as usize;
    let wb = 4096u64;
    let num_warps = (total as u64).div_ceil(wb);
    let total_u = total as u64;

    let run = || -> Result<RunOut, cudarc::driver::DriverError> {
        let t_all = Instant::now();
        let d_comp = stream.clone_htod(&comp)?; // upload: just the compressed bytes
        let d_poff = stream.clone_htod(&poff)?;
        let d_plen = stream.clone_htod(&plen)?;
        let d_ooff = stream.clone_htod(&ooff)?;
        let d_olen = stream.clone_htod(&olen)?;
        let mut d_out: CudaSlice<u8> = stream.alloc_zeros::<u8>(total)?;
        let mut d_status = stream.alloc_zeros::<i32>(nblocks as usize)?;
        stream.synchronize()?;
        let t_gpu = Instant::now();

        // 1. inflate every block in parallel. The inflate kernel is heavy
        // (per-thread Huffman tables), so use a small block to stay within the
        // SM's register/local-memory budget.
        let cfg_b = LaunchConfig {
            grid_dim: (nblocks.div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&inflate);
        b.arg(&d_comp);
        b.arg(&d_poff);
        b.arg(&d_plen);
        b.arg(&d_ooff);
        b.arg(&d_olen);
        b.arg(&nblocks);
        b.arg(&mut d_out);
        b.arg(&mut d_status);
        unsafe { b.launch(cfg_b) }?;

        // 2. newline index (on device)
        let mut d_counts = stream.alloc_zeros::<u32>(num_warps as usize)?;
        let cfg_w = LaunchConfig::for_num_elems((num_warps * 32) as u32);
        let mut b = stream.launch_builder(&nl_count);
        b.arg(&d_out);
        b.arg(&total_u);
        b.arg(&wb);
        b.arg(&mut d_counts);
        unsafe { b.launch(cfg_w) }?;
        let counts = stream.clone_dtoh(&d_counts)?;
        let (base, num_rows) = prefix_sum(&counts);
        let d_base = stream.clone_htod(&base)?;
        let mut d_nl = stream.alloc_zeros::<u32>(num_rows)?;
        let mut b = stream.launch_builder(&nl_scatter);
        b.arg(&d_out);
        b.arg(&total_u);
        b.arg(&wb);
        b.arg(&d_base);
        b.arg(&mut d_nl);
        unsafe { b.launch(cfg_w) }?;

        // 3. query → 16 bytes
        let mut d_c = stream.alloc_zeros::<u64>(1)?;
        let mut d_s = stream.alloc_zeros::<u64>(1)?;
        let (nr, col_u, t2) = (num_rows as u32, col as u32, thr);
        let mut b = stream.launch_builder(&query);
        b.arg(&d_out);
        b.arg(&d_nl);
        b.arg(&nr);
        b.arg(&col_u);
        b.arg(&t2);
        b.arg(&mut d_c);
        b.arg(&mut d_s);
        unsafe { b.launch(LaunchConfig::for_num_elems(nr)) }?;
        stream.synchronize()?;
        let gpu_compute = t_gpu.elapsed().as_secs_f64();

        let status = stream.clone_dtoh(&d_status)?;
        let res = (stream.clone_dtoh(&d_c)?[0], stream.clone_dtoh(&d_s)?[0]);
        Ok((res, status, gpu_compute, t_all.elapsed().as_secs_f64()))
    };

    // validate inflate per block, and the query result
    let (res, status, _, _) = run()?;
    for (i, &st) in status.iter().enumerate() {
        assert!(
            st >= 0 && st as u32 == olen[i],
            "block {i} inflate failed: status {st} (expected {})",
            olen[i]
        );
    }
    assert_eq!(res, cpu_ref, "GPU pipeline result != CPU");
    println!("\ncorrectness: all {nblocks} blocks inflated to exact ISIZE; GPU query == CPU ✓");

    let (mut best_c, mut best_e) = (f64::MAX, f64::MAX);
    for _ in 0..5 {
        let (_, _, c, e) = run()?;
        best_c = best_c.min(c);
        best_e = best_e.min(e);
    }
    // throughput reported over the *uncompressed* bytes processed.
    println!(
        "\nGPU on-device (.vcf.gz→inflate→parse→query):\n  \
         end-to-end (upload compressed + all GPU + 16B down): {:.2} GiB/s\n  \
         GPU compute only (inflate+index+query)             : {:.1} GiB/s",
        n as f64 / best_e / 1073741824.0,
        n as f64 / best_c / 1073741824.0
    );
    println!(
        "\nend-to-end speedup over the CPU pipeline: {:.1}x  (upload was {:.1} MiB compressed, not {:.0} MiB raw)",
        (n as f64 / best_e) / (n as f64 / cpu_dt),
        cn as f64 / 1048576.0,
        n as f64 / 1048576.0
    );
    Ok(())
}
