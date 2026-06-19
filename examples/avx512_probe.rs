//! Prototype probe: does TRUE 512-bit-wide structural indexing beat the kernels'
//! current "AVX-512 lite" path on this box?
//!
//! falx's runtime `avx512` kernel classifies a 64-byte block as TWO 256-bit halves
//! (`_mm256_cmpeq_epi8_mask` ×2, OR'd into a u64). This probe compares that against
//! a single `_mm512_loadu_si512` + `_mm512_cmpeq_epi8_mask` (64 bytes / instruction),
//! on the CSV structural-index hot loop (quote-parity + `\n`/`,` classification).
//! Both share the same clmul prefix-xor and the same index extract, so the delta is
//! purely classification width. Run on an in-RAM buffer (no I/O).
//!
//! Run: cargo run --release --example avx512_probe -- <file.csv> [iters]

#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::{Duration, Instant};

#[cfg(target_arch = "x86_64")]
fn best(iters: usize, mut f: impl FnMut() -> Duration) -> Duration {
    let mut b = Duration::MAX;
    for _ in 0..iters {
        b = b.min(f());
    }
    b
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[inline]
    pub fn push_idx(mut m: u64, base: u32, out: &mut Vec<u32>) {
        while m != 0 {
            out.push(base + m.trailing_zeros());
            m &= m - 1;
        }
    }

    #[target_feature(enable = "pclmulqdq")]
    unsafe fn prefix_xor(mask: u64) -> u64 {
        let ones = _mm_set1_epi8(-1);
        let v = _mm_set_epi64x(0, mask as i64);
        let r = _mm_clmulepi64_si128(v, ones, 0);
        _mm_cvtsi128_si64(r) as u64
    }

    // ---- current kernel path: 64 bytes as two 256-bit halves ----
    #[target_feature(enable = "avx512bw", enable = "avx512vl")]
    unsafe fn eq256(lo: __m256i, hi: __m256i, byte: u8) -> u64 {
        let n = _mm256_set1_epi8(byte as i8);
        let l = _mm256_cmpeq_epi8_mask(lo, n) as u64;
        let h = _mm256_cmpeq_epi8_mask(hi, n) as u64;
        l | (h << 32)
    }

    #[target_feature(
        enable = "avx512f",
        enable = "avx512bw",
        enable = "avx512vl",
        enable = "pclmulqdq"
    )]
    unsafe fn step256(ptr: *const u8, carry: &mut u64) -> u64 {
        // SAFETY: caller guarantees 64 readable bytes.
        let (lo, hi) = unsafe {
            (
                _mm256_loadu_si256(ptr as *const __m256i),
                _mm256_loadu_si256(ptr.add(32) as *const __m256i),
            )
        };
        let q = unsafe { eq256(lo, hi, 34) };
        let parity = unsafe { prefix_xor(q) } ^ *carry;
        *carry = ((parity as i64) >> 63) as u64;
        let nlc = unsafe { eq256(lo, hi, 10) } | unsafe { eq256(lo, hi, 44) };
        nlc & !parity
    }

    // ---- true 512-bit-wide path: 64 bytes / instruction ----
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn eq512(v: __m512i, byte: u8) -> u64 {
        _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(byte as i8))
    }

    #[target_feature(
        enable = "avx512f",
        enable = "avx512bw",
        enable = "avx512vl",
        enable = "pclmulqdq"
    )]
    unsafe fn step512(ptr: *const u8, carry: &mut u64) -> u64 {
        // SAFETY: caller guarantees 64 readable bytes.
        let v = unsafe { _mm512_loadu_si512(ptr.cast()) };
        let q = unsafe { eq512(v, 34) };
        let parity = unsafe { prefix_xor(q) } ^ *carry;
        *carry = ((parity as i64) >> 63) as u64;
        let nlc = unsafe { eq512(v, 10) } | unsafe { eq512(v, 44) };
        nlc & !parity
    }

    macro_rules! define_runners {
        ($classify:ident, $index:ident, $step:ident) => {
            // classify-only: forces step, accumulates popcount, no extract.
            #[target_feature(
                enable = "avx512f",
                enable = "avx512bw",
                enable = "avx512vl",
                enable = "pclmulqdq"
            )]
            pub unsafe fn $classify(data: &[u8]) -> u64 {
                let mut carry = 0u64;
                let mut acc = 0u64;
                let mut off = 0usize;
                while off + 64 <= data.len() {
                    let m = unsafe { $step(data.as_ptr().add(off), &mut carry) };
                    acc = acc.wrapping_add(m.count_ones() as u64);
                    off += 64;
                }
                acc
            }
            // full structural index: extract delimiter positions.
            #[target_feature(
                enable = "avx512f",
                enable = "avx512bw",
                enable = "avx512vl",
                enable = "pclmulqdq"
            )]
            pub unsafe fn $index(data: &[u8], out: &mut Vec<u32>) {
                let mut carry = 0u64;
                let mut off = 0usize;
                while off + 64 <= data.len() {
                    let m = unsafe { $step(data.as_ptr().add(off), &mut carry) };
                    push_idx(m, off as u32, out);
                    off += 64;
                }
            }
        };
    }
    define_runners!(classify256, index256, step256);
    define_runners!(classify512, index512, step512);
}

#[cfg(target_arch = "x86_64")]
fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: avx512_probe <file.csv> [iters]");
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(9);

    if !(std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq"))
    {
        println!("AVX-512 (f/bw/vl) + pclmulqdq not available on this CPU — skipping.");
        return;
    }

    let data = std::fs::read(&path).expect("read");
    let gib = data.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    let g = |dt: Duration| gib / dt.as_secs_f64();
    let ms = |dt: Duration| dt.as_secs_f64() * 1000.0;

    // Correctness: both paths must agree (popcount and full index).
    let c256 = unsafe { x86::classify256(&data) };
    let c512 = unsafe { x86::classify512(&data) };
    let mut i256 = Vec::new();
    let mut i512 = Vec::new();
    unsafe { x86::index256(&data, &mut i256) };
    unsafe { x86::index512(&data, &mut i512) };
    assert_eq!(c256, c512, "classify popcount diverges");
    assert_eq!(i256, i512, "structural index diverges");
    println!(
        "file: {path}  ({gib:.2} GiB)  best of {iters}\nagree: {} structural positions, popcount {c256}\n",
        i256.len()
    );

    let cap = i256.len();
    let t_c256 = best(iters, || {
        let t = Instant::now();
        black_box(unsafe { x86::classify256(&data) });
        t.elapsed()
    });
    let t_c512 = best(iters, || {
        let t = Instant::now();
        black_box(unsafe { x86::classify512(&data) });
        t.elapsed()
    });
    let t_i256 = best(iters, || {
        let mut out = Vec::with_capacity(cap);
        let t = Instant::now();
        unsafe { x86::index256(&data, &mut out) };
        let dt = t.elapsed();
        black_box(&out);
        dt
    });
    let t_i512 = best(iters, || {
        let mut out = Vec::with_capacity(cap);
        let t = Instant::now();
        unsafe { x86::index512(&data, &mut out) };
        let dt = t.elapsed();
        black_box(&out);
        dt
    });

    println!("                                  best(ms)   GiB/s");
    println!(
        "  classify-only  256-halves   : {:>8.1}  {:>6.2}",
        ms(t_c256),
        g(t_c256)
    );
    println!(
        "  classify-only  true-512     : {:>8.1}  {:>6.2}",
        ms(t_c512),
        g(t_c512)
    );
    println!(
        "  full index     256-halves   : {:>8.1}  {:>6.2}",
        ms(t_i256),
        g(t_i256)
    );
    println!(
        "  full index     true-512     : {:>8.1}  {:>6.2}",
        ms(t_i512),
        g(t_i512)
    );
    println!(
        "\nclassify true-512 vs 256-halves: {:.2}x | full-index: {:.2}x",
        g(t_c512) / g(t_c256),
        g(t_i512) / g(t_i256)
    );
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("avx512_probe: x86_64-only.");
}
