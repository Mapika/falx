//! AVX2 + PCLMULQDQ structural indexer.
//!
//! Processes 64 bytes per iteration as two 32-byte AVX2 lanes, reduced to
//! 64-bit masks. The only state carried between blocks is quote parity,
//! stored pre-broadcast as 0 or all-ones so it can be applied with one XOR.

use std::arch::x86_64::*;

/// Append the byte offsets of all unquoted `,` and `\n` in `data` to `out`.
///
/// Callers must ensure the CPU supports AVX2 and PCLMULQDQ (the safe
/// dispatcher in the crate root checks this).
///
/// # Safety
///
/// Callers must ensure AVX2 and PCLMULQDQ are available; the function reads
/// `data` in 64-byte blocks via intrinsics.
#[target_feature(enable = "avx2", enable = "pclmulqdq")]
pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {
    let mut carry_inside: u64 = 0;
    let mut offset = 0usize;

    while offset + 64 <= data.len() {
        // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
        let structural =
            unsafe { classify_block(data.as_ptr().add(offset), &mut carry_inside) };
        push_indexes(structural, offset as u32, out);
        offset += 64;
    }

    let rem = data.len() - offset;
    if rem > 0 {
        // Zero padding is safe: NUL is in no character class, so the pad
        // produces no structural bits and cannot flip quote parity.
        let mut buf = [0u8; 64];
        buf[..rem].copy_from_slice(&data[offset..]);
        // SAFETY: buf is a readable 64-byte buffer.
        let structural = unsafe { classify_block(buf.as_ptr(), &mut carry_inside) };
        push_indexes(structural, offset as u32, out);
    }
}

/// Classify one 64-byte block and return its structural mask, updating the
/// cross-block quote-parity carry.
#[target_feature(enable = "avx2", enable = "pclmulqdq")]
unsafe fn classify_block(ptr: *const u8, carry_inside: &mut u64) -> u64 {
    // SAFETY: caller guarantees 64 readable bytes at `ptr`.
    let (lo, hi) = unsafe {
        (
            _mm256_loadu_si256(ptr as *const __m256i),
            _mm256_loadu_si256(ptr.add(32) as *const __m256i),
        )
    };

    let quotes = eq_mask(lo, hi, b'"');
    let commas = eq_mask(lo, hi, b',');
    let newlines = eq_mask(lo, hi, b'\n');

    // prefix_xor turns "where are the quotes" into "which bytes are inside a
    // quoted region"; XOR with the carry accounts for a region left open by
    // a previous block.
    let inside = prefix_xor(quotes) ^ *carry_inside;
    // Arithmetic right shift broadcasts the last byte's parity to all bits.
    *carry_inside = ((inside as i64) >> 63) as u64;

    (commas | newlines) & !inside
}

/// Per-byte equality mask over a 64-byte block held in two AVX2 registers.
#[target_feature(enable = "avx2")]
fn eq_mask(lo: __m256i, hi: __m256i, byte: u8) -> u64 {
    let needle = _mm256_set1_epi8(byte as i8);
    let lo_bits = _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, needle)) as u32 as u64;
    let hi_bits = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, needle)) as u32 as u64;
    lo_bits | (hi_bits << 32)
}

/// Prefix XOR of a 64-bit mask: bit i of the result is the XOR of bits 0..=i.
///
/// Carry-less multiplication by all-ones is exactly this computation, in one
/// instruction (the simdjson trick).
#[target_feature(enable = "pclmulqdq")]
fn prefix_xor(mask: u64) -> u64 {
    let ones = _mm_set1_epi8(-1);
    let value = _mm_set_epi64x(0, mask as i64);
    let product = _mm_clmulepi64_si128::<0>(value, ones);
    _mm_cvtsi128_si64(product) as u64
}

/// Decode the set bits of `mask` into absolute byte offsets.
///
/// Writes indexes in unconditional chunks of 8 (`tzcnt` of an empty mask is
/// 64, producing harmless garbage past the real entries), then sets the
/// length from the popcount. Trades a handful of wasted stores for the
/// removal of an unpredictable branch per set bit — the simdjson
/// `flatten_bits` technique.
#[inline]
fn push_indexes(mut mask: u64, base: u32, out: &mut Vec<u32>) {
    let count = mask.count_ones() as usize;
    if count == 0 {
        return;
    }
    let len = out.len();
    // Chunked writes overshoot by at most 7 entries; keep them in capacity.
    out.reserve(count + 8);
    // SAFETY: capacity covers len + count + 8; every write below lands
    // within that region, and set_len only exposes the `count` real entries.
    unsafe {
        let mut ptr = out.as_mut_ptr().add(len);
        let mut remaining = count as isize;
        while remaining > 0 {
            for j in 0..8 {
                *ptr.add(j) = base + mask.trailing_zeros();
                mask &= mask.wrapping_sub(1);
            }
            ptr = ptr.add(8);
            remaining -= 8;
        }
        out.set_len(len + count);
    }
}
