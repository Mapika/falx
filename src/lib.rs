//! M0 spike for the falx parser generator: a hand-written, simdjson-style
//! SIMD structural indexer for CSV.
//!
//! The kernel answers one question at multiple GB/s: *where are the unquoted
//! `,` and `\n` bytes?* Those positions (the "structural indexes") are the
//! hard, branchy part of CSV parsing; everything downstream (slicing fields,
//! unescaping quotes) is cheap pointer math over them.
//!
//! Pipeline per 64-byte block, the algebra a future generator will emit:
//!
//! 1. **Classify** — SIMD compares produce one 64-bit mask per character
//!    class: quotes, commas, newlines.
//! 2. **Contextualize** — a carry-less multiply (PCLMULQDQ) computes the
//!    prefix-XOR of the quote mask, yielding an "inside a quoted field" mask
//!    with zero branches. RFC 4180 escaped quotes (`""`) toggle the parity
//!    twice and need no special handling. One bit of state (quote parity)
//!    carries across blocks.
//! 3. **Extract** — `(commas | newlines) & !inside_quotes`, then iterate the
//!    set bits with trailing-zero counts into an index buffer.
//!
//! `\r` is never structural: record handling treats `\n` as the terminator
//! and consumers trim a preceding `\r`.

#[cfg(target_arch = "x86_64")]
pub mod avx2;
pub mod codegen;
pub mod formats;
pub mod kernels;
pub mod interp;
pub mod ir;
pub mod scalar;
#[cfg(feature = "spec")]
pub mod spec;
pub mod synth;
pub mod synth_formats;

/// Append the byte offsets of all unquoted `,` and `\n` in `data` to `out`,
/// using the fastest kernel this CPU supports.
pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_structurals(data, out) };
        return;
    }
    scalar::index_structurals(data, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*; avoids a dev-dependency for test data generation.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    fn assert_kernels_agree(data: &[u8]) {
        let mut reference = Vec::new();
        scalar::index_structurals(data, &mut reference);
        let mut fast = Vec::new();
        index_structurals(data, &mut fast);
        assert_eq!(reference, fast, "kernel divergence on {:?}", data);
    }

    #[test]
    fn hand_picked_cases() {
        let cases: &[&[u8]] = &[
            b"",
            b"a,b,c\n",
            b"a,b,c",
            b"\"a,b\",c\n",
            b"\"a\"\"b\",c\n",      // escaped quote inside quoted field
            b"\"multi\nline\",x\n", // newline inside quoted field
            b"\"unclosed,quote\n and more,fields",
            b"\r\n,\r\n",
            b",,,\n,,,\n",
            b"\"\",\"\"\n", // empty quoted fields
        ];
        for case in cases {
            assert_kernels_agree(case);
        }
    }

    #[test]
    fn block_boundary_cases() {
        // Quotes and structurals placed at and around the 64-byte seams,
        // where carry bugs live.
        for pos in [62usize, 63, 64, 65, 126, 127, 128, 129] {
            for ch in [b'"', b',', b'\n'] {
                let mut data = vec![b'x'; 200];
                data[pos] = ch;
                assert_kernels_agree(&data);
                // Same, but starting inside an open quote.
                data[0] = b'"';
                assert_kernels_agree(&data);
            }
        }
    }

    /// The IR interpreter running the CSV graph must agree with both the
    /// scalar reference and the hand-written AVX2 kernel: three independent
    /// implementations of the same semantics.
    #[test]
    fn ir_interpreter_matches_kernels() {
        let graph = formats::csv();
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let alphabet = b"\",\n\rxy";
        for _ in 0..2000 {
            let len = (rng.next() % 300) as usize;
            let data: Vec<u8> = (0..len)
                .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                .collect();
            let mut reference = Vec::new();
            scalar::index_structurals(&data, &mut reference);
            let mut from_ir = Vec::new();
            interp::run(&graph, &data, &mut from_ir);
            assert_eq!(reference, from_ir, "IR divergence on {:?}", data);
            let mut fast = Vec::new();
            index_structurals(&data, &mut fast);
            assert_eq!(reference, fast, "kernel divergence on {:?}", data);
        }
    }

    #[test]
    fn randomized_differential() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        // Alphabet heavily weighted toward structural bytes to stress the
        // quote-parity carry logic.
        let alphabet = b"\",\n\rxy";
        for _ in 0..2000 {
            let len = (rng.next() % 300) as usize;
            let data: Vec<u8> = (0..len)
                .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                .collect();
            assert_kernels_agree(&data);
        }
    }
}
