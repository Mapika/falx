//! falx — a parser *generator*: it compiles declarative format specs into
//! simdjson-style branchless SIMD parsing kernels.
//!
//! A format spec becomes a graph in a small bitstream IR ([`ir`]). The
//! interpreter ([`interp`]) executes that graph as the ground-truth reference;
//! the code generator ([`codegen`]) emits a self-contained SIMD kernel file;
//! the synthesizer ([`synth`]) discovers new bit-parallel kernels; and the
//! experimental typed-AST emitter ([`emit`]) lowers the same graph to Rust or
//! CUDA-C. See the README for the full tour.

#[cfg(feature = "bgzf")]
pub mod bgzf;
pub mod codegen;
pub mod egraph;
/// Experimental typed-AST code emitter (multi-backend), alongside `codegen`.
pub mod emit;
pub mod formats;
pub mod graph_opt;
pub mod interp;
pub mod ir;
pub mod kernels;
pub mod scalar;
#[cfg(feature = "spec")]
pub mod spec;
pub mod synth;
pub mod synth_formats;

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

    /// The IR interpreter running the CSV graph must agree with the scalar
    /// byte-at-a-time reference across randomized inputs (escaped quotes and
    /// newlines inside quoted fields included).
    #[test]
    fn ir_interpreter_matches_scalar() {
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
        }
    }
}
