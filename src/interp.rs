//! Reference interpreter for bitstream graphs.
//!
//! Evaluates a [`Graph`] over input data 64 bytes at a time, one `u64` value
//! per node per block, threading each stateful node's carry between blocks.
//! This is the semantic ground truth the codegen backend (M2) must match;
//! it favors obviousness over speed.
//!
//! The final partial block is zero-padded. NUL bytes can still match a
//! `Class` whose set contains 0, so graphs over formats where NUL is
//! meaningful would misclassify the pad — acceptable for the current format
//! family, and the codegen backend will share the same convention.

use crate::ir::{Graph, Op};

/// Per-node scratch and carry state for one evaluation pass.
struct Machine<'g> {
    graph: &'g Graph,
    /// Current block's value for each node.
    values: Vec<u64>,
    /// Carried state for each node (unused slots stay 0):
    /// `ShiftLeft1` keeps the operand's top bit, `PrefixXor` keeps the
    /// running parity pre-broadcast to all 64 bits, `Add` keeps its
    /// carry-out bit.
    carries: Vec<u64>,
}

impl<'g> Machine<'g> {
    fn new(graph: &'g Graph) -> Self {
        Self {
            graph,
            values: vec![0; graph.nodes().len()],
            carries: vec![0; graph.nodes().len()],
        }
    }

    /// Evaluate every node over one 64-byte block and return the output mask.
    fn step(&mut self, block: &[u8; 64]) -> u64 {
        for (i, op) in self.graph.nodes().iter().enumerate() {
            self.values[i] = match *op {
                Op::Class(class) => {
                    let mut mask = 0u64;
                    for (bit, &byte) in block.iter().enumerate() {
                        mask |= (class.contains(byte) as u64) << bit;
                    }
                    mask
                }
                Op::Const(pattern) => pattern,
                Op::Not(a) => !self.values[a.0 as usize],
                Op::And(a, b) => self.values[a.0 as usize] & self.values[b.0 as usize],
                Op::Or(a, b) => self.values[a.0 as usize] | self.values[b.0 as usize],
                Op::Xor(a, b) => self.values[a.0 as usize] ^ self.values[b.0 as usize],
                Op::ShiftLeft1(a) => {
                    let v = self.values[a.0 as usize];
                    let out = (v << 1) | self.carries[i];
                    self.carries[i] = v >> 63;
                    out
                }
                Op::PrefixXor(a) => {
                    let parity = prefix_xor(self.values[a.0 as usize]) ^ self.carries[i];
                    self.carries[i] = ((parity as i64) >> 63) as u64;
                    parity
                }
                Op::Add(a, b) => {
                    let (partial, c1) =
                        self.values[a.0 as usize].overflowing_add(self.values[b.0 as usize]);
                    let (sum, c2) = partial.overflowing_add(self.carries[i]);
                    self.carries[i] = (c1 | c2) as u64;
                    sum
                }
            };
        }
        self.values[self.graph.output().0 as usize]
    }
}

/// Bit i of the result is the XOR of bits 0..=i (log-step shift cascade; the
/// scalar equivalent of the PCLMULQDQ trick).
fn prefix_xor(mut x: u64) -> u64 {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    x
}

/// Run `graph` over `data`, appending the byte offsets of the output
/// stream's set bits to `out`.
pub fn run(graph: &Graph, data: &[u8], out: &mut Vec<u32>) {
    let mut machine = Machine::new(graph);
    let mut offset = 0usize;

    let mut push_block = |machine: &mut Machine, block: &[u8; 64], base: usize, limit: usize| {
        let mut mask = machine.step(block);
        // Discard pad bits beyond the real input.
        if limit < 64 {
            mask &= (1u64 << limit) - 1;
        }
        while mask != 0 {
            out.push(base as u32 + mask.trailing_zeros());
            mask &= mask - 1;
        }
    };

    while offset + 64 <= data.len() {
        let block: &[u8; 64] = data[offset..offset + 64].try_into().unwrap();
        push_block(&mut machine, block, offset, 64);
        offset += 64;
    }

    let rem = data.len() - offset;
    if rem > 0 {
        let mut block = [0u8; 64];
        block[..rem].copy_from_slice(&data[offset..]);
        push_block(&mut machine, &block, offset, rem);
    }
}
