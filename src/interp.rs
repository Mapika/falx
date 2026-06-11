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
        // Seeded shifts start with a carried-in 1 (stream position 0
        // behaves as preceded by a match); everything else starts at 0.
        let carries = graph
            .nodes()
            .iter()
            .map(|op| matches!(op, Op::ShiftLeft1Seeded(_)) as u64)
            .collect();
        Self {
            graph,
            values: vec![0; graph.nodes().len()],
            carries,
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
                Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) => {
                    let v = self.values[a.0 as usize];
                    let out = (v << 1) | self.carries[i];
                    self.carries[i] = v >> 63;
                    out
                }
                Op::Regions(q, s, n) => resolve_regions(
                    self.values[q.0 as usize],
                    self.values[s.0 as usize],
                    self.values[n.0 as usize],
                    &mut self.carries[i],
                ),
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

/// Three-state region resolution for [`Op::Regions`]: walk the set bits
/// of `q | s | n` in position order with a normal/quote/comment state
/// machine, filling the inert mask between region open and close events.
/// `state` is the carried region state (0 normal, 1 quote, 2 comment).
fn resolve_regions(q: u64, s: u64, n: u64, state: &mut u64) -> u64 {
    const NORMAL: u64 = 0;
    const QUOTE: u64 = 1;
    const COMMENT: u64 = 2;
    let mut inert = 0u64;
    // A region continuing from the previous block fills from bit 0.
    let mut run_start = 0u32;
    let mut events = q | s | n;
    while events != 0 {
        let p = events.trailing_zeros();
        let bit = 1u64 << p;
        match *state {
            QUOTE => {
                if q & bit != 0 {
                    inert |= range_mask(run_start, p);
                    *state = NORMAL;
                }
            }
            COMMENT => {
                if n & bit != 0 {
                    inert |= range_mask(run_start, p);
                    *state = NORMAL;
                }
            }
            _ => {
                if q & bit != 0 {
                    *state = QUOTE;
                    run_start = p;
                } else if s & bit != 0 {
                    *state = COMMENT;
                    run_start = p;
                }
            }
        }
        events &= events - 1;
    }
    if *state != NORMAL {
        inert |= range_mask(run_start, 64);
    }
    inert
}

/// Bits `[from, to)` set.
fn range_mask(from: u32, to: u32) -> u64 {
    let hi = if to >= 64 { !0u64 } else { (1u64 << to) - 1 };
    hi & !((1u64 << from) - 1)
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
