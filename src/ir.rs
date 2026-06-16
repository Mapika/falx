//! The bitstream IR: the algebra that SIMD parsing kernels are made of.
//!
//! A program is a DAG of operations over *bitstreams* — conceptually infinite
//! bit vectors with one bit per input byte. Executed blockwise (64 bytes at a
//! time), every operation maps to one or two machine instructions, which is
//! what makes this IR a credible codegen target: the M0 hand-written CSV
//! kernel is exactly the graph built by [`crate::formats::delimited`].
//!
//! Operations are either stateless per block (`Class`, bitwise logic) or
//! carry a small fixed state across blocks (`ShiftLeft1` carries one bit,
//! `PrefixXor` carries a parity). That carried state is the entire memory of
//! a kernel, which is why these parsers stream at GB/s: no lookback, no
//! backtracking, no allocation.
//!
//! Node operands always refer to earlier nodes (enforced by the builder), so
//! a graph is in topological order by construction and evaluators can run a
//! single forward pass per block.

/// A set of byte values, the predicate of a [`Op::Class`] node.
///
/// Backends are free to implement small classes as SIMD compares and large
/// ones as shuffle-based lookup tables; the IR only cares about membership.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CharClass {
    bits: [u64; 4],
}

impl CharClass {
    pub const fn empty() -> Self {
        Self { bits: [0; 4] }
    }

    pub const fn from_byte(byte: u8) -> Self {
        let mut class = Self::empty();
        class.bits[(byte >> 6) as usize] |= 1 << (byte & 63);
        class
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut class = Self::empty();
        for &byte in bytes {
            class.bits[(byte >> 6) as usize] |= 1 << (byte & 63);
        }
        class
    }

    pub const fn contains(&self, byte: u8) -> bool {
        self.bits[(byte >> 6) as usize] & (1 << (byte & 63)) != 0
    }

    /// The class as a 256-bit membership bitmap (word `b >> 6`, bit
    /// `b & 63`) — the scalar lookup-table representation.
    pub const fn words(&self) -> [u64; 4] {
        self.bits
    }

    /// Rebuild a class from its membership bitmap (inverse of [`Self::words`]).
    pub const fn from_words(bits: [u64; 4]) -> Self {
        Self { bits }
    }

    pub fn len(&self) -> usize {
        self.bits
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.bits == [0; 4]
    }

    pub fn members(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=255u8).filter(|&byte| self.contains(byte))
    }
}

impl std::fmt::Debug for CharClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CharClass[")?;
        for byte in 0..=255u8 {
            if self.contains(byte) {
                write!(f, "{}", byte.escape_ascii())?;
            }
        }
        write!(f, "]")
    }
}

/// Reference to a node within its graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeId(pub(crate) u32);

/// One bitstream operation.
#[derive(Clone, Debug)]
pub enum Op {
    /// Bit i is set iff input byte i is in the class.
    Class(CharClass),
    /// The same 64-bit pattern in every block. Patterns whose period divides
    /// 64 (such as the even/odd-position masks used by escape handling) are
    /// position-consistent across the whole stream.
    Const(u64),
    Not(NodeId),
    And(NodeId, NodeId),
    Or(NodeId, NodeId),
    Xor(NodeId, NodeId),
    /// Bit i of the result is bit i-1 of the operand ("the previous byte
    /// matched"). Carries one bit across blocks.
    ShiftLeft1(NodeId),
    /// Like [`Op::ShiftLeft1`] but the very first block's carried-in bit is
    /// 1: the stream behaves as if preceded by one matching byte. Used to
    /// make stream position 0 count as a line start.
    ShiftLeft1Seeded(NodeId),
    /// Sequential three-state region resolution for comment support:
    /// given real-quote toggle bits, line-start comment-candidate bits,
    /// and terminator bits, the result marks every byte inside a quoted
    /// region (open quote included, close excluded — prefix-XOR
    /// convention) or inside a comment (comment byte included, the
    /// terminating newline excluded). Quote bits are inert inside
    /// comments and candidate bits are inert inside quotes, which is the
    /// interleaving no bit-parallel parity can express: this is the one
    /// operation that propagates *within* a block, walking the set bits
    /// of its inputs in position order (cheap: events are rare). Carries
    /// the region state (normal/quote/comment) across blocks.
    Regions(NodeId, NodeId, NodeId),
    /// Bit i is the XOR of operand bits 0..=i — running parity, the
    /// quote-context primitive. Carries one parity bit across blocks.
    PrefixXor(NodeId),
    /// 64-bit binary addition of the two operand blocks, with the carry-out
    /// propagated into the next block. Addition makes a set bit ripple
    /// through a run of contiguous set bits — the primitive behind
    /// odd/even-length run detection for backslash escapes.
    Add(NodeId, NodeId),
}

/// A bitstream program: nodes in topological order plus a designated output
/// stream whose set bits are the structural positions.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<Op>,
    output: Option<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn nodes(&self) -> &[Op] {
        &self.nodes
    }

    /// The designated output stream.
    ///
    /// Panics if no output was set; a graph without an output is a
    /// construction bug, not a runtime condition.
    pub fn output(&self) -> NodeId {
        self.output.expect("graph has no output node set")
    }

    pub fn set_output(&mut self, id: NodeId) {
        self.check(id);
        self.output = Some(id);
    }

    fn check(&self, id: NodeId) {
        assert!(
            (id.0 as usize) < self.nodes.len(),
            "operand {id:?} does not exist in this graph"
        );
    }

    fn push(&mut self, op: Op) -> NodeId {
        let id = NodeId(u32::try_from(self.nodes.len()).expect("graph too large"));
        self.nodes.push(op);
        id
    }

    pub fn class(&mut self, class: CharClass) -> NodeId {
        self.push(Op::Class(class))
    }

    pub fn class_byte(&mut self, byte: u8) -> NodeId {
        self.class(CharClass::from_byte(byte))
    }

    pub fn constant(&mut self, pattern: u64) -> NodeId {
        self.push(Op::Const(pattern))
    }

    pub fn not(&mut self, a: NodeId) -> NodeId {
        self.check(a);
        self.push(Op::Not(a))
    }

    pub fn and(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.check(a);
        self.check(b);
        self.push(Op::And(a, b))
    }

    pub fn or(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.check(a);
        self.check(b);
        self.push(Op::Or(a, b))
    }

    pub fn xor(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.check(a);
        self.check(b);
        self.push(Op::Xor(a, b))
    }

    pub fn shift_left1(&mut self, a: NodeId) -> NodeId {
        self.check(a);
        self.push(Op::ShiftLeft1(a))
    }

    pub fn shift_left1_seeded(&mut self, a: NodeId) -> NodeId {
        self.check(a);
        self.push(Op::ShiftLeft1Seeded(a))
    }

    pub fn regions(
        &mut self,
        quotes: NodeId,
        comment_starts: NodeId,
        terminators: NodeId,
    ) -> NodeId {
        self.check(quotes);
        self.check(comment_starts);
        self.check(terminators);
        self.push(Op::Regions(quotes, comment_starts, terminators))
    }

    pub fn prefix_xor(&mut self, a: NodeId) -> NodeId {
        self.check(a);
        self.push(Op::PrefixXor(a))
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.check(a);
        self.check(b);
        self.push(Op::Add(a, b))
    }
}
