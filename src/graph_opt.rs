//! Cost-weighted graph normalization before native codegen.
//!
//! This is intentionally a small equality-style simplifier, not a full
//! e-graph engine yet. It rebuilds only the nodes needed by parser outputs,
//! applies deterministic boolean rewrites, and preserves stateful operations
//! as opaque nodes.
//!
//! Rewrites come in two tiers. Conservative rewrites (CSE, constant folding,
//! commutative canonicalization, idempotent/inverse identities, dead-node
//! sweeping) never grow the graph. Speculative rewrites (Not-extraction
//! through Xor, class-algebra fusion of `And`/`Or`/`Xor` over `Class`
//! operands) usually win but can lose when the rewritten subterm is shared,
//! so both candidates are built and the cheaper one under the [`CostModel`]
//! is extracted — the same build-candidates-then-extract shape a future
//! e-graph implementation would scale up.

use crate::formats::DelimitedParts;
use crate::ir::{CharClass, Graph, NodeId, Op};
use crate::synth::{CostModel, graph_cost};
use std::collections::HashMap;

/// Largest fused class produced by `Or`/`Xor` class fusion. Classes up to 8
/// members compile to SIMD byte compares; past that codegen switches to the
/// shuffle-lookup path, whose cost the flat per-class model cannot see, so
/// fusion stops at the compare-friendly boundary.
const FUSED_CLASS_LIMIT: usize = 8;

#[derive(Clone)]
pub struct OptimizedParts {
    pub parts: DelimitedParts,
    pub stats: OptimizationStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptimizationStats {
    pub original_nodes: usize,
    pub optimized_nodes: usize,
    pub removed_nodes: usize,
    pub original_cost: u32,
    pub optimized_cost: u32,
    pub applied: bool,
}

pub fn optimize_parts(parts: DelimitedParts, model: CostModel) -> OptimizedParts {
    let mut roots = vec![parts.graph.output(), parts.terminators];
    if let Some((opens, closes)) = parts.nest {
        roots.extend([opens, closes]);
    }

    let optimized = optimize_roots(&parts.graph, &roots, model);
    if !optimized.stats.applied {
        let original_nodes = parts.graph.nodes().len();
        let original_cost = graph_cost(&parts.graph, &model);
        return OptimizedParts {
            parts,
            stats: OptimizationStats {
                original_nodes,
                optimized_nodes: original_nodes,
                removed_nodes: 0,
                original_cost,
                optimized_cost: original_cost,
                applied: false,
            },
        };
    }

    let OptimizedGraph {
        mut graph,
        roots,
        stats,
    } = optimized;
    graph.set_output(roots[0]);

    let nest = parts.nest.map(|_| (roots[2], roots[3]));
    OptimizedParts {
        parts: DelimitedParts {
            graph,
            terminators: roots[1],
            nest,
        },
        stats,
    }
}

struct OptimizedGraph {
    graph: Graph,
    roots: Vec<NodeId>,
    stats: OptimizationStats,
}

fn optimize_roots(source: &Graph, roots: &[NodeId], model: CostModel) -> OptimizedGraph {
    // Two candidates, cheapest-wins extraction: the conservative rebuild only
    // applies rewrites that can never grow the graph (CSE, constant folding,
    // dead-node pruning), while the speculative rebuild adds restructuring
    // rewrites (Not-extraction through Xor, class-algebra fusion) that win on
    // typical graphs but can lose when a rewritten subterm is shared.
    let conservative = rebuild(source, roots, false);
    let speculative = rebuild(source, roots, true);
    let (graph, mapped_roots) =
        if graph_cost(&speculative.0, &model) < graph_cost(&conservative.0, &model) {
            speculative
        } else {
            conservative
        };
    let original_nodes = source.nodes().len();
    let optimized_nodes = graph.nodes().len();
    let stats = OptimizationStats {
        original_nodes,
        optimized_nodes,
        removed_nodes: original_nodes.saturating_sub(optimized_nodes),
        original_cost: graph_cost(source, &model),
        optimized_cost: graph_cost(&graph, &model),
        applied: false,
    };
    let stats = OptimizationStats {
        applied: stats.optimized_cost < stats.original_cost,
        ..stats
    };
    OptimizedGraph {
        graph,
        roots: mapped_roots,
        stats,
    }
}

fn rebuild(source: &Graph, roots: &[NodeId], speculative: bool) -> (Graph, Vec<NodeId>) {
    let mut rebuilder = Rebuilder::new(source, speculative);
    let mapped_roots: Vec<NodeId> = roots
        .iter()
        .copied()
        .map(|root| rebuilder.optimize_node(root))
        .collect();
    // The rebuilder interns operands before simplification decides whether
    // they survive, so rewrites strand intermediates (e.g. the Not nodes a
    // collapsed `!(a ^ !b)` chain leaves behind). Sweep so dead nodes don't
    // distort candidate costs.
    sweep(&rebuilder.finish(), &mapped_roots)
}

/// Drop every node unreachable from `roots`, compacting ids while preserving
/// topological order.
fn sweep(graph: &Graph, roots: &[NodeId]) -> (Graph, Vec<NodeId>) {
    let nodes = graph.nodes();
    let mut live = vec![false; nodes.len()];
    let mut stack: Vec<NodeId> = roots.to_vec();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut live[id.0 as usize], true) {
            continue;
        }
        match nodes[id.0 as usize] {
            Op::Class(_) | Op::Const(_) => {}
            Op::Not(a) | Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) | Op::PrefixXor(a) => {
                stack.push(a);
            }
            Op::And(a, b) | Op::Or(a, b) | Op::Xor(a, b) | Op::Add(a, b) => {
                stack.extend([a, b]);
            }
            Op::Regions(a, b, c) => stack.extend([a, b, c]),
        }
    }

    let mut compact = Graph::new();
    let mut map: Vec<Option<NodeId>> = vec![None; nodes.len()];
    for (index, op) in nodes.iter().enumerate() {
        if !live[index] {
            continue;
        }
        let at = |id: NodeId| map[id.0 as usize].expect("operand precedes node in topo order");
        let new_id = match *op {
            Op::Class(class) => compact.class(class),
            Op::Const(pattern) => compact.constant(pattern),
            Op::Not(a) => compact.not(at(a)),
            Op::And(a, b) => compact.and(at(a), at(b)),
            Op::Or(a, b) => compact.or(at(a), at(b)),
            Op::Xor(a, b) => compact.xor(at(a), at(b)),
            Op::ShiftLeft1(a) => compact.shift_left1(at(a)),
            Op::ShiftLeft1Seeded(a) => compact.shift_left1_seeded(at(a)),
            Op::PrefixXor(a) => compact.prefix_xor(at(a)),
            Op::Add(a, b) => compact.add(at(a), at(b)),
            Op::Regions(q, s, n) => compact.regions(at(q), at(s), at(n)),
        };
        map[index] = Some(new_id);
    }
    let roots = roots
        .iter()
        .map(|root| map[root.0 as usize].expect("root is live by construction"))
        .collect();
    (compact, roots)
}

/// Facts the rebuilder knows about a node in the *rebuilt* graph, used by
/// later rewrites without re-walking the graph.
#[derive(Clone, Copy, Default)]
struct NodeFacts {
    constant: Option<u64>,
    not_child: Option<NodeId>,
    class: Option<CharClass>,
}

struct Rebuilder<'g> {
    source: &'g Graph,
    graph: Graph,
    memo: Vec<Option<NodeId>>,
    cse: HashMap<CseKey, NodeId>,
    facts: Vec<NodeFacts>,
    depth: Vec<u32>,
    speculative: bool,
}

impl<'g> Rebuilder<'g> {
    fn new(source: &'g Graph, speculative: bool) -> Self {
        Self {
            source,
            graph: Graph::new(),
            memo: vec![None; source.nodes().len()],
            cse: HashMap::new(),
            facts: Vec::new(),
            depth: depths(source),
            speculative,
        }
    }

    /// Map both operands, visiting the deeper subtree first (Sethi–Ullman
    /// order). Codegen emits nodes in graph order, so visiting the shallow
    /// operand last places it adjacent to its consumer instead of hoisting
    /// it to the top of the block, where its value would stay live across
    /// the whole computation and raise register pressure in the emitted
    /// kernel loop.
    fn map_operands(&mut self, a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if self.depth[a.0 as usize] >= self.depth[b.0 as usize] {
            let a = self.optimize_node(a);
            let b = self.optimize_node(b);
            (a, b)
        } else {
            let b = self.optimize_node(b);
            let a = self.optimize_node(a);
            (a, b)
        }
    }

    fn finish(self) -> Graph {
        self.graph
    }

    fn optimize_node(&mut self, old: NodeId) -> NodeId {
        if let Some(mapped) = self.memo[old.0 as usize] {
            return mapped;
        }

        let mapped = match self.source.nodes()[old.0 as usize].clone() {
            Op::Class(class) => self.intern_class(class),
            Op::Const(pattern) => self.intern_const(pattern),
            Op::Not(a) => {
                let a = self.optimize_node(a);
                self.simplify_not(a)
            }
            Op::And(a, b) => {
                let (a, b) = self.map_operands(a, b);
                self.simplify_and(a, b)
            }
            Op::Or(a, b) => {
                let (a, b) = self.map_operands(a, b);
                self.simplify_or(a, b)
            }
            Op::Xor(a, b) => {
                let (a, b) = self.map_operands(a, b);
                self.simplify_xor(a, b)
            }
            Op::ShiftLeft1(a) => {
                let a = self.optimize_node(a);
                self.intern_unary(CseKey::ShiftLeft1(a.0), |graph| graph.shift_left1(a))
            }
            Op::ShiftLeft1Seeded(a) => {
                let a = self.optimize_node(a);
                self.intern_unary(CseKey::ShiftLeft1Seeded(a.0), |graph| {
                    graph.shift_left1_seeded(a)
                })
            }
            Op::PrefixXor(a) => {
                let a = self.optimize_node(a);
                self.intern_unary(CseKey::PrefixXor(a.0), |graph| graph.prefix_xor(a))
            }
            Op::Add(a, b) => {
                let (a, b) = self.map_operands(a, b);
                let (a, b) = ordered(a, b);
                self.intern_binary(CseKey::Add(a.0, b.0), |graph| graph.add(a, b))
            }
            Op::Regions(q, s, n) => {
                let mut by_depth = [q, s, n];
                by_depth.sort_by_key(|id| std::cmp::Reverse(self.depth[id.0 as usize]));
                for operand in by_depth {
                    self.optimize_node(operand);
                }
                let q = self.optimize_node(q);
                let s = self.optimize_node(s);
                let n = self.optimize_node(n);
                self.intern_ternary(CseKey::Regions(q.0, s.0, n.0), |graph| {
                    graph.regions(q, s, n)
                })
            }
        };
        self.memo[old.0 as usize] = Some(mapped);
        mapped
    }

    fn simplify_not(&mut self, a: NodeId) -> NodeId {
        if let Some(value) = self.constant(a) {
            return self.intern_const(!value);
        }
        if let Some(inner) = self.not_child(a) {
            return inner;
        }
        self.intern_not(a)
    }

    fn simplify_and(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if a == b {
            return a;
        }
        if self.are_inverses(a, b) {
            return self.intern_const(0);
        }
        match (self.constant(a), self.constant(b)) {
            (Some(x), Some(y)) => return self.intern_const(x & y),
            (Some(0), _) | (_, Some(0)) => return self.intern_const(0),
            (Some(u64::MAX), _) => return b,
            (_, Some(u64::MAX)) => return a,
            _ => {}
        }
        // Intersection can only shrink a class, so no size gate.
        if let Some(fused) = self.fuse_classes(a, b, |x, y| x & y, 256) {
            return fused;
        }
        let (a, b) = ordered(a, b);
        self.intern_binary(CseKey::And(a.0, b.0), |graph| graph.and(a, b))
    }

    fn simplify_or(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if a == b {
            return a;
        }
        if self.are_inverses(a, b) {
            return self.intern_const(u64::MAX);
        }
        match (self.constant(a), self.constant(b)) {
            (Some(x), Some(y)) => return self.intern_const(x | y),
            (Some(0), _) => return b,
            (_, Some(0)) => return a,
            (Some(u64::MAX), _) | (_, Some(u64::MAX)) => return self.intern_const(u64::MAX),
            _ => {}
        }
        if let Some(fused) = self.fuse_classes(a, b, |x, y| x | y, FUSED_CLASS_LIMIT) {
            return fused;
        }
        let (a, b) = ordered(a, b);
        self.intern_binary(CseKey::Or(a.0, b.0), |graph| graph.or(a, b))
    }

    fn simplify_xor(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if a == b {
            return self.intern_const(0);
        }
        if self.are_inverses(a, b) {
            return self.intern_const(u64::MAX);
        }
        match (self.constant(a), self.constant(b)) {
            (Some(x), Some(y)) => return self.intern_const(x ^ y),
            (Some(0), _) => return b,
            (_, Some(0)) => return a,
            (Some(u64::MAX), _) => return self.simplify_not(b),
            (_, Some(u64::MAX)) => return self.simplify_not(a),
            _ => {}
        }
        if let Some(fused) = self.fuse_classes(a, b, |x, y| x ^ y, FUSED_CLASS_LIMIT) {
            return fused;
        }
        // Not-extraction: Xor(a, !b) == !Xor(a, b). The pulled-out Not then
        // cancels against an enclosing Not via `simplify_not`, which is what
        // collapses the odd-backslash escape chain
        // `!(EVEN ^ !Add(..)) -> EVEN ^ Add(..)`.
        if self.speculative {
            match (self.not_child(a), self.not_child(b)) {
                (Some(x), Some(y)) => return self.simplify_xor(x, y),
                (Some(x), None) => {
                    let inner = self.simplify_xor(x, b);
                    return self.simplify_not(inner);
                }
                (None, Some(y)) => {
                    let inner = self.simplify_xor(a, y);
                    return self.simplify_not(inner);
                }
                (None, None) => {}
            }
        }
        let (a, b) = ordered(a, b);
        self.intern_binary(CseKey::Xor(a.0, b.0), |graph| graph.xor(a, b))
    }

    /// Fold a bitwise op over two `Class` operands into a single class node
    /// covering the combined membership, when the result stays at or below
    /// `limit` members (so it never leaves codegen's compare-friendly range).
    /// Speculative: a fused class is a *new* node, so this loses when both
    /// operand classes have other users.
    fn fuse_classes(
        &mut self,
        a: NodeId,
        b: NodeId,
        op: impl Fn(u64, u64) -> u64,
        limit: usize,
    ) -> Option<NodeId> {
        if !self.speculative {
            return None;
        }
        let (ca, cb) = (self.class_of(a)?, self.class_of(b)?);
        let (wa, wb) = (ca.words(), cb.words());
        let fused = CharClass::from_words([
            op(wa[0], wb[0]),
            op(wa[1], wb[1]),
            op(wa[2], wb[2]),
            op(wa[3], wb[3]),
        ]);
        if fused.is_empty() {
            return Some(self.intern_const(0));
        }
        if fused.len() > limit {
            return None;
        }
        Some(self.intern_class(fused))
    }

    fn intern_class(&mut self, class: CharClass) -> NodeId {
        let key = CseKey::Class(class.words());
        let facts = NodeFacts {
            class: Some(class),
            ..NodeFacts::default()
        };
        self.intern_with(key, facts, |graph| graph.class(class))
    }

    fn intern_const(&mut self, value: u64) -> NodeId {
        let facts = NodeFacts {
            constant: Some(value),
            ..NodeFacts::default()
        };
        self.intern_with(CseKey::Const(value), facts, |graph| graph.constant(value))
    }

    fn intern_not(&mut self, a: NodeId) -> NodeId {
        let facts = NodeFacts {
            not_child: Some(a),
            ..NodeFacts::default()
        };
        self.intern_with(CseKey::Not(a.0), facts, |graph| graph.not(a))
    }

    fn intern_unary(&mut self, key: CseKey, push: impl FnOnce(&mut Graph) -> NodeId) -> NodeId {
        self.intern_with(key, NodeFacts::default(), push)
    }

    fn intern_binary(&mut self, key: CseKey, push: impl FnOnce(&mut Graph) -> NodeId) -> NodeId {
        self.intern_with(key, NodeFacts::default(), push)
    }

    fn intern_ternary(&mut self, key: CseKey, push: impl FnOnce(&mut Graph) -> NodeId) -> NodeId {
        self.intern_with(key, NodeFacts::default(), push)
    }

    fn intern_with(
        &mut self,
        key: CseKey,
        facts: NodeFacts,
        push: impl FnOnce(&mut Graph) -> NodeId,
    ) -> NodeId {
        if let Some(&id) = self.cse.get(&key) {
            return id;
        }
        let id = push(&mut self.graph);
        self.cse.insert(key, id);
        self.facts.push(facts);
        debug_assert_eq!(self.facts.len(), self.graph.nodes().len());
        id
    }

    fn constant(&self, id: NodeId) -> Option<u64> {
        self.facts[id.0 as usize].constant
    }

    fn not_child(&self, id: NodeId) -> Option<NodeId> {
        self.facts[id.0 as usize].not_child
    }

    fn class_of(&self, id: NodeId) -> Option<CharClass> {
        self.facts[id.0 as usize].class
    }

    fn are_inverses(&self, a: NodeId, b: NodeId) -> bool {
        self.not_child(a) == Some(b) || self.not_child(b) == Some(a)
    }
}

/// Expression depth of every node — leaves are 0. Nodes are topologically
/// ordered by construction, so one forward pass suffices.
fn depths(graph: &Graph) -> Vec<u32> {
    let mut depth = vec![0u32; graph.nodes().len()];
    for (index, op) in graph.nodes().iter().enumerate() {
        let at = |id: NodeId| depth[id.0 as usize];
        depth[index] = match *op {
            Op::Class(_) | Op::Const(_) => 0,
            Op::Not(a) | Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) | Op::PrefixXor(a) => {
                at(a) + 1
            }
            Op::And(a, b) | Op::Or(a, b) | Op::Xor(a, b) | Op::Add(a, b) => at(a).max(at(b)) + 1,
            Op::Regions(q, s, n) => at(q).max(at(s)).max(at(n)) + 1,
        };
    }
    depth
}

fn ordered(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a.0 <= b.0 { (a, b) } else { (b, a) }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum CseKey {
    Class([u64; 4]),
    Const(u64),
    Not(u32),
    And(u32, u32),
    Or(u32, u32),
    Xor(u32, u32),
    ShiftLeft1(u32),
    ShiftLeft1Seeded(u32),
    PrefixXor(u32),
    Add(u32, u32),
    Regions(u32, u32, u32),
}
