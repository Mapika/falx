//! Cost-weighted graph normalization before native codegen.
//!
//! This is intentionally a small equality-style simplifier, not a full
//! e-graph engine yet. It rebuilds only the nodes needed by parser outputs,
//! applies deterministic boolean rewrites, and preserves stateful operations
//! as opaque nodes.

use crate::formats::DelimitedParts;
use crate::ir::{CharClass, Graph, NodeId, Op};
use crate::synth::{CostModel, graph_cost};
use std::collections::HashMap;

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
    let mut rebuilder = Rebuilder::new(source);
    let mapped_roots = roots
        .iter()
        .copied()
        .map(|root| rebuilder.optimize_node(root))
        .collect();
    let graph = rebuilder.finish();
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

struct Rebuilder<'g> {
    source: &'g Graph,
    graph: Graph,
    memo: Vec<Option<NodeId>>,
    cse: HashMap<CseKey, NodeId>,
    constants: Vec<Option<u64>>,
    not_child: Vec<Option<NodeId>>,
}

impl<'g> Rebuilder<'g> {
    fn new(source: &'g Graph) -> Self {
        Self {
            source,
            graph: Graph::new(),
            memo: vec![None; source.nodes().len()],
            cse: HashMap::new(),
            constants: Vec::new(),
            not_child: Vec::new(),
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
                let a = self.optimize_node(a);
                let b = self.optimize_node(b);
                self.simplify_and(a, b)
            }
            Op::Or(a, b) => {
                let a = self.optimize_node(a);
                let b = self.optimize_node(b);
                self.simplify_or(a, b)
            }
            Op::Xor(a, b) => {
                let a = self.optimize_node(a);
                let b = self.optimize_node(b);
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
                let a = self.optimize_node(a);
                let b = self.optimize_node(b);
                let (a, b) = ordered(a, b);
                self.intern_binary(CseKey::Add(a.0, b.0), |graph| graph.add(a, b))
            }
            Op::Regions(q, s, n) => {
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
        let (a, b) = ordered(a, b);
        self.intern_binary(CseKey::Xor(a.0, b.0), |graph| graph.xor(a, b))
    }

    fn intern_class(&mut self, class: CharClass) -> NodeId {
        let key = CseKey::Class(class.words());
        self.intern_with(key, None, None, |graph| graph.class(class))
    }

    fn intern_const(&mut self, value: u64) -> NodeId {
        self.intern_with(CseKey::Const(value), Some(value), None, |graph| {
            graph.constant(value)
        })
    }

    fn intern_not(&mut self, a: NodeId) -> NodeId {
        self.intern_with(CseKey::Not(a.0), None, Some(a), |graph| graph.not(a))
    }

    fn intern_unary(&mut self, key: CseKey, push: impl FnOnce(&mut Graph) -> NodeId) -> NodeId {
        self.intern_with(key, None, None, push)
    }

    fn intern_binary(&mut self, key: CseKey, push: impl FnOnce(&mut Graph) -> NodeId) -> NodeId {
        self.intern_with(key, None, None, push)
    }

    fn intern_ternary(&mut self, key: CseKey, push: impl FnOnce(&mut Graph) -> NodeId) -> NodeId {
        self.intern_with(key, None, None, push)
    }

    fn intern_with(
        &mut self,
        key: CseKey,
        constant: Option<u64>,
        not_child: Option<NodeId>,
        push: impl FnOnce(&mut Graph) -> NodeId,
    ) -> NodeId {
        if let Some(&id) = self.cse.get(&key) {
            return id;
        }
        let id = push(&mut self.graph);
        self.cse.insert(key, id);
        self.constants.push(constant);
        self.not_child.push(not_child);
        debug_assert_eq!(self.constants.len(), self.graph.nodes().len());
        debug_assert_eq!(self.not_child.len(), self.graph.nodes().len());
        id
    }

    fn constant(&self, id: NodeId) -> Option<u64> {
        self.constants[id.0 as usize]
    }

    fn not_child(&self, id: NodeId) -> Option<NodeId> {
        self.not_child[id.0 as usize]
    }

    fn are_inverses(&self, a: NodeId, b: NodeId) -> bool {
        self.not_child(a) == Some(b) || self.not_child(b) == Some(a)
    }
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
