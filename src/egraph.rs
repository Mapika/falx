//! Equality-saturation graph optimization.
//!
//! [`crate::graph_opt`] is a two-candidate simplifier: it builds one
//! conservative rebuild and one speculative rebuild of the whole graph and
//! keeps the cheaper. That choice is global — speculate everywhere or nowhere.
//! This module is the e-graph the `graph_opt` docstring anticipated: it keeps
//! *every* equivalent form of every subterm at once (congruence closure over a
//! union-find of e-classes), saturates a rule set, and extracts the globally
//! cheapest program under the [`CostModel`]. The payoff over `graph_opt` is
//! per-subterm mixing — speculate on the subterm where fusion wins while
//! staying conservative on the subterm where a shared `Not` makes it lose, in
//! the *same* graph.
//!
//! Soundness rests on `Not`/`And`/`Or`/`Xor` being genuinely bitwise over the
//! u64 blocks the kernels process: associativity, commutativity, De Morgan and
//! distributivity hold bit-for-bit regardless of what feeds them, and the
//! stateful ops (`ShiftLeft1`, `PrefixXor`, `Add`, `Regions`) are carried as
//! opaque operators — no algebraic rule rewrites *through* them, because their
//! cross-block carry semantics make reassociation unsound.
//!
//! Determinism is a hard requirement: codegen must stay byte-identical across
//! runs, so every map that influences the result is ordered (`BTreeMap`) and
//! every iteration is by ascending e-class id / insertion order, with a total
//! tie-break on the canonical e-node during extraction.

use crate::formats::DelimitedParts;
use crate::graph_opt::{OptimizationStats, OptimizedParts};
use crate::ir::{CharClass, Graph, NodeId, Op};
use crate::synth::{CostModel, graph_cost};
use std::collections::BTreeMap;

/// Largest fused class produced by `Or`/`Xor` class fusion — past 8 members
/// codegen leaves the compare-friendly range, so fusion stops there. `And`
/// (intersection) can only shrink a class, so it uses 256 (always fuse). These
/// mirror [`crate::graph_opt`].
const FUSED_OR_XOR_LIMIT: usize = 8;
const FUSED_AND_LIMIT: usize = 256;

/// Saturation backstops. Real dialect graphs are tiny (≤18 nodes), so these
/// never bind in practice; they only bound pathological distributive blow-up.
/// Extraction is sound even if a cap trips mid-saturation.
/// Per-phase saturation pass cap; convergence detection ends a phase early once
/// a pass makes no progress, so this only bounds pathological inputs.
const MAX_ITERS: usize = 60;
const MAX_ENODES: usize = 100_000;

/// Largest same-op chain [`EGraph::rules_ac`] will flatten in one shot; beyond
/// this it leaves the node alone (a backstop against pathological inputs).
const AC_FLATTEN_CAP: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct EClassId(u32);

/// An e-node: an [`Op`] whose operands are e-classes, not nodes. `Class` holds
/// the raw membership bitmap (rather than [`CharClass`]) so the enum derives
/// `Ord`/`Hash` for hashconsing and the extraction tie-break.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum ENode {
    Class([u64; 4]),
    Const(u64),
    Not(EClassId),
    And(EClassId, EClassId),
    Or(EClassId, EClassId),
    Xor(EClassId, EClassId),
    ShiftLeft1(EClassId),
    ShiftLeft1Seeded(EClassId),
    PrefixXor(EClassId),
    Add(EClassId, EClassId),
    Regions(EClassId, EClassId, EClassId),
}

fn weight(node: &ENode, model: &CostModel) -> u64 {
    let w = match node {
        ENode::Class(_) => model.class,
        ENode::Const(_) => model.constant,
        ENode::Not(_) | ENode::And(..) | ENode::Or(..) | ENode::Xor(..) => model.bitwise,
        ENode::ShiftLeft1(_) | ENode::ShiftLeft1Seeded(_) => model.shift,
        ENode::Add(..) => model.add,
        ENode::PrefixXor(_) => model.prefix_xor,
        ENode::Regions(..) => model.regions,
    };
    w as u64
}

fn children(node: &ENode) -> Vec<EClassId> {
    match *node {
        ENode::Class(_) | ENode::Const(_) => Vec::new(),
        ENode::Not(a) | ENode::ShiftLeft1(a) | ENode::ShiftLeft1Seeded(a) | ENode::PrefixXor(a) => {
            vec![a]
        }
        ENode::And(a, b) | ENode::Or(a, b) | ENode::Xor(a, b) | ENode::Add(a, b) => vec![a, b],
        ENode::Regions(a, b, c) => vec![a, b, c],
    }
}

struct EGraph {
    /// Union-find parent per e-class.
    parent: Vec<EClassId>,
    /// E-nodes grouped under their canonical e-class. After [`Self::repair`]
    /// each canonical class's vec is sorted+deduped and non-canonical classes
    /// are empty.
    classes: Vec<Vec<ENode>>,
    /// Canonical e-node -> canonical e-class. Ordered for determinism.
    hashcons: BTreeMap<ENode, EClassId>,
    /// Count of real merges performed, for the saturation fixpoint check.
    unions: u64,
    /// Saturation phase. When false, only the cheap, non-exploding rewrites
    /// (the two-candidate optimizer's set) run; when true, the expensive new
    /// rules (AC-flatten, factoring) also run. See [`Self::saturate`].
    full: bool,
}

impl EGraph {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            classes: Vec::new(),
            hashcons: BTreeMap::new(),
            unions: 0,
            full: false,
        }
    }

    /// Immutable union-find lookup (no path compression) — used wherever a
    /// shared borrow of `self` is already held.
    fn find_imm(&self, id: EClassId) -> EClassId {
        let mut id = id;
        while self.parent[id.0 as usize] != id {
            id = self.parent[id.0 as usize];
        }
        id
    }

    /// Rewrite an e-node's operands to their canonical classes, and sort the
    /// operands of commutative ops so `op(a,b)` and `op(b,a)` hashcons to one
    /// node — commutativity for free, with no rewrite rule that could loop.
    fn canonicalize(&self, node: &ENode) -> ENode {
        let f = |id: EClassId| self.find_imm(id);
        match *node {
            ENode::Class(w) => ENode::Class(w),
            ENode::Const(c) => ENode::Const(c),
            ENode::Not(a) => ENode::Not(f(a)),
            ENode::ShiftLeft1(a) => ENode::ShiftLeft1(f(a)),
            ENode::ShiftLeft1Seeded(a) => ENode::ShiftLeft1Seeded(f(a)),
            ENode::PrefixXor(a) => ENode::PrefixXor(f(a)),
            ENode::And(a, b) => {
                let (a, b) = ord(f(a), f(b));
                ENode::And(a, b)
            }
            ENode::Or(a, b) => {
                let (a, b) = ord(f(a), f(b));
                ENode::Or(a, b)
            }
            ENode::Xor(a, b) => {
                let (a, b) = ord(f(a), f(b));
                ENode::Xor(a, b)
            }
            ENode::Add(a, b) => {
                let (a, b) = ord(f(a), f(b));
                ENode::Add(a, b)
            }
            // Regions operands have distinct roles (quotes/comments/terms) — not
            // commutative, so they are canonicalized but never reordered.
            ENode::Regions(a, b, c) => ENode::Regions(f(a), f(b), f(c)),
        }
    }

    fn add(&mut self, node: ENode) -> EClassId {
        let node = self.canonicalize(&node);
        if let Some(&id) = self.hashcons.get(&node) {
            return self.find_imm(id);
        }
        let id = EClassId(u32::try_from(self.classes.len()).expect("e-graph too large"));
        self.parent.push(id);
        self.classes.push(vec![node.clone()]);
        self.hashcons.insert(node, id);
        id
    }

    /// Merge two e-classes. Keeps the lower id as the representative (a
    /// deterministic, monotone choice) and moves the absorbed class's e-nodes
    /// onto it. Returns whether a real merge happened.
    fn union(&mut self, a: EClassId, b: EClassId) -> bool {
        let a = self.find_imm(a);
        let b = self.find_imm(b);
        if a == b {
            return false;
        }
        let (keep, gone) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        self.parent[gone.0 as usize] = keep;
        let moved = std::mem::take(&mut self.classes[gone.0 as usize]);
        self.classes[keep.0 as usize].extend(moved);
        self.unions += 1;
        true
    }

    /// Restore congruence closure: if two e-nodes in *different* classes
    /// canonicalize to the same form, those classes must merge, which can
    /// expose further collisions, so iterate to a fixpoint. Graphs are tiny; a
    /// from-scratch rescan per round is simpler and just as deterministic as
    /// incremental parent tracking. Each productive round performs at least one
    /// real merge, and merges are bounded by the class count, so this
    /// terminates — the loop bound is a backstop against ever spinning.
    fn repair(&mut self) {
        for _ in 0..=self.classes.len() {
            let mut seen: BTreeMap<ENode, EClassId> = BTreeMap::new();
            let mut to_union: Vec<(EClassId, EClassId)> = Vec::new();
            for c in 0..self.classes.len() {
                let cid = EClassId(c as u32);
                if self.find_imm(cid) != cid {
                    continue;
                }
                for k in 0..self.classes[c].len() {
                    let canon = self.canonicalize(&self.classes[c][k]);
                    match seen.get(&canon) {
                        // Only a cross-class match is a congruence merge. A
                        // repeated canon within one class is just a duplicate
                        // e-node (left by a prior union) — `dedup_classes`
                        // drops it; unioning a class with itself would no-op and
                        // spin the fixpoint forever.
                        Some(&other) if other != cid => to_union.push((other, cid)),
                        _ => {
                            seen.entry(canon).or_insert(cid);
                        }
                    }
                }
            }
            if to_union.is_empty() {
                self.hashcons = seen;
                self.dedup_classes();
                return;
            }
            for (a, b) in to_union {
                self.union(a, b);
            }
        }
        // Backstop only: normalize and rebuild the hashcons from the deduped
        // canonical classes so the graph is left consistent.
        self.dedup_classes();
        self.hashcons = BTreeMap::new();
        for c in 0..self.classes.len() {
            let cid = EClassId(c as u32);
            if self.find_imm(cid) != cid {
                continue;
            }
            for k in 0..self.classes[c].len() {
                let canon = self.canonicalize(&self.classes[c][k]);
                self.hashcons.entry(canon).or_insert(cid);
            }
        }
    }

    /// Canonicalize, sort and dedup every canonical class's e-node list (and
    /// clear non-canonical lists). Leaves the graph in the ordered shape the
    /// snapshot and extraction passes rely on.
    fn dedup_classes(&mut self) {
        for c in 0..self.classes.len() {
            let cid = EClassId(c as u32);
            if self.find_imm(cid) != cid {
                self.classes[c].clear();
                continue;
            }
            let mut nodes = std::mem::take(&mut self.classes[c]);
            for n in &mut nodes {
                *n = self.canonicalize(n);
            }
            nodes.sort();
            nodes.dedup();
            self.classes[c] = nodes;
        }
    }

    // --- fact accessors over a class's e-nodes (post-repair: canonical) ---

    fn nodes_of(&self, class: EClassId) -> Vec<ENode> {
        self.classes[self.find_imm(class).0 as usize].clone()
    }

    fn const_of(&self, class: EClassId) -> Option<u64> {
        self.nodes_of(class).into_iter().find_map(|n| match n {
            ENode::Const(v) => Some(v),
            _ => None,
        })
    }

    fn class_words(&self, class: EClassId) -> Option<[u64; 4]> {
        self.nodes_of(class).into_iter().find_map(|n| match n {
            ENode::Class(w) => Some(w),
            _ => None,
        })
    }

    /// The inner operand `x` if the class contains a `Not(x)` e-node.
    fn not_child(&self, class: EClassId) -> Option<EClassId> {
        self.nodes_of(class).into_iter().find_map(|n| match n {
            ENode::Not(x) => Some(self.find_imm(x)),
            _ => None,
        })
    }

    fn are_inverses(&self, a: EClassId, b: EClassId) -> bool {
        let (a, b) = (self.find_imm(a), self.find_imm(b));
        self.not_child(a) == Some(b) || self.not_child(b) == Some(a)
    }

    fn snapshot(&self) -> Vec<(EClassId, ENode)> {
        let mut out = Vec::new();
        for c in 0..self.classes.len() {
            let cid = EClassId(c as u32);
            if self.find_imm(cid) != cid {
                continue;
            }
            for node in &self.classes[c] {
                out.push((cid, node.clone()));
            }
        }
        out
    }

    /// Staged saturation. Phase 1 runs only the cheap rewrites (the
    /// two-candidate optimizer's set) to a fixpoint — these never blow up, so
    /// the e-graph after phase 1 already contains every form that optimizer
    /// could reach, guaranteeing the extracted cost is never worse than it.
    /// Phase 2 then adds the expensive new rules (AC-flatten, factoring) under
    /// the iteration / e-node budget: if it caps out on a huge graph it can only
    /// *improve* on phase 1, never regress.
    fn saturate(&mut self) {
        self.repair();
        // Phase 1 runs only the cheap rules (the two-candidate optimizer's set,
        // which never blow up) to a fixpoint, so the e-graph then contains every
        // form that optimizer could reach — the result can never be worse than
        // it. Phase 2 adds the expensive new rules (AC-flatten, factoring) under
        // the same budget; capping out on a huge graph can only improve on
        // phase 1, never regress.
        self.full = false;
        self.run_passes();
        self.full = true;
        self.run_passes();
    }

    fn run_passes(&mut self) {
        for _ in 0..MAX_ITERS {
            let nodes_before = self.classes.len();
            let unions_before = self.unions;
            for (cls, node) in self.snapshot() {
                self.apply_rules(cls, &node);
                if self.classes.len() > MAX_ENODES {
                    break;
                }
            }
            self.repair();
            let no_growth = self.classes.len() == nodes_before;
            let no_union = self.unions == unions_before;
            if (no_growth && no_union) || self.classes.len() > MAX_ENODES {
                break;
            }
        }
    }

    /// Add every equivalent form of `node` reachable in one step and union it
    /// into `cls`. Non-destructive: the original form survives, so extraction
    /// can still choose it.
    fn apply_rules(&mut self, cls: EClassId, node: &ENode) {
        match *node {
            ENode::Class(_) | ENode::Const(_) => {}
            ENode::Not(a) => self.rules_not(cls, a),
            ENode::And(a, b) => self.rules_and(cls, a, b),
            ENode::Or(a, b) => self.rules_or(cls, a, b),
            ENode::Xor(a, b) => self.rules_xor(cls, a, b),
            // Opaque stateful ops: no algebraic rewrites.
            ENode::ShiftLeft1(_)
            | ENode::ShiftLeft1Seeded(_)
            | ENode::PrefixXor(_)
            | ENode::Add(..)
            | ENode::Regions(..) => {}
        }
    }

    fn rules_not(&mut self, cls: EClassId, a: EClassId) {
        if let Some(v) = self.const_of(a) {
            let folded = self.add(ENode::Const(!v));
            self.union(cls, folded);
        }
        // Not(Not(x)) = x.
        if let Some(inner) = self.not_child(a) {
            self.union(cls, inner);
        }
        // De Morgan: Not(And(x,y)) = Or(Not x, Not y); Not(Or(x,y)) = And(Not x, Not y).
        for n in self.nodes_of(a) {
            match n {
                ENode::And(x, y) => {
                    let nx = self.add(ENode::Not(x));
                    let ny = self.add(ENode::Not(y));
                    let r = self.add(ENode::Or(nx, ny));
                    self.union(cls, r);
                }
                ENode::Or(x, y) => {
                    let nx = self.add(ENode::Not(x));
                    let ny = self.add(ENode::Not(y));
                    let r = self.add(ENode::And(nx, ny));
                    self.union(cls, r);
                }
                _ => {}
            }
        }
    }

    fn rules_and(&mut self, cls: EClassId, a: EClassId, b: EClassId) {
        if self.find_imm(a) == self.find_imm(b) {
            self.union(cls, a); // And(x,x) = x
            return;
        }
        if self.are_inverses(a, b) {
            let zero = self.add(ENode::Const(0));
            self.union(cls, zero); // And(x,!x) = 0
            return;
        }
        match (self.const_of(a), self.const_of(b)) {
            (Some(x), Some(y)) => {
                let r = self.add(ENode::Const(x & y));
                self.union(cls, r);
                return;
            }
            (Some(0), _) | (_, Some(0)) => {
                let z = self.add(ENode::Const(0));
                self.union(cls, z);
                return;
            }
            (Some(u64::MAX), _) => {
                self.union(cls, b);
                return;
            }
            (_, Some(u64::MAX)) => {
                self.union(cls, a);
                return;
            }
            _ => {}
        }
        if let Some(r) = self.fuse(a, b, |x, y| x & y, FUSED_AND_LIMIT) {
            self.union(cls, r);
        }
        // De Morgan reverse: And(Not x, Not y) = Not(Or(x,y)).
        if let (Some(x), Some(y)) = (self.not_child(a), self.not_child(b)) {
            let or = self.add(ENode::Or(x, y));
            let r = self.add(ENode::Not(or));
            self.union(cls, r);
        }
        if self.full {
            self.rules_ac(cls, a, b, AndOrXor::And);
            // Factor: And(Or(a,b), Or(a,c)) = Or(a, And(b,c)).
            self.rules_factor(cls, a, b, AndOrXor::And);
        }
    }

    fn rules_or(&mut self, cls: EClassId, a: EClassId, b: EClassId) {
        if self.find_imm(a) == self.find_imm(b) {
            self.union(cls, a); // Or(x,x) = x
            return;
        }
        if self.are_inverses(a, b) {
            let m = self.add(ENode::Const(u64::MAX));
            self.union(cls, m); // Or(x,!x) = MAX
            return;
        }
        match (self.const_of(a), self.const_of(b)) {
            (Some(x), Some(y)) => {
                let r = self.add(ENode::Const(x | y));
                self.union(cls, r);
                return;
            }
            (Some(0), _) => {
                self.union(cls, b);
                return;
            }
            (_, Some(0)) => {
                self.union(cls, a);
                return;
            }
            (Some(u64::MAX), _) | (_, Some(u64::MAX)) => {
                let m = self.add(ENode::Const(u64::MAX));
                self.union(cls, m);
                return;
            }
            _ => {}
        }
        if let Some(r) = self.fuse(a, b, |x, y| x | y, FUSED_OR_XOR_LIMIT) {
            self.union(cls, r);
        }
        // De Morgan reverse: Or(Not x, Not y) = Not(And(x,y)) — cancels Nots.
        if let (Some(x), Some(y)) = (self.not_child(a), self.not_child(b)) {
            let and = self.add(ENode::And(x, y));
            let r = self.add(ENode::Not(and));
            self.union(cls, r);
        }
        if self.full {
            self.rules_ac(cls, a, b, AndOrXor::Or);
            // Factor: Or(And(a,b), And(a,c)) = And(a, Or(b,c)).
            self.rules_factor(cls, a, b, AndOrXor::Or);
        }
    }

    fn rules_xor(&mut self, cls: EClassId, a: EClassId, b: EClassId) {
        if self.find_imm(a) == self.find_imm(b) {
            let z = self.add(ENode::Const(0));
            self.union(cls, z); // Xor(x,x) = 0
            return;
        }
        if self.are_inverses(a, b) {
            let m = self.add(ENode::Const(u64::MAX));
            self.union(cls, m); // Xor(x,!x) = MAX
            return;
        }
        match (self.const_of(a), self.const_of(b)) {
            (Some(x), Some(y)) => {
                let r = self.add(ENode::Const(x ^ y));
                self.union(cls, r);
                return;
            }
            (Some(0), _) => {
                self.union(cls, b);
                return;
            }
            (_, Some(0)) => {
                self.union(cls, a);
                return;
            }
            (Some(u64::MAX), _) => {
                let r = self.add(ENode::Not(b));
                self.union(cls, r); // Xor(MAX,x) = Not x
                return;
            }
            (_, Some(u64::MAX)) => {
                let r = self.add(ENode::Not(a));
                self.union(cls, r);
                return;
            }
            _ => {}
        }
        if let Some(r) = self.fuse(a, b, |x, y| x ^ y, FUSED_OR_XOR_LIMIT) {
            self.union(cls, r);
        }
        // Not-extraction: Xor(a, Not b) = Not(Xor(a,b)). Applying on either
        // side, the pulled-out Not cancels against an enclosing Not via the
        // Not(Not x) rule — this collapses the odd-backslash escape chain.
        if let Some(x) = self.not_child(a) {
            let inner = self.add(ENode::Xor(x, b));
            let r = self.add(ENode::Not(inner));
            self.union(cls, r);
        }
        if let Some(y) = self.not_child(b) {
            let inner = self.add(ENode::Xor(a, y));
            let r = self.add(ENode::Not(inner));
            self.union(cls, r);
        }
        if self.full {
            self.rules_ac(cls, a, b, AndOrXor::Xor);
        }
    }

    /// Associative-commutative normalization, the scalable replacement for
    /// pairwise associativity (which enumerates all parenthesizations of a
    /// same-op chain — Catalan blow-up). Flatten the chain into one operand
    /// multiset, combine everything that can combine in a single pass (fuse all
    /// `Class` operands, fold all `Const`s, drop duplicates / cancel inverse and
    /// Xor pairs), rebuild one canonical right-leaning tree, and union it in.
    /// O(k log k) in the chain length, and idempotent at the fixpoint (the
    /// canonical form re-flattens to itself), so saturation terminates.
    fn rules_ac(&mut self, cls: EClassId, a: EClassId, b: EClassId, op: AndOrXor) {
        // 1. Flatten the same-op chain. `seen` stops self-referential e-nodes
        // (e.g. the `Or(x,0)=x` identity merges `Or(x,0)` into x's class) from
        // looping, and bounds pathological chains.
        let mut operands: Vec<EClassId> = Vec::new();
        let mut stack = vec![a, b];
        let mut seen = std::collections::BTreeSet::new();
        while let Some(c) = stack.pop() {
            if operands.len() + stack.len() >= AC_FLATTEN_CAP {
                return;
            }
            let c = self.find_imm(c);
            if seen.insert(c)
                && let Some((x, y)) = self.smallest_same_op(c, op)
            {
                stack.push(x);
                stack.push(y);
                continue;
            }
            operands.push(c);
        }

        // 2a. Partition into folded const, fused class, and opaque operands.
        let mut const_acc: Option<u64> = None;
        let mut class_acc: Option<[u64; 4]> = None;
        let mut class_ops: Vec<EClassId> = Vec::new();
        let mut items: Vec<EClassId> = Vec::new();
        for &c in &operands {
            if let Some(v) = self.const_of(c) {
                const_acc = Some(const_acc.map_or(v, |p| op.apply(p, v)));
            } else if let Some(w) = self.class_words(c) {
                class_ops.push(c);
                class_acc = Some(match class_acc {
                    None => w,
                    Some(p) => [
                        op.apply(p[0], w[0]),
                        op.apply(p[1], w[1]),
                        op.apply(p[2], w[2]),
                        op.apply(p[3], w[3]),
                    ],
                });
            } else {
                items.push(c);
            }
        }

        // 2b. Realize the fused class (or fold an empty one into the const).
        if let Some(w) = class_acc {
            if w == [0; 4] {
                const_acc = Some(const_acc.map_or(0, |p| op.apply(p, 0)));
            } else if w.iter().map(|x| x.count_ones()).sum::<u32>() as usize <= op.limit() {
                let node = self.add(ENode::Class(w));
                items.push(node);
            } else {
                items.extend(class_ops); // too wide to stay compare-friendly
            }
        }

        // 2c. Const absorbers; a whole-expression collapse short-circuits.
        let mut negate = false;
        if let Some(v) = const_acc {
            match op {
                AndOrXor::And => {
                    if v == 0 {
                        let z = self.add(ENode::Const(0));
                        self.union(cls, z);
                        return;
                    } else if v != u64::MAX {
                        let n = self.add(ENode::Const(v));
                        items.push(n);
                    }
                }
                AndOrXor::Or => {
                    if v == u64::MAX {
                        let m = self.add(ENode::Const(u64::MAX));
                        self.union(cls, m);
                        return;
                    } else if v != 0 {
                        let n = self.add(ENode::Const(v));
                        items.push(n);
                    }
                }
                AndOrXor::Xor => {
                    if v == u64::MAX {
                        negate = !negate; // Xor(rest, all-ones) = Not(rest)
                    } else if v != 0 {
                        let n = self.add(ENode::Const(v));
                        items.push(n);
                    }
                }
            }
        }

        // 2d. Idempotence / cancellation / inverse pairs.
        items.sort();
        let reduced = match op {
            AndOrXor::And | AndOrXor::Or => {
                items.dedup();
                if self.has_inverse_pair(&items) {
                    let r = self.add(ENode::Const(op.identity() ^ u64::MAX));
                    self.union(cls, r); // And(x,!x)=0, Or(x,!x)=MAX
                    return;
                }
                items
            }
            AndOrXor::Xor => self.xor_reduce(items, &mut negate),
        };

        // 3. Rebuild one canonical tree and union it into the class.
        let result = self.build_ac(op, &reduced, negate);
        self.union(cls, result);
    }

    /// Operands of the lowest-keyed same-op e-node in a class, if any — a
    /// deterministic single representative to flatten through.
    fn smallest_same_op(&self, class: EClassId, op: AndOrXor) -> Option<(EClassId, EClassId)> {
        self.nodes_of(class)
            .iter()
            .filter_map(|n| op.match_node(n))
            .min()
    }

    /// Whether some operand's complement is also present (`x` and `!x`).
    fn has_inverse_pair(&self, items: &[EClassId]) -> bool {
        items
            .iter()
            .enumerate()
            .any(|(i, &x)| items[i + 1..].iter().any(|&y| self.are_inverses(x, y)))
    }

    /// Xor reduction: keep operands of odd multiplicity (`x ^ x = 0`), then
    /// cancel each inverse pair (`x ^ !x = all-ones`) into a `negate` toggle.
    fn xor_reduce(&self, items: Vec<EClassId>, negate: &mut bool) -> Vec<EClassId> {
        let mut odd: Vec<EClassId> = Vec::new();
        let mut i = 0;
        while i < items.len() {
            let mut j = i;
            while j < items.len() && items[j] == items[i] {
                j += 1;
            }
            if (j - i) % 2 == 1 {
                odd.push(items[i]);
            }
            i = j;
        }
        let mut removed = vec![false; odd.len()];
        for a in 0..odd.len() {
            if removed[a] {
                continue;
            }
            for b in (a + 1)..odd.len() {
                if !removed[b] && self.are_inverses(odd[a], odd[b]) {
                    removed[a] = true;
                    removed[b] = true;
                    *negate = !*negate;
                    break;
                }
            }
        }
        odd.into_iter()
            .zip(removed)
            .filter_map(|(c, r)| (!r).then_some(c))
            .collect()
    }

    /// Build a canonical right-leaning `op` tree over the (sorted) operands,
    /// optionally negated. Empty → the op's identity element; one operand → that
    /// operand. `op.make` re-canonicalizes, so the tree is hashcons-stable.
    fn build_ac(&mut self, op: AndOrXor, reduced: &[EClassId], negate: bool) -> EClassId {
        let base = match reduced.split_last() {
            None => self.add(ENode::Const(op.identity())),
            Some((&last, rest)) => {
                let mut acc = last;
                for &c in rest.iter().rev() {
                    acc = self.add(op.make(c, acc));
                }
                acc
            }
        };
        if negate {
            self.add(ENode::Not(base))
        } else {
            base
        }
    }

    /// Factor a shared operand out of two sibling sub-ops:
    /// `outer(inner(f,p), inner(f,q)) = inner(f, outer(p,q))`, where for an
    /// `Or` outer the inner is `And` (and vice-versa). This is the sharing win
    /// the global two-candidate optimizer cannot express.
    fn rules_factor(&mut self, cls: EClassId, a: EClassId, b: EClassId, outer: AndOrXor) {
        let inner = match outer {
            AndOrXor::Or => AndOrXor::And,
            AndOrXor::And => AndOrXor::Or,
            AndOrXor::Xor => return,
        };
        let an: Vec<(EClassId, EClassId)> = self
            .nodes_of(a)
            .iter()
            .filter_map(|n| inner.match_node(n))
            .collect();
        let bn: Vec<(EClassId, EClassId)> = self
            .nodes_of(b)
            .iter()
            .filter_map(|n| inner.match_node(n))
            .collect();
        for &(ax, ay) in &an {
            for &(bx, by) in &bn {
                // Operands are sorted, so the shared factor can sit on either
                // side of each inner node — check the matching pairings.
                let shared = if ax == bx {
                    Some((ax, ay, by))
                } else if ax == by {
                    Some((ax, ay, bx))
                } else if ay == bx {
                    Some((ay, ax, by))
                } else if ay == by {
                    Some((ay, ax, bx))
                } else {
                    None
                };
                if let Some((f, p, q)) = shared {
                    let combined = self.add(outer.make(p, q));
                    let r = self.add(inner.make(f, combined));
                    self.union(cls, r);
                }
            }
        }
    }

    /// Fold a bitwise op over two `Class` operands into one class covering the
    /// combined membership, when the result stays within `limit` members.
    /// Empty membership is the all-zero stream = `Const(0)`.
    fn fuse(
        &mut self,
        a: EClassId,
        b: EClassId,
        op: impl Fn(u64, u64) -> u64,
        limit: usize,
    ) -> Option<EClassId> {
        let (wa, wb) = (self.class_words(a)?, self.class_words(b)?);
        let fused = [
            op(wa[0], wb[0]),
            op(wa[1], wb[1]),
            op(wa[2], wb[2]),
            op(wa[3], wb[3]),
        ];
        if fused == [0; 4] {
            return Some(self.add(ENode::Const(0)));
        }
        let popcount: u32 = fused.iter().map(|w| w.count_ones()).sum();
        if popcount as usize > limit {
            return None;
        }
        Some(self.add(ENode::Class(fused)))
    }

    /// Extract the cheapest program for each root. Cost is minimized to a
    /// fixpoint (monotone decreasing → terminates), with a total tie-break of
    /// (cost, node-count, canonical e-node) so the result is deterministic.
    /// Self-referential e-nodes (created by identity rules like `Or(x,0)=x`)
    /// never win: positive weights plus an always-present acyclic alternative
    /// make their derived cost strictly larger.
    fn extract(&self, roots: &[EClassId], model: &CostModel) -> (Graph, Vec<NodeId>) {
        let n = self.classes.len();
        let mut cost = vec![u64::MAX; n];
        let mut ncount = vec![u64::MAX; n];
        let mut best: Vec<Option<ENode>> = vec![None; n];
        loop {
            let mut changed = false;
            for c in 0..n {
                let cid = EClassId(c as u32);
                if self.find_imm(cid) != cid {
                    continue;
                }
                for node in &self.classes[c] {
                    let mut tot = weight(node, model);
                    let mut cnt = 1u64;
                    let mut ready = true;
                    for ch in children(node) {
                        let ci = self.find_imm(ch).0 as usize;
                        if cost[ci] == u64::MAX {
                            ready = false;
                            break;
                        }
                        tot = tot.saturating_add(cost[ci]);
                        cnt = cnt.saturating_add(ncount[ci]);
                    }
                    if !ready {
                        continue;
                    }
                    let better = tot < cost[c]
                        || (tot == cost[c] && cnt < ncount[c])
                        || (tot == cost[c]
                            && cnt == ncount[c]
                            && best[c].as_ref().is_none_or(|b| node < b));
                    if better {
                        cost[c] = tot;
                        ncount[c] = cnt;
                        best[c] = Some(node.clone());
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Depth of each chosen subtree, for Sethi–Ullman emit ordering below.
        let mut depth = vec![0u32; n];
        loop {
            let mut changed = false;
            for c in 0..n {
                let cid = EClassId(c as u32);
                if self.find_imm(cid) != cid {
                    continue;
                }
                if let Some(node) = &best[c] {
                    let d = children(node)
                        .iter()
                        .map(|ch| depth[self.find_imm(*ch).0 as usize])
                        .max()
                        .map_or(0, |m| m + 1);
                    if d != depth[c] {
                        depth[c] = d;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let mut builder = Builder {
            eg: self,
            best: &best,
            depth: &depth,
            graph: Graph::new(),
            memo: vec![None; n],
            building: vec![false; n],
        };
        let mapped = roots.iter().map(|&r| builder.emit(r)).collect();
        (builder.graph, mapped)
    }
}

/// The three commutative boolean ops, shared by the associativity and
/// factoring rules.
#[derive(Clone, Copy)]
enum AndOrXor {
    And,
    Or,
    Xor,
}

impl AndOrXor {
    fn make(self, a: EClassId, b: EClassId) -> ENode {
        match self {
            AndOrXor::And => ENode::And(a, b),
            AndOrXor::Or => ENode::Or(a, b),
            AndOrXor::Xor => ENode::Xor(a, b),
        }
    }

    fn apply(self, x: u64, y: u64) -> u64 {
        match self {
            AndOrXor::And => x & y,
            AndOrXor::Or => x | y,
            AndOrXor::Xor => x ^ y,
        }
    }

    /// Identity element: `op(x, identity) == x` (And→all-ones, Or/Xor→0).
    fn identity(self) -> u64 {
        match self {
            AndOrXor::And => u64::MAX,
            AndOrXor::Or | AndOrXor::Xor => 0,
        }
    }

    /// Largest fused-class membership that stays codegen-friendly. Intersection
    /// (`And`) can only shrink, so it always fuses; `Or`/`Xor` stop at 8.
    fn limit(self) -> usize {
        match self {
            AndOrXor::And => FUSED_AND_LIMIT,
            AndOrXor::Or | AndOrXor::Xor => FUSED_OR_XOR_LIMIT,
        }
    }

    fn match_node(self, node: &ENode) -> Option<(EClassId, EClassId)> {
        match (self, node) {
            (AndOrXor::And, &ENode::And(a, b))
            | (AndOrXor::Or, &ENode::Or(a, b))
            | (AndOrXor::Xor, &ENode::Xor(a, b)) => Some((a, b)),
            _ => None,
        }
    }
}

/// Reconstructs a [`Graph`] from the chosen e-node per class, emitting operands
/// before consumers so the result stays topologically ordered.
struct Builder<'a> {
    eg: &'a EGraph,
    best: &'a [Option<ENode>],
    depth: &'a [u32],
    graph: Graph,
    memo: Vec<Option<NodeId>>,
    building: Vec<bool>,
}

impl Builder<'_> {
    fn depth_of(&self, class: EClassId) -> u32 {
        self.depth[self.eg.find_imm(class).0 as usize]
    }

    /// Emit two operands deeper-subtree-first (Sethi–Ullman): the shallow
    /// operand lands adjacent to its consumer instead of being hoisted to the
    /// top of the block, where it would stay live across the whole computation
    /// and raise register pressure in the emitted kernel loop. Matches the
    /// ordering `graph_opt`'s rebuild uses; without it, equal-cost extractions
    /// schedule measurably worse. Returns operands in (a, b) order.
    fn emit_pair(&mut self, a: EClassId, b: EClassId) -> (NodeId, NodeId) {
        if self.depth_of(a) >= self.depth_of(b) {
            let ea = self.emit(a);
            let eb = self.emit(b);
            (ea, eb)
        } else {
            let eb = self.emit(b);
            let ea = self.emit(a);
            (ea, eb)
        }
    }

    fn emit(&mut self, class: EClassId) -> NodeId {
        let c = self.eg.find_imm(class).0 as usize;
        if let Some(id) = self.memo[c] {
            return id;
        }
        debug_assert!(!self.building[c], "cycle in extracted graph");
        self.building[c] = true;
        let node = self.best[c].clone().expect("reachable class has a best node");
        let id = match node {
            ENode::Class(w) => self.graph.class(CharClass::from_words(w)),
            ENode::Const(v) => self.graph.constant(v),
            ENode::Not(a) => {
                let a = self.emit(a);
                self.graph.not(a)
            }
            ENode::And(a, b) => {
                let (a, b) = self.emit_pair(a, b);
                self.graph.and(a, b)
            }
            ENode::Or(a, b) => {
                let (a, b) = self.emit_pair(a, b);
                self.graph.or(a, b)
            }
            ENode::Xor(a, b) => {
                let (a, b) = self.emit_pair(a, b);
                self.graph.xor(a, b)
            }
            ENode::ShiftLeft1(a) => {
                let a = self.emit(a);
                self.graph.shift_left1(a)
            }
            ENode::ShiftLeft1Seeded(a) => {
                let a = self.emit(a);
                self.graph.shift_left1_seeded(a)
            }
            ENode::PrefixXor(a) => {
                let a = self.emit(a);
                self.graph.prefix_xor(a)
            }
            ENode::Add(a, b) => {
                let (a, b) = self.emit_pair(a, b);
                self.graph.add(a, b)
            }
            ENode::Regions(q, s, t) => {
                // Visit deepest-first for register pressure, then emit in role
                // order (memoized, so position is fixed by the first visit).
                let mut order = [q, s, t];
                order.sort_by_key(|c| std::cmp::Reverse(self.depth_of(*c)));
                for c in order {
                    self.emit(c);
                }
                let q = self.emit(q);
                let s = self.emit(s);
                let t = self.emit(t);
                self.graph.regions(q, s, t)
            }
        };
        self.building[c] = false;
        self.memo[c] = Some(id);
        id
    }
}

fn ord(a: EClassId, b: EClassId) -> (EClassId, EClassId) {
    if a.0 <= b.0 { (a, b) } else { (b, a) }
}

/// Translate a source [`Op`] into the e-node over already-interned operand
/// classes (the source graph is topologically ordered, so operands precede).
fn op_to_enode(op: &Op, map: &[EClassId]) -> ENode {
    let m = |id: NodeId| map[id.0 as usize];
    match *op {
        Op::Class(c) => ENode::Class(c.words()),
        Op::Const(p) => ENode::Const(p),
        Op::Not(a) => ENode::Not(m(a)),
        Op::And(a, b) => ENode::And(m(a), m(b)),
        Op::Or(a, b) => ENode::Or(m(a), m(b)),
        Op::Xor(a, b) => ENode::Xor(m(a), m(b)),
        Op::ShiftLeft1(a) => ENode::ShiftLeft1(m(a)),
        Op::ShiftLeft1Seeded(a) => ENode::ShiftLeft1Seeded(m(a)),
        Op::PrefixXor(a) => ENode::PrefixXor(m(a)),
        Op::Add(a, b) => ENode::Add(m(a), m(b)),
        Op::Regions(q, s, t) => ENode::Regions(m(q), m(s), m(t)),
    }
}

/// Optimize a dialect's [`DelimitedParts`] by equality saturation, preserving
/// the role of every output node (structural output, terminators, optional
/// bracket pair). Mirrors [`crate::graph_opt::optimize_parts`]: the original
/// form of each root is always extractable, so the result never costs more
/// than the input, and it is adopted only when strictly cheaper.
pub fn optimize_parts(parts: DelimitedParts, model: CostModel) -> OptimizedParts {
    let mut root_nodes = vec![parts.graph.output(), parts.terminators];
    if let Some((opens, closes)) = parts.nest {
        root_nodes.extend([opens, closes]);
    }

    let original_nodes = parts.graph.nodes().len();
    let original_cost = graph_cost(&parts.graph, &model);

    let mut eg = EGraph::new();
    let mut map = vec![EClassId(0); parts.graph.nodes().len()];
    for (i, op) in parts.graph.nodes().iter().enumerate() {
        map[i] = eg.add(op_to_enode(op, &map));
    }
    let root_classes: Vec<EClassId> = root_nodes.iter().map(|r| map[r.0 as usize]).collect();

    eg.saturate();
    let (mut graph, mapped) = eg.extract(&root_classes, &model);

    let optimized_cost = graph_cost(&graph, &model);
    let optimized_nodes = graph.nodes().len();
    // Cost is the objective. The original form of every root is always
    // extractable, so `optimized_cost <= original_cost` holds and the e-graph
    // never regresses; adopt whenever it is strictly cheaper. (We do not gate on
    // node count — a cheaper graph with more low-cost nodes is still a win under
    // the model; trusting that is the cost model's job, calibrated separately.)
    let applied = optimized_cost < original_cost;

    if !applied {
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

    graph.set_output(mapped[0]);
    let nest = parts.nest.map(|_| (mapped[2], mapped[3]));
    OptimizedParts {
        parts: DelimitedParts {
            graph,
            terminators: mapped[1],
            nest,
        },
        stats: OptimizationStats {
            original_nodes,
            optimized_nodes,
            removed_nodes: original_nodes.saturating_sub(optimized_nodes),
            original_cost,
            optimized_cost,
            applied: true,
        },
    }
}
