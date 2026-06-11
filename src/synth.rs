//! Bottom-up enumerative synthesis over the bitstream IR.
//!
//! The ops in [`crate::ir`] encode bit-parallel tricks that humans found by
//! hand: quote context as prefix-XOR, odd-run escape detection as carry
//! ripple. This module inverts the direction. Given a byte-at-a-time
//! reference semantics — the state machine a person would naively write —
//! it *searches* the IR algebra for an equivalent branchless graph, so the
//! generator can discover kernels instead of compiling known ones.
//!
//! Method: classic bottom-up enumeration with observational-equivalence
//! pruning. Terms are enumerated in order of expression size; a term enters
//! the bank only if its behavior on a discriminating multi-block corpus is
//! new (`Not(Not(x))` dies on arrival). A term whose corpus behavior equals
//! the target's is reconstructed as a real [`Graph`] (with common
//! subexpressions shared) and differentially verified against the reference
//! on thousands of fresh random inputs via [`interp::run`]; a verification
//! failure becomes a new corpus input and the search restarts (CEGIS).
//!
//! Solved problems can be fed back as [`LeafKind::Derived`] leaves — the
//! library-learning hook that lets small discoveries compose into ones that
//! plain enumeration could never reach.
//!
//! [`crate::ir::Op::Regions`] is deliberately excluded from the search
//! space: it is the IR's sequential escape hatch, and the point here is to
//! find (or demonstrate the absence of) *bit-parallel* solutions.

use crate::interp;
use crate::ir::{CharClass, Graph, NodeId, Op};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

/// A starting stream the synthesizer may use as an expression leaf.
#[derive(Clone)]
pub struct Leaf {
    pub name: String,
    pub kind: LeafKind,
}

#[derive(Clone)]
pub enum LeafKind {
    /// Byte-class membership stream.
    Class(CharClass),
    /// The same 64-bit pattern in every block (e.g. the even-position mask).
    Const(u64),
    /// A previously synthesized (or hand-built) subgraph, spliced in by
    /// value when a solution uses it.
    Derived(Graph),
}

impl Leaf {
    pub fn class(name: &str, bytes: &[u8]) -> Self {
        Self { name: name.into(), kind: LeafKind::Class(CharClass::from_bytes(bytes)) }
    }

    pub fn constant(name: &str, pattern: u64) -> Self {
        Self { name: name.into(), kind: LeafKind::Const(pattern) }
    }

    pub fn derived(name: &str, graph: Graph) -> Self {
        Self { name: name.into(), kind: LeafKind::Derived(graph) }
    }
}

/// A stream semantics: input bytes in, one mask word per 64-byte block out.
pub type MaskFn<'a> = &'a dyn Fn(&[u8]) -> Vec<u64>;

/// What the synthesizer must match: ground-truth semantics plus an
/// optional relevance mask.
pub struct Spec<'a> {
    /// Per-block output masks for the given input (the serial machine).
    pub reference: MaskFn<'a>,
    /// Per-block masks of the positions where the output MATTERS. Bits
    /// outside the mask are unconstrained — e.g. an escaped-positions
    /// stream consumed only as `quotes & !escaped` need only be right at
    /// quote bytes. Don't-cares admit strictly smaller circuits. `None`
    /// requires exact equality everywhere.
    pub care: Option<MaskFn<'a>>,
}

impl<'a> Spec<'a> {
    pub fn exact(reference: MaskFn<'a>) -> Self {
        Self { reference, care: None }
    }

    pub fn with_care(reference: MaskFn<'a>, care: MaskFn<'a>) -> Self {
        Self { reference, care: Some(care) }
    }
}

/// Per-op cost weights for ranking solutions. Node count is NOT a cost
/// model: a `PrefixXor` is a carry-less multiply on AVX2 and a six-shift
/// cascade in scalar code, while the ops it might replace are single-cycle
/// — a lesson measured directly in this repo (see `escaped_positions`).
#[derive(Clone, Copy)]
pub struct CostModel {
    pub class: u32,
    pub constant: u32,
    pub bitwise: u32,
    pub shift: u32,
    pub add: u32,
    pub prefix_xor: u32,
    pub regions: u32,
}

impl CostModel {
    /// The dispatched kernel: classes are a couple of compares, constants
    /// are free registers, PrefixXor is PCLMULQDQ plus GPR/XMM round trips.
    pub fn avx2() -> Self {
        Self { class: 2, constant: 0, bitwise: 1, shift: 2, add: 2, prefix_xor: 10, regions: 50 }
    }

    /// The portable fallback: classes walk 64 bytes, PrefixXor is the
    /// log-step XOR cascade.
    pub fn scalar() -> Self {
        Self { class: 20, constant: 0, bitwise: 1, shift: 2, add: 2, prefix_xor: 12, regions: 50 }
    }

    /// Uniform weights: plain DAG node count.
    pub fn nodes() -> Self {
        Self { class: 1, constant: 1, bitwise: 1, shift: 1, add: 1, prefix_xor: 1, regions: 1 }
    }
}

/// Weighted cost of a graph under a model (leaves included).
pub fn graph_cost(graph: &Graph, model: &CostModel) -> u32 {
    graph
        .nodes()
        .iter()
        .map(|op| match op {
            Op::Class(_) => model.class,
            Op::Const(_) => model.constant,
            Op::Not(_) | Op::And(..) | Op::Or(..) | Op::Xor(..) => model.bitwise,
            Op::ShiftLeft1(_) | Op::ShiftLeft1Seeded(_) => model.shift,
            Op::Add(..) => model.add,
            Op::PrefixXor(_) => model.prefix_xor,
            Op::Regions(..) => model.regions,
        })
        .sum()
}

/// What "level" means during enumeration.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Levels are expression-tree node counts: classic size-ordered
    /// bottom-up enumeration.
    TreeSize,
    /// Levels are expression-tree COSTS under the budget's cost model:
    /// Dijkstra-style enumeration. Cheap forms surface first and expensive
    /// subtrees (PrefixXor-heavy) are implicitly deprioritized, so the
    /// first verified match is already minimal in tree cost (graph cost
    /// can still improve through sharing — keep a small settle window).
    Cost,
}

/// Search limits. Enumeration is exhaustive in levels (tree size or tree
/// cost, per `order`) until a limit trips, so a [`Outcome::NotFound`]
/// reports exactly how far the frontier got.
pub struct Budget {
    /// Largest level to enumerate: tree node count for
    /// [`Order::TreeSize`], tree cost for [`Order::Cost`].
    pub max_level: usize,
    /// Total candidate evaluations across the whole search.
    pub max_candidates: u64,
    /// Cap on behaviorally-distinct terms kept in the bank.
    pub max_bank: usize,
    /// After the first verified match, keep enumerating this many further
    /// levels collecting cheaper equivalents (0 = finish the current
    /// level only).
    pub settle_levels: usize,
    /// Cost model that ranks equivalent solutions (and defines levels
    /// under [`Order::Cost`]).
    pub cost: CostModel,
    /// Enumeration order.
    pub order: Order,
    /// Print per-level search statistics to stderr.
    pub progress: bool,
}

fn un_weight(op: UnOp, order: Order, model: &CostModel) -> usize {
    match order {
        Order::TreeSize => 1,
        Order::Cost => match op {
            UnOp::Not => model.bitwise as usize,
            UnOp::Shl | UnOp::ShlSeed => model.shift as usize,
            UnOp::PXor => model.prefix_xor as usize,
        },
    }
}

fn bin_weight(op: BinOp, order: Order, model: &CostModel) -> usize {
    match order {
        Order::TreeSize => 1,
        Order::Cost => match op {
            BinOp::And | BinOp::Or | BinOp::Xor => model.bitwise as usize,
            BinOp::Add => model.add as usize,
        },
    }
}

fn leaf_level(leaf: &Leaf, order: Order, model: &CostModel) -> usize {
    match order {
        Order::TreeSize => 1,
        Order::Cost => match &leaf.kind {
            LeafKind::Class(_) => model.class as usize,
            LeafKind::Const(_) => model.constant as usize,
            LeafKind::Derived(graph) => graph_cost(graph, model) as usize,
        },
    }
}

/// Tree cost of a template body with the hole free (the weight its
/// application adds on top of the child's level).
fn ttree_weight(body: &TTree, leaves: &[Leaf], order: Order, model: &CostModel) -> usize {
    match order {
        Order::TreeSize => 1,
        Order::Cost => match body {
            TTree::Hole => 0,
            TTree::Leaf(li) => leaf_level(&leaves[*li as usize], order, model),
            TTree::Un(op, a) => {
                un_weight(*op, order, model) + ttree_weight(a, leaves, order, model)
            }
            TTree::Bin(op, a, b) => {
                bin_weight(*op, order, model)
                    + ttree_weight(a, leaves, order, model)
                    + ttree_weight(b, leaves, order, model)
            }
        },
    }
}

#[derive(Debug)]
pub struct Stats {
    /// Candidate terms evaluated (including behavioral duplicates).
    pub candidates: u64,
    /// Behaviorally distinct terms banked.
    pub bank_unique: usize,
    /// Largest level (tree size or tree cost, per the budget's order)
    /// whose enumeration ran to completion.
    pub completed_level: usize,
    /// CEGIS restarts (corpus extensions from verification failures).
    pub restarts: usize,
    /// The bank hit its cap: enumeration continued over the frozen bank
    /// (matching still works) but sizes past that point are not complete.
    pub bank_saturated: bool,
    pub elapsed_ms: u128,
}

pub struct Solution {
    /// Human-readable expression over the leaf names.
    pub expr: String,
    /// The discovered program as a real IR graph, output set, CSE applied.
    pub graph: Graph,
    /// Node count of `graph` (leaves included).
    pub dag_nodes: usize,
    /// Weighted cost under the budget's [`CostModel`] — what the search
    /// actually minimizes among equivalents.
    pub cost: u32,
    /// Expression tree size at which the solution was found.
    pub tree_size: usize,
    /// Fresh random inputs the solution was differentially verified on.
    pub verified_inputs: usize,
    /// Every distinct corpus-matching form seen during the search as
    /// `(cost, expr)`, cheapest first, capped at 8. Only the returned
    /// solution itself is fully verified; alternates matched the corpus.
    pub alternates: Vec<(u32, String)>,
    pub stats: Stats,
}

pub enum Outcome {
    Found(Box<Solution>),
    NotFound(Stats),
}

/// Search for an IR graph whose output stream matches the spec on every
/// input. The spec's `reference` receives raw input bytes and returns one
/// mask word per 64-byte block (final partial block zero-padded); corpus
/// inputs must be multiples of 64 bytes so block masks need no pad
/// handling. Among equivalent solutions, the cheapest under the budget's
/// cost model wins.
pub fn synthesize(
    leaves: &[Leaf],
    corpus: &[Vec<u8>],
    spec: &Spec,
    budget: &Budget,
) -> Outcome {
    let start = Instant::now();
    let mut inputs = corpus.to_vec();
    let mut candidates = 0u64;
    let mut restarts = 0usize;
    loop {
        match attempt(leaves, &[], &inputs, spec, budget, &mut candidates, restarts, start, false) {
            Attempt::Found(solution) => return Outcome::Found(solution),
            Attempt::Exhausted(stats, _) => return Outcome::NotFound(stats),
            Attempt::Counterexample(input) => {
                // The corpus failed to discriminate a wrong candidate from
                // the target; fold the witness in and restart. Each restart
                // strictly strengthens the corpus, so this terminates.
                restarts += 1;
                assert!(restarts <= 16, "CEGIS failed to converge: corpus too weak");
                inputs.push(input);
            }
        }
    }
}

pub struct MultiSolution {
    /// One expression per spec, in spec order. Later outputs may name
    /// earlier ones as `O0`, `O1`, ... — cross-output reuse is the point.
    pub exprs: Vec<String>,
    /// All outputs merged into ONE graph with common subexpressions
    /// shared; `outputs[k]` is spec k's stream. `graph.output()` is set to
    /// the first output.
    pub graph: Graph,
    pub outputs: Vec<NodeId>,
    /// Cost of the merged graph — what a fused kernel would pay.
    pub shared_cost: u32,
    /// Sum of the per-spec graph costs — what separate kernels would pay.
    pub separate_cost: u32,
    pub stats: Stats,
}

pub enum MultiOutcome {
    Found(Box<MultiSolution>),
    NotFound { failed_spec: usize, stats: Stats },
}

/// One output of a multi-output synthesis: its spec plus the leaves it
/// declares as inputs — the way a real kernel output names its byte
/// classes. Earlier outputs join every later library automatically.
pub struct MultiSpec<'a> {
    pub leaves: &'a [Leaf],
    pub spec: Spec<'a>,
}

/// Synthesize several output streams against one corpus, sharing work
/// across them: specs are solved in order, and every solved stream joins
/// the leaf library for the streams after it (named `O0`, `O1`, ...), so
/// later outputs can be expressed in terms of earlier ones at size 1 —
/// which is how real kernels are built (`step_nested` returns three masks
/// from one shared DAG). Each spec searches over ITS declared leaves plus
/// the solved streams, keeping every search small. The merged graph
/// shares all common subexpressions; `shared_cost` vs `separate_cost` is
/// the fusion win.
pub fn synthesize_multi(
    corpus: &[Vec<u8>],
    specs: &[MultiSpec],
    budget: &Budget,
) -> MultiOutcome {
    let mut derived: Vec<Leaf> = Vec::new();
    let mut solutions: Vec<Solution> = Vec::new();
    for (k, multi_spec) in specs.iter().enumerate() {
        let mut library: Vec<Leaf> = multi_spec.leaves.to_vec();
        library.extend(derived.iter().cloned());
        match synthesize(&library, corpus, &multi_spec.spec, budget) {
            Outcome::Found(solution) => {
                derived.push(Leaf::derived(&format!("O{k}"), solution.graph.clone()));
                solutions.push(*solution);
            }
            Outcome::NotFound(stats) => {
                return MultiOutcome::NotFound { failed_spec: k, stats };
            }
        }
    }
    let mut graph = Graph::new();
    let mut cse: HashMap<CseKey, NodeId> = HashMap::new();
    let outputs: Vec<NodeId> =
        solutions.iter().map(|sol| splice(&mut graph, &mut cse, &sol.graph)).collect();
    graph.set_output(outputs[0]);
    let shared_cost = graph_cost(&graph, &budget.cost);
    let separate_cost = solutions.iter().map(|sol| sol.cost).sum();
    let stats = Stats {
        candidates: solutions.iter().map(|sol| sol.stats.candidates).sum(),
        bank_unique: solutions.iter().map(|sol| sol.stats.bank_unique).max().unwrap_or(0),
        completed_level: solutions.iter().map(|sol| sol.stats.completed_level).min().unwrap_or(0),
        restarts: solutions.iter().map(|sol| sol.stats.restarts).sum(),
        bank_saturated: solutions.iter().any(|sol| sol.stats.bank_saturated),
        elapsed_ms: solutions.iter().map(|sol| sol.stats.elapsed_ms).sum(),
    };
    MultiOutcome::Found(Box::new(MultiSolution {
        exprs: solutions.into_iter().map(|sol| sol.expr).collect(),
        graph,
        outputs,
        shared_cost,
        separate_cost,
        stats,
    }))
}

/// Limits for [`synthesize_auto`]: rounds of bounded enumeration, each
/// followed by promotion of the highest-scoring banked terms to new leaves.
pub struct AutoBudget {
    pub rounds: usize,
    /// Inner search budget, applied afresh each round.
    pub per_round: Budget,
    /// Terms promoted to leaves after each exhausted round.
    pub promotions: usize,
    /// Hard cap on the leaf library (base leaves included).
    pub max_leaves: usize,
}

#[derive(Debug)]
pub struct RoundReport {
    pub round: usize,
    pub stats: Stats,
    /// Expressions promoted to leaves after this round.
    pub promoted: Vec<String>,
}

pub enum AutoOutcome {
    Found(Box<Solution>, Vec<RoundReport>),
    NotFound(Vec<RoundReport>),
}

/// [`synthesize`], but the system invents its own abstractions: when a
/// round of enumeration exhausts its budget, banked terms are scored by
/// three target-agnostic-of-domain signals — *gate* (precision x recall of
/// the term's bits against the target's), *generativity* (how many
/// behaviorally novel terms used it as a direct child), and *near-miss
/// harvest* (subterm frequency among the candidates Hamming-closest to the
/// target) — and the best become size-1 leaves for the next round. Library
/// learning, with the library chosen by search instead of by a human.
pub fn synthesize_auto(
    base_leaves: &[Leaf],
    corpus: &[Vec<u8>],
    spec: &Spec,
    auto: &AutoBudget,
) -> AutoOutcome {
    let start = Instant::now();
    let mut leaves: Vec<Leaf> = base_leaves.to_vec();
    let mut templates: Vec<Template> = Vec::new();
    let mut inputs = corpus.to_vec();
    let mut reports = Vec::new();
    for round in 1..=auto.rounds {
        let mut candidates = 0u64;
        let mut restarts = 0usize;
        let (stats, dump) = loop {
            match attempt(
                &leaves, &templates, &inputs, spec, &auto.per_round, &mut candidates, restarts,
                start, true,
            ) {
                Attempt::Found(solution) => return AutoOutcome::Found(solution, reports),
                Attempt::Exhausted(stats, dump) => {
                    break (stats, dump.expect("dump requested"));
                }
                Attempt::Counterexample(input) => {
                    restarts += 1;
                    assert!(restarts <= 16, "CEGIS failed to converge: corpus too weak");
                    inputs.push(input);
                }
            }
        };
        let room = auto.max_leaves.saturating_sub(leaves.len()).min(auto.promotions);
        let (picked_leaves, picked_templates) =
            promote(&leaves, &templates, &dump, &Corpus::new(&inputs), room);
        let mut promoted: Vec<String> =
            picked_leaves.iter().map(|leaf| leaf.name.clone()).collect();
        promoted
            .extend(picked_templates.iter().map(|tpl| format!("template {}", tpl.name)));
        reports.push(RoundReport { round, stats, promoted });
        if picked_leaves.is_empty() && picked_templates.is_empty() {
            break; // No new vocabulary: further rounds would repeat this one.
        }
        leaves.extend(picked_leaves);
        templates.extend(picked_templates);
    }
    AutoOutcome::NotFound(reports)
}

/// Score the bank of an exhausted round and return the terms worth
/// promoting to leaves, deduplicated behaviorally against the existing
/// library. Only terms containing at least one stateful op are eligible:
/// they carry information across positions, which is what raw enumeration
/// cannot cheaply rebuild and what makes an abstraction extend reach.
fn promote(
    existing: &[Leaf],
    existing_templates: &[Template],
    dump: &Dump,
    corpus: &Corpus,
    room: usize,
) -> (Vec<Leaf>, Vec<Template>) {
    let bank = &dump.bank;
    if room == 0 || bank.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Score against the care-masked target: don't-care bits must not drive
    // promotion any more than they drive matching.
    let target: Vec<u64> = match &dump.care {
        None => dump.target.clone(),
        Some(care) => dump.target.iter().zip(care).map(|(&t, &c)| t & c).collect(),
    };
    let care_word = |block: usize| dump.care.as_ref().map_or(!0u64, |care| care[block]);
    let eval_of = |i: usize| -> &[u64] { &dump.evals[i * dump.eval_len..][..dump.eval_len] };
    let target_ones: u64 = target.iter().map(|w| w.count_ones() as u64).sum();
    let total_bits = (corpus.total_blocks * 64) as u64;

    // One pass: direct-child counts (generativity) and stateful closure.
    let mut child_count = vec![0u32; bank.len()];
    let mut stateful = vec![false; bank.len()];
    for (i, entry) in bank.iter().enumerate() {
        match entry.term {
            Term::Leaf(_) => {}
            Term::Un(op, a) => {
                child_count[a as usize] += 1;
                stateful[i] = !matches!(op, UnOp::Not) || stateful[a as usize];
            }
            Term::Bin(op, a, b) => {
                child_count[a as usize] += 1;
                child_count[b as usize] += 1;
                stateful[i] =
                    matches!(op, BinOp::Add) || stateful[a as usize] || stateful[b as usize];
            }
            Term::Tpl(t, a) => {
                child_count[a as usize] += 1;
                stateful[i] =
                    existing_templates[t as usize].stateful || stateful[a as usize];
            }
        }
    }

    // Near-miss harvest: subterm frequency among the 64 candidates closest
    // to the target in Hamming distance.
    let mut by_distance: Vec<(u64, u32)> = (0..bank.len())
        .map(|i| {
            let d: u64 = eval_of(i)
                .iter()
                .zip(&target)
                .enumerate()
                .map(|(block, (&e, &t))| ((e ^ t) & care_word(block)).count_ones() as u64)
                .sum();
            (d, i as u32)
        })
        .collect();
    let cut = by_distance.len().min(64);
    by_distance.select_nth_unstable(cut - 1);
    by_distance[..cut].sort_unstable();
    let mut miss_count = vec![0u32; bank.len()];
    for &(_, idx) in &by_distance[..cut] {
        let mut stack = vec![idx];
        while let Some(i) = stack.pop() {
            miss_count[i as usize] += 1;
            match bank[i as usize].term {
                Term::Leaf(_) => {}
                Term::Un(_, a) | Term::Tpl(_, a) => stack.push(a),
                Term::Bin(_, a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
            }
        }
    }

    // Eligibility: small reusable fragments only. Abstractions are
    // vocabulary, not solutions — a deep composite that nearly matches the
    // target is overfit glue and crowds the next round's search, so the
    // size cap is load-bearing (an MDL-style prior, DreamCoder-fashion).
    const MAX_FRAGMENT: u16 = 6;
    let eligible: Vec<u32> = bank
        .iter()
        .enumerate()
        .filter_map(|(i, entry)| {
            if entry.size < 2 || entry.size > MAX_FRAGMENT || !stateful[i] {
                return None;
            }
            let ones: u64 = eval_of(i).iter().map(|w| w.count_ones() as u64).sum();
            (ones != 0 && ones != total_bits).then_some(i as u32)
        })
        .collect();

    let gate = |i: u32| -> f64 {
        if target_ones == 0 {
            return 0.0;
        }
        // Precision counts only bits the spec cares about.
        let ones: u64 = eval_of(i as usize)
            .iter()
            .enumerate()
            .map(|(block, &e)| (e & care_word(block)).count_ones() as u64)
            .sum();
        if ones == 0 {
            return 0.0;
        }
        let inter: u64 = eval_of(i as usize)
            .iter()
            .zip(&target)
            .map(|(&e, &t)| (e & t).count_ones() as u64)
            .sum();
        (inter as f64 / target_ones as f64) * (inter as f64 / ones as f64)
    };
    let mut by_gate = eligible.clone();
    by_gate.sort_unstable_by(|&a, &b| gate(b).total_cmp(&gate(a)));
    let mut by_generativity = eligible.clone();
    by_generativity.sort_unstable_by_key(|&i| std::cmp::Reverse(child_count[i as usize]));
    let mut by_miss = eligible;
    by_miss.sort_unstable_by_key(|&i| std::cmp::Reverse(miss_count[i as usize]));

    // Round-robin across the three rankings, greedily keeping only picks
    // that are behaviorally DIVERSE: a candidate must differ from every
    // already-taken stream (library leaves included) in enough bits.
    // Without this, one high-scoring semantic family floods every slot
    // with boundary-condition cousins of itself.
    let min_distance = (total_bits / 64).max(16);
    let mut taken_evals: Vec<Box<[u64]>> = existing
        .iter()
        .map(|leaf| {
            let mut eval = vec![0u64; corpus.total_blocks];
            eval_leaf(leaf, corpus, &mut eval);
            eval.into_boxed_slice()
        })
        .collect();
    let mut picks = Vec::new();
    let lists = [&by_gate, &by_generativity, &by_miss];
    let mut cursors = [0usize; 3];
    let mut stalled = 0;
    let mut which = 0;
    while picks.len() < room && stalled < 3 {
        let list = lists[which];
        let cursor = &mut cursors[which];
        let mut advanced = false;
        while *cursor < list.len() {
            let idx = list[*cursor];
            *cursor += 1;
            let entry = &bank[idx as usize];
            let eval = eval_of(idx as usize);
            let diverse = taken_evals.iter().all(|taken| {
                let d: u64 = taken
                    .iter()
                    .zip(eval)
                    .map(|(&a, &b)| (a ^ b).count_ones() as u64)
                    .sum();
                d >= min_distance
            });
            if diverse {
                taken_evals.push(eval.to_vec().into_boxed_slice());
                picks.push(Leaf::derived(
                    &term_expr(existing, existing_templates, bank, &entry.term),
                    term_graph(existing, existing_templates, bank, &entry.term),
                ));
                advanced = true;
                break;
            }
        }
        stalled = if advanced { 0 } else { stalled + 1 };
        which = (which + 1) % 3;
    }

    // Anti-unification: mine single-hole patterns from pairs of the
    // nearest misses. Recurring structure beats concrete instantiations —
    // one template transfers across every stream it could wrap.
    let mut mined: HashMap<TTree, u32> = HashMap::new();
    let near: Vec<u32> =
        by_distance[..by_distance.len().min(32)].iter().map(|&(_, i)| i).collect();
    for (ai, &a) in near.iter().enumerate() {
        for &b in &near[ai + 1..] {
            let pattern = lgg(bank, &bank[a as usize].term, &bank[b as usize].term);
            if pattern.holes() == 1 && pattern.nodes() >= 3 && pattern.stateful() {
                *mined.entry(pattern).or_insert(0) += 1;
            }
        }
    }
    let mut ranked: Vec<(TTree, u32)> = mined.into_iter().collect();
    ranked.sort_unstable_by(|a, b| {
        b.1.cmp(&a.1)
            .then(a.0.nodes().cmp(&b.0.nodes()))
            .then_with(|| a.0.pattern(existing).cmp(&b.0.pattern(existing)))
    });
    let templates: Vec<Template> = ranked
        .into_iter()
        .filter(|(body, _)| existing_templates.iter().all(|t| t.body != *body))
        .take(2)
        .map(|(body, _)| {
            let name = body.pattern(existing);
            let stateful = body.stateful();
            Template { body, name, stateful }
        })
        .collect();
    (picks, templates)
}

// --- Search space ---------------------------------------------------------

const UN_OPS: [UnOp; 4] = [UnOp::Not, UnOp::Shl, UnOp::ShlSeed, UnOp::PXor];
const BIN_OPS: [BinOp; 4] = [BinOp::And, BinOp::Or, BinOp::Xor, BinOp::Add];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum UnOp {
    Not,
    Shl,
    ShlSeed,
    PXor,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum BinOp {
    And,
    Or,
    Xor,
    Add,
}

#[derive(Clone, Copy)]
enum Term {
    Leaf(u16),
    Un(UnOp, u32),
    Bin(BinOp, u32, u32),
    /// Application of a mined template (index into the template library)
    /// to a banked child. Counts as size 1 + child size: that compression
    /// is exactly what promotion buys.
    Tpl(u16, u32),
}

/// A single-hole pattern mined by anti-unification of near-miss terms —
/// the DreamCoder move: promote recurring STRUCTURE, not just concrete
/// streams, so one abstraction transfers across instantiations.
#[derive(Clone, PartialEq, Eq, Hash)]
enum TTree {
    Hole,
    Leaf(u16),
    Un(UnOp, Box<TTree>),
    Bin(BinOp, Box<TTree>, Box<TTree>),
}

#[derive(Clone)]
struct Template {
    body: TTree,
    /// Display pattern with `_` at the hole.
    name: String,
    stateful: bool,
}

impl TTree {
    fn nodes(&self) -> usize {
        match self {
            TTree::Hole | TTree::Leaf(_) => 1,
            TTree::Un(_, a) => 1 + a.nodes(),
            TTree::Bin(_, a, b) => 1 + a.nodes() + b.nodes(),
        }
    }

    fn holes(&self) -> usize {
        match self {
            TTree::Hole => 1,
            TTree::Leaf(_) => 0,
            TTree::Un(_, a) => a.holes(),
            TTree::Bin(_, a, b) => a.holes() + b.holes(),
        }
    }

    fn stateful(&self) -> bool {
        match self {
            TTree::Hole | TTree::Leaf(_) => false,
            TTree::Un(op, a) => !matches!(op, UnOp::Not) || a.stateful(),
            TTree::Bin(op, a, b) => {
                matches!(op, BinOp::Add) || a.stateful() || b.stateful()
            }
        }
    }

    fn pattern(&self, leaves: &[Leaf]) -> String {
        match self {
            TTree::Hole => "_".into(),
            TTree::Leaf(li) => leaves[*li as usize].name.clone(),
            TTree::Un(op, a) => format!("{}({})", un_name(*op), a.pattern(leaves)),
            TTree::Bin(op, a, b) => {
                format!("{}({}, {})", bin_name(*op), a.pattern(leaves), b.pattern(leaves))
            }
        }
    }
}

fn un_name(op: UnOp) -> &'static str {
    match op {
        UnOp::Not => "Not",
        UnOp::Shl => "Shl1",
        UnOp::ShlSeed => "Shl1Seeded",
        UnOp::PXor => "PrefixXor",
    }
}

fn bin_name(op: BinOp) -> &'static str {
    match op {
        BinOp::And => "And",
        BinOp::Or => "Or",
        BinOp::Xor => "Xor",
        BinOp::Add => "Add",
    }
}

/// Evaluate a template body over the corpus with the hole bound to `hole`.
fn eval_ttree(body: &TTree, leaves: &[Leaf], corpus: &Corpus, hole: &[u64]) -> Vec<u64> {
    match body {
        TTree::Hole => hole.to_vec(),
        TTree::Leaf(li) => {
            let mut out = vec![0u64; corpus.total_blocks];
            eval_leaf(&leaves[*li as usize], corpus, &mut out);
            out
        }
        TTree::Un(op, a) => {
            let a = eval_ttree(a, leaves, corpus, hole);
            let mut out = vec![0u64; corpus.total_blocks];
            eval_un(*op, &a, corpus, &mut out);
            out
        }
        TTree::Bin(op, a, b) => {
            let a = eval_ttree(a, leaves, corpus, hole);
            let b = eval_ttree(b, leaves, corpus, hole);
            let mut out = vec![0u64; corpus.total_blocks];
            eval_bin(*op, &a, &b, corpus, &mut out);
            out
        }
    }
}

/// Least general generalization of two terms: equal structure survives,
/// disagreements become holes. Template applications generalize to holes
/// (no templates-of-templates).
fn lgg(bank: &[Entry], a: &Term, b: &Term) -> TTree {
    match (*a, *b) {
        (Term::Leaf(x), Term::Leaf(y)) if x == y => TTree::Leaf(x),
        (Term::Un(op1, c1), Term::Un(op2, c2)) if op1 == op2 => TTree::Un(
            op1,
            Box::new(lgg(bank, &bank[c1 as usize].term, &bank[c2 as usize].term)),
        ),
        (Term::Bin(op1, a1, b1), Term::Bin(op2, a2, b2)) if op1 == op2 => TTree::Bin(
            op1,
            Box::new(lgg(bank, &bank[a1 as usize].term, &bank[a2 as usize].term)),
            Box::new(lgg(bank, &bank[b1 as usize].term, &bank[b2 as usize].term)),
        ),
        _ => TTree::Hole,
    }
}

struct Entry {
    term: Term,
    size: u16,
}

// --- Corpus evaluation -----------------------------------------------------

/// Inputs flattened to per-node evaluation vectors: one u64 per 64-byte
/// block, blocks of all inputs concatenated. Stateful ops reset their
/// carries at every input boundary, exactly like a fresh interpreter run.
struct Corpus {
    inputs: Vec<Vec<u8>>,
    blocks: Vec<usize>,
    total_blocks: usize,
}

impl Corpus {
    fn new(inputs: &[Vec<u8>]) -> Self {
        let blocks: Vec<usize> = inputs
            .iter()
            .map(|input| {
                assert!(
                    !input.is_empty() && input.len() % 64 == 0,
                    "corpus inputs must be nonempty multiples of 64 bytes"
                );
                input.len() / 64
            })
            .collect();
        let total_blocks = blocks.iter().sum();
        Self { inputs: inputs.to_vec(), blocks, total_blocks }
    }
}

/// Bit i of the result is the XOR of bits 0..=i (the scalar PrefixXor,
/// duplicated from the interpreter so the search loop stays self-contained).
fn prefix_xor(mut x: u64) -> u64 {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    x
}

fn eval_class(class: &CharClass, corpus: &Corpus, out: &mut [u64]) {
    let mut off = 0;
    for input in &corpus.inputs {
        for chunk in input.chunks_exact(64) {
            let mut mask = 0u64;
            for (bit, &byte) in chunk.iter().enumerate() {
                mask |= (class.contains(byte) as u64) << bit;
            }
            out[off] = mask;
            off += 1;
        }
    }
}

fn eval_un(op: UnOp, a: &[u64], corpus: &Corpus, out: &mut [u64]) {
    match op {
        UnOp::Not => {
            for (o, &v) in out.iter_mut().zip(a) {
                *o = !v;
            }
        }
        UnOp::Shl | UnOp::ShlSeed => {
            let seed = (op == UnOp::ShlSeed) as u64;
            let mut off = 0;
            for &nb in &corpus.blocks {
                let mut carry = seed;
                for k in off..off + nb {
                    out[k] = (a[k] << 1) | carry;
                    carry = a[k] >> 63;
                }
                off += nb;
            }
        }
        UnOp::PXor => {
            let mut off = 0;
            for &nb in &corpus.blocks {
                let mut carry = 0u64;
                for k in off..off + nb {
                    let parity = prefix_xor(a[k]) ^ carry;
                    out[k] = parity;
                    carry = ((parity as i64) >> 63) as u64;
                }
                off += nb;
            }
        }
    }
}

fn eval_bin(op: BinOp, a: &[u64], b: &[u64], corpus: &Corpus, out: &mut [u64]) {
    match op {
        BinOp::And => {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = x & y;
            }
        }
        BinOp::Or => {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = x | y;
            }
        }
        BinOp::Xor => {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = x ^ y;
            }
        }
        BinOp::Add => {
            let mut off = 0;
            for &nb in &corpus.blocks {
                let mut carry = 0u64;
                for k in off..off + nb {
                    let (partial, c1) = a[k].overflowing_add(b[k]);
                    let (sum, c2) = partial.overflowing_add(carry);
                    out[k] = sum;
                    carry = (c1 | c2) as u64;
                }
                off += nb;
            }
        }
    }
}

/// Evaluate a graph over one input and return per-block masks (the
/// positions [`interp::run`] reports, re-packed as bits).
fn graph_masks(graph: &Graph, input: &[u8]) -> Vec<u64> {
    let mut positions = Vec::new();
    interp::run(graph, input, &mut positions);
    let mut masks = vec![0u64; input.len().div_ceil(64)];
    for p in positions {
        masks[(p / 64) as usize] |= 1u64 << (p % 64);
    }
    masks
}

fn eval_leaf(leaf: &Leaf, corpus: &Corpus, out: &mut [u64]) {
    match &leaf.kind {
        LeafKind::Class(class) => eval_class(class, corpus, out),
        LeafKind::Const(pattern) => out.fill(*pattern),
        LeafKind::Derived(graph) => {
            let mut off = 0;
            for input in &corpus.inputs {
                for mask in graph_masks(graph, input) {
                    out[off] = mask;
                    off += 1;
                }
            }
        }
    }
}

// --- Hashing for observational dedup ---------------------------------------

fn hash128(words: &[u64]) -> u128 {
    let mut a = 0xcbf2_9ce4_8422_2325u64;
    let mut b = 0x9e37_79b9_7f4a_7c15u64;
    for &w in words {
        a = (a ^ w).wrapping_mul(0x0000_0100_0000_01b3);
        b = (b ^ w.rotate_left(32)).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    }
    ((a as u128) << 64) | b as u128
}

/// The keys are already strong 128-bit hashes; fold instead of rehashing.
#[derive(Default)]
struct FoldHasher(u64);

impl Hasher for FoldHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.0 = (self.0 ^ u64::from_le_bytes(buf)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type SeenMap = HashMap<u128, (), BuildHasherDefault<FoldHasher>>;
type LocalSeen = std::collections::HashSet<u128, BuildHasherDefault<FoldHasher>>;

/// One unit of parallel work: an operator applied over a stripe of the
/// bank. Rows carry their operator because per-op level weights select
/// different child levels under [`Order::Cost`].
#[derive(Clone, Copy)]
enum Row {
    /// (op, left child level, left child index); right children span the
    /// complementary level.
    Bin(BinOp, usize, usize),
    Un(UnOp, usize),
    Tpl(u16, usize),
}

/// True expression-tree size of a term whose children are banked.
fn term_size(bank: &[Entry], term: &Term) -> u16 {
    match *term {
        Term::Leaf(_) => 1,
        Term::Un(_, a) | Term::Tpl(_, a) => 1 + bank[a as usize].size,
        Term::Bin(_, a, b) => 1 + bank[a as usize].size + bank[b as usize].size,
    }
}

/// One thread's share of a level's candidate stream.
#[derive(Default)]
struct ShardOut {
    count: u64,
    /// (hash, term) of candidates new to both the frozen global map and
    /// this shard. Evals are NOT kept — the merge re-derives survivors.
    fresh: Vec<(u128, Term)>,
    /// Candidates whose corpus behavior matched the target.
    matches: Vec<Term>,
}

/// Evaluate one term whose children (if any) are already banked, writing
/// into `out`.
fn eval_term(
    leaves: &[Leaf],
    templates: &[Template],
    evals: &[u64],
    corpus: &Corpus,
    term: Term,
    out: &mut [u64],
) {
    let len = corpus.total_blocks;
    match term {
        Term::Leaf(li) => eval_leaf(&leaves[li as usize], corpus, out),
        Term::Un(op, a) => eval_un(op, &evals[a as usize * len..][..len], corpus, out),
        Term::Bin(op, a, b) => eval_bin(
            op,
            &evals[a as usize * len..][..len],
            &evals[b as usize * len..][..len],
            corpus,
            out,
        ),
        Term::Tpl(t, a) => {
            let applied = eval_ttree(
                &templates[t as usize].body,
                leaves,
                corpus,
                &evals[a as usize * len..][..len],
            );
            out.copy_from_slice(&applied);
        }
    }
}

// --- The enumeration loop ---------------------------------------------------

enum Attempt {
    Found(Box<Solution>),
    Exhausted(Stats, Option<Dump>),
    Counterexample(Vec<u8>),
}

/// The remains of an exhausted search, kept for abstraction scoring.
struct Dump {
    bank: Vec<Entry>,
    /// Eval vectors of all banked terms, contiguous (entry i occupies
    /// words [i*eval_len, (i+1)*eval_len)).
    evals: Vec<u64>,
    eval_len: usize,
    target: Vec<u64>,
    care: Option<Vec<u64>>,
}

struct Search<'a> {
    leaves: &'a [Leaf],
    corpus: Corpus,
    target: Vec<u64>,
    /// Precomputed care masks over the corpus (None = all bits matter).
    care: Option<Vec<u64>>,
    spec: &'a Spec<'a>,
    budget: &'a Budget,
    templates: &'a [Template],
    /// Level weight each template application adds (precomputed per order).
    template_weights: Vec<usize>,
    bank: Vec<Entry>,
    /// Arena of eval vectors, one corpus-length stripe per banked term.
    evals: Vec<u64>,
    /// Banked terms grouped by level (tree size or tree cost, per order).
    by_level: Vec<Vec<u32>>,
    seen: SeenMap,
    candidates: &'a mut u64,
    restarts: usize,
    start: Instant,
    /// Cheapest verified solution so far (search settles before returning).
    best: Option<Box<Solution>>,
    /// Tree size of the first verified match — settling is measured from here.
    found_at: Option<usize>,
    /// Distinct corpus-matching forms seen, as (cost, expr).
    alternates: Vec<(u32, String)>,
}

/// `(eval ^ target) & care == 0`, with a fast exact path.
fn masked_eq(eval: &[u64], target: &[u64], care: Option<&[u64]>) -> bool {
    match care {
        None => eval == target,
        Some(care) => eval
            .iter()
            .zip(target)
            .zip(care)
            .all(|((&e, &t), &c)| (e ^ t) & c == 0),
    }
}

#[expect(clippy::too_many_arguments, reason = "internal driver shared by two front doors")]
fn attempt(
    leaves: &[Leaf],
    templates: &[Template],
    inputs: &[Vec<u8>],
    spec: &Spec,
    budget: &Budget,
    candidates: &mut u64,
    restarts: usize,
    start: Instant,
    want_dump: bool,
) -> Attempt {
    let corpus = Corpus::new(inputs);
    let mut target = Vec::with_capacity(corpus.total_blocks);
    for input in &corpus.inputs {
        let masks = (spec.reference)(input);
        assert_eq!(masks.len(), input.len() / 64, "reference returned wrong block count");
        target.extend_from_slice(&masks);
    }
    let care = spec.care.map(|care_fn| {
        let mut masks = Vec::with_capacity(corpus.total_blocks);
        for input in &corpus.inputs {
            let blocks = care_fn(input);
            assert_eq!(blocks.len(), input.len() / 64, "care returned wrong block count");
            masks.extend_from_slice(&blocks);
        }
        masks
    });

    let mut search = Search {
        leaves,
        templates,
        corpus,
        target,
        care,
        spec,
        budget,
        template_weights: templates
            .iter()
            .map(|tpl| ttree_weight(&tpl.body, leaves, budget.order, &budget.cost))
            .collect(),
        bank: Vec::new(),
        evals: Vec::new(),
        by_level: vec![Vec::new(); budget.max_level + 1],
        seen: SeenMap::default(),
        candidates,
        restarts,
        start,
        best: None,
        found_at: None,
        alternates: Vec::new(),
    };
    match search.run() {
        RunOutcome::Found(solution) => Attempt::Found(solution),
        RunOutcome::Counterexample(witness) => Attempt::Counterexample(witness),
        RunOutcome::Exhausted(stats) => {
            let eval_len = search.corpus.total_blocks;
            let dump = want_dump.then_some(Dump {
                bank: search.bank,
                evals: search.evals,
                eval_len,
                target: search.target,
                care: search.care,
            });
            Attempt::Exhausted(stats, dump)
        }
    }
}

enum RunOutcome {
    Found(Box<Solution>),
    Exhausted(Stats),
    Counterexample(Vec<u8>),
}

/// Per-candidate verdict from the budget/dedup/match pipeline.
enum Verdict {
    Continue,
    Stop(RunOutcome),
}

impl Search<'_> {
    fn run(&mut self) -> RunOutcome {
        let mut scratch = vec![0u64; self.corpus.total_blocks];

        for li in 0..self.leaves.len() {
            let level = leaf_level(&self.leaves[li], self.budget.order, &self.budget.cost);
            if level > self.budget.max_level {
                continue; // Too expensive to ever appear in a solution.
            }
            eval_leaf(&self.leaves[li], &self.corpus, &mut scratch);
            if let Verdict::Stop(outcome) = self.consider(Term::Leaf(li as u16), &scratch, level)
            {
                return outcome;
            }
        }

        for level in 1..=self.budget.max_level {
            if let Some(outcome) = self.enumerate_level(level, &mut scratch) {
                return outcome;
            }
            self.progress(level);
            // Settle: after the first verified match, finish settle_levels
            // further levels hunting cheaper equivalents, then stop.
            if let Some(found_level) = self.found_at
                && level >= found_level + self.budget.settle_levels
            {
                break;
            }
        }
        self.finish(self.budget.max_level)
    }

    /// One whole level, enumerated in parallel: candidate rows are
    /// sharded across threads, each shard evaluates/hashes/dedups against
    /// the FROZEN global map plus a local set, and a serial merge (in shard
    /// order, so bank layout stays deterministic) re-evaluates only the
    /// genuinely new terms into the arena. Shards never store eval vectors
    /// — only (hash, term) pairs — which is what keeps memory flat while
    /// the candidate stream is 10-50x larger than the survivor set.
    ///
    /// Rows carry their operator because under [`Order::Cost`] each op's
    /// weight selects different child levels.
    fn enumerate_level(&mut self, level: usize, scratch: &mut [u64]) -> Option<RunOutcome> {
        let order = self.budget.order;
        let model = self.budget.cost;
        let mut rows: Vec<Row> = Vec::new();
        let mut weights: Vec<u64> = Vec::new();
        // Binary rows, smallest left child first: deep right spines
        // (common in real kernels) surface before the level completes.
        for op in BIN_OPS {
            let w = bin_weight(op, order, &model);
            if w > level {
                continue;
            }
            let rem = level - w;
            for l1 in 0..=rem / 2 {
                let l2 = rem - l1;
                if l2 > self.budget.max_level {
                    continue;
                }
                let n2 = self.by_level[l2].len() as u64;
                if n2 == 0 {
                    continue;
                }
                for ii in 0..self.by_level[l1].len() {
                    rows.push(Row::Bin(op, l1, ii));
                    weights.push(n2);
                }
            }
        }
        for op in UN_OPS {
            let w = un_weight(op, order, &model);
            if w > level {
                continue;
            }
            for ii in 0..self.by_level[level - w].len() {
                rows.push(Row::Un(op, ii));
                weights.push(1);
            }
        }
        for ti in 0..self.templates.len() {
            let w = self.template_weights[ti];
            if w == 0 || w > level {
                continue;
            }
            for ii in 0..self.by_level[level - w].len() {
                rows.push(Row::Tpl(ti as u16, ii));
                weights.push(1);
            }
        }

        const BATCH: u64 = 2_000_000;
        let mut row = 0usize;
        while row < rows.len() {
            // Take rows until ~BATCH candidates, then process and merge so
            // budget checks stay fine-grained.
            let mut end = row;
            let mut batch_weight = 0u64;
            while end < rows.len() && batch_weight < BATCH {
                batch_weight += weights[end];
                end += 1;
            }
            let outs = self.evaluate_rows(level, &rows[row..end]);
            row = end;

            *self.candidates += outs.iter().map(|out| out.count).sum::<u64>();
            if *self.candidates > self.budget.max_candidates {
                return Some(self.finish(level.saturating_sub(1)));
            }
            for out in &outs {
                for &(hash, term) in &out.fresh {
                    // A full bank freezes admissions but NOT the search:
                    // deeper matches only need already-banked children, so
                    // aborting here would throw away exactly the levels
                    // where solutions live.
                    if self.bank.len() >= self.budget.max_bank {
                        break;
                    }
                    if let std::collections::hash_map::Entry::Vacant(slot) =
                        self.seen.entry(hash)
                    {
                        slot.insert(());
                        eval_term(self.leaves, self.templates, &self.evals, &self.corpus, term, scratch);
                        let idx = self.bank.len() as u32;
                        let size = term_size(&self.bank, &term);
                        self.bank.push(Entry { term, size });
                        self.evals.extend_from_slice(scratch);
                        self.by_level[level].push(idx);
                    }
                }
            }
            for out in outs {
                for term in out.matches {
                    if let Some(witness) = self.handle_match(term, level) {
                        return Some(RunOutcome::Counterexample(witness));
                    }
                }
            }
        }
        None
    }

    /// Parallel phase: evaluate every candidate in the given rows. Returns
    /// one shard output per thread, in deterministic shard order.
    fn evaluate_rows(&self, level: usize, rows: &[Row]) -> Vec<ShardOut> {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
        let total: u64 = rows.len() as u64;
        let per_shard = total.div_ceil(threads as u64).max(1) as usize;
        let order = self.budget.order;
        let model = self.budget.cost;
        let corpus = &self.corpus;
        let evals = &self.evals;
        let by_level = &self.by_level;
        let seen = &self.seen;
        let target = &self.target;
        let care = self.care.as_deref();
        let leaves = self.leaves;
        let templates = self.templates;
        let template_weights = &self.template_weights;
        let eval_len = corpus.total_blocks;
        std::thread::scope(|scope| {
            let handles: Vec<_> = rows
                .chunks(per_shard)
                .map(|chunk| {
                    scope.spawn(move || {
                        let eval_of =
                            |i: u32| -> &[u64] { &evals[i as usize * eval_len..][..eval_len] };
                        let mut scratch = vec![0u64; eval_len];
                        let mut local_seen: LocalSeen = LocalSeen::default();
                        let mut out = ShardOut::default();
                        let mut emit = |term: Term, eval: &[u64], out: &mut ShardOut| {
                            out.count += 1;
                            if masked_eq(eval, target, care) {
                                out.matches.push(term);
                                return;
                            }
                            let hash = hash128(eval);
                            if !seen.contains_key(&hash) && local_seen.insert(hash) {
                                out.fresh.push((hash, term));
                            }
                        };
                        for &row in chunk {
                            match row {
                                Row::Un(op, ii) => {
                                    let child =
                                        by_level[level - un_weight(op, order, &model)][ii];
                                    eval_un(op, eval_of(child), corpus, &mut scratch);
                                    emit(Term::Un(op, child), &scratch, &mut out);
                                }
                                Row::Tpl(ti, ii) => {
                                    let child =
                                        by_level[level - template_weights[ti as usize]][ii];
                                    let applied = eval_ttree(
                                        &templates[ti as usize].body,
                                        leaves,
                                        corpus,
                                        eval_of(child),
                                    );
                                    emit(Term::Tpl(ti, child), &applied, &mut out);
                                }
                                Row::Bin(op, l1, ii) => {
                                    let l2 = level - bin_weight(op, order, &model) - l1;
                                    let i = by_level[l1][ii];
                                    let jj_start = if l1 == l2 { ii } else { 0 };
                                    for &j in &by_level[l2][jj_start..] {
                                        eval_bin(op, eval_of(i), eval_of(j), corpus, &mut scratch);
                                        emit(Term::Bin(op, i, j), &scratch, &mut out);
                                    }
                                }
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("shard panicked")).collect()
        })
    }

    /// Terminal outcome: the best verified solution if one exists (with
    /// alternates attached), otherwise exhaustion.
    fn finish(&mut self, completed_level: usize) -> RunOutcome {
        let stats = self.stats(completed_level);
        match self.best.take() {
            Some(mut solution) => {
                self.alternates.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                solution.alternates = std::mem::take(&mut self.alternates);
                solution.stats = stats;
                RunOutcome::Found(solution)
            }
            None => RunOutcome::Exhausted(stats),
        }
    }

    fn consider(&mut self, term: Term, eval: &[u64], level: usize) -> Verdict {
        *self.candidates += 1;
        if *self.candidates > self.budget.max_candidates {
            return Verdict::Stop(self.finish(level.saturating_sub(1)));
        }
        if masked_eq(eval, &self.target, self.care.as_deref()) {
            return match self.handle_match(term, level) {
                Some(witness) => Verdict::Stop(RunOutcome::Counterexample(witness)),
                None => Verdict::Continue,
            };
        }
        let key = hash128(eval);
        if self.bank.len() < self.budget.max_bank
            && let std::collections::hash_map::Entry::Vacant(slot) = self.seen.entry(key)
        {
            slot.insert(());
            let idx = self.bank.len() as u32;
            let size = term_size(&self.bank, &term);
            self.bank.push(Entry { term, size });
            self.evals.extend_from_slice(eval);
            self.by_level[level].push(idx);
        }
        Verdict::Continue
    }

    /// A candidate matched the corpus. Record it as an alternate; if it is
    /// cheaper than the current best, verify it and adopt it. Returns a
    /// counterexample witness if verification fails (CEGIS restart).
    fn handle_match(&mut self, term: Term, level: usize) -> Option<Vec<u8>> {
        let graph = term_graph(self.leaves, self.templates, &self.bank, &term);
        let cost = graph_cost(&graph, &self.budget.cost);
        let expr = term_expr(self.leaves, self.templates, &self.bank, &term);
        if self.alternates.len() < 8 && self.alternates.iter().all(|(_, e)| *e != expr) {
            self.alternates.push((cost, expr.clone()));
        }
        if self.best.as_ref().is_some_and(|best| cost >= best.cost) {
            return None;
        }
        match self.verify(&graph) {
            Err(witness) => Some(witness),
            Ok(verified_inputs) => {
                self.found_at.get_or_insert(level);
                let stats = self.stats(level);
                self.best = Some(Box::new(Solution {
                    expr,
                    dag_nodes: graph.nodes().len(),
                    cost,
                    graph,
                    tree_size: term_size(&self.bank, &term) as usize,
                    verified_inputs,
                    alternates: Vec::new(),
                    stats,
                }));
                None
            }
        }
    }

    /// Differential verification on fresh random inputs, equality required
    /// only at care positions.
    fn verify(&self, graph: &Graph) -> Result<usize, Vec<u8>> {
        let alphabet = self.alphabet();
        let mut rng = Rng(0x5DEE_CE66_D1CE_CAFE ^ (self.restarts as u64) << 32);
        const ROUNDS: usize = 4000;
        for _ in 0..ROUNDS {
            let input = random_input(&alphabet, &mut rng);
            let expected = (self.spec.reference)(&input);
            let care = self.spec.care.map(|care_fn| care_fn(&input));
            let actual = graph_masks(graph, &input);
            let agree = expected.iter().zip(&actual).enumerate().all(|(block, (&e, &a))| {
                let mut relevant = care.as_ref().map_or(!0, |masks| masks[block]);
                // Pad bits beyond the input are never compared.
                let tail = input.len() - block * 64;
                if tail < 64 {
                    relevant &= (1u64 << tail) - 1;
                }
                (e ^ a) & relevant == 0
            });
            if !agree {
                // Pad the witness to a whole block with a byte the corpus
                // already uses as filler so it can join the corpus.
                let mut witness = input;
                witness.resize(witness.len().next_multiple_of(64).max(64), alphabet[0]);
                return Err(witness);
            }
        }
        Ok(ROUNDS)
    }

    /// Filler byte first: it pads counterexample witnesses.
    fn alphabet(&self) -> Vec<u8> {
        let mut seen = [false; 256];
        for input in &self.corpus.inputs {
            for &b in input {
                seen[b as usize] = true;
            }
        }
        let mut bytes = vec![b'x'];
        bytes.extend((0..=255u8).filter(|&b| seen[b as usize] && b != b'x'));
        bytes
    }

    fn stats(&self, completed_level: usize) -> Stats {
        Stats {
            candidates: *self.candidates,
            bank_unique: self.bank.len(),
            bank_saturated: self.bank.len() >= self.budget.max_bank,
            completed_level,
            restarts: self.restarts,
            elapsed_ms: self.start.elapsed().as_millis(),
        }
    }

    fn progress(&self, level: usize) {
        if self.budget.progress {
            eprintln!(
                "    level {level:2} complete: {} distinct terms, {} candidates, {:.1}s",
                self.bank.len(),
                self.candidates,
                self.start.elapsed().as_secs_f64()
            );
        }
    }

}

// --- Reconstruction as a shared-subexpression IR graph ---

fn term_graph(leaves: &[Leaf], templates: &[Template], bank: &[Entry], top: &Term) -> Graph {
    let mut g = Graph::new();
    let mut cse: HashMap<CseKey, NodeId> = HashMap::new();
    let mut memo: Vec<Option<NodeId>> = vec![None; bank.len()];
    let output = emit(&mut g, &mut cse, &mut memo, leaves, templates, bank, top);
    g.set_output(output);
    g
}

fn emit_idx(
    g: &mut Graph,
    cse: &mut HashMap<CseKey, NodeId>,
    memo: &mut Vec<Option<NodeId>>,
    leaves: &[Leaf],
    templates: &[Template],
    bank: &[Entry],
    idx: u32,
) -> NodeId {
    if let Some(id) = memo[idx as usize] {
        return id;
    }
    let term = bank[idx as usize].term;
    let id = emit(g, cse, memo, leaves, templates, bank, &term);
    memo[idx as usize] = Some(id);
    id
}

fn emit(
    g: &mut Graph,
    cse: &mut HashMap<CseKey, NodeId>,
    memo: &mut Vec<Option<NodeId>>,
    leaves: &[Leaf],
    templates: &[Template],
    bank: &[Entry],
    term: &Term,
) -> NodeId {
    match *term {
        Term::Leaf(li) => emit_leaf(g, cse, &leaves[li as usize]),
        Term::Un(op, a) => {
            let a = emit_idx(g, cse, memo, leaves, templates, bank, a);
            match op {
                UnOp::Not => keyed(g, cse, CseKey::Not(a.0), |g| g.not(a)),
                UnOp::Shl => keyed(g, cse, CseKey::Shl(a.0), |g| g.shift_left1(a)),
                UnOp::ShlSeed => keyed(g, cse, CseKey::ShlSeed(a.0), |g| g.shift_left1_seeded(a)),
                UnOp::PXor => keyed(g, cse, CseKey::PXor(a.0), |g| g.prefix_xor(a)),
            }
        }
        Term::Bin(op, a, b) => {
            let a = emit_idx(g, cse, memo, leaves, templates, bank, a);
            let b = emit_idx(g, cse, memo, leaves, templates, bank, b);
            let (lo, hi) = (a.0.min(b.0), a.0.max(b.0));
            match op {
                BinOp::And => keyed(g, cse, CseKey::And(lo, hi), |g| g.and(a, b)),
                BinOp::Or => keyed(g, cse, CseKey::Or(lo, hi), |g| g.or(a, b)),
                BinOp::Xor => keyed(g, cse, CseKey::Xor(lo, hi), |g| g.xor(a, b)),
                BinOp::Add => keyed(g, cse, CseKey::Add(lo, hi), |g| g.add(a, b)),
            }
        }
        Term::Tpl(t, a) => {
            let hole = emit_idx(g, cse, memo, leaves, templates, bank, a);
            emit_ttree(g, cse, leaves, &templates[t as usize].body, hole)
        }
    }
}

fn emit_leaf(g: &mut Graph, cse: &mut HashMap<CseKey, NodeId>, leaf: &Leaf) -> NodeId {
    match &leaf.kind {
        LeafKind::Class(class) => keyed(g, cse, CseKey::Class(class.words()), |g| g.class(*class)),
        LeafKind::Const(pattern) => {
            keyed(g, cse, CseKey::Const(*pattern), |g| g.constant(*pattern))
        }
        LeafKind::Derived(sub) => splice(g, cse, sub),
    }
}

/// Emit a template body with the hole bound to an existing node.
fn emit_ttree(
    g: &mut Graph,
    cse: &mut HashMap<CseKey, NodeId>,
    leaves: &[Leaf],
    body: &TTree,
    hole: NodeId,
) -> NodeId {
    match body {
        TTree::Hole => hole,
        TTree::Leaf(li) => emit_leaf(g, cse, &leaves[*li as usize]),
        TTree::Un(op, a) => {
            let a = emit_ttree(g, cse, leaves, a, hole);
            match op {
                UnOp::Not => keyed(g, cse, CseKey::Not(a.0), |g| g.not(a)),
                UnOp::Shl => keyed(g, cse, CseKey::Shl(a.0), |g| g.shift_left1(a)),
                UnOp::ShlSeed => keyed(g, cse, CseKey::ShlSeed(a.0), |g| g.shift_left1_seeded(a)),
                UnOp::PXor => keyed(g, cse, CseKey::PXor(a.0), |g| g.prefix_xor(a)),
            }
        }
        TTree::Bin(op, a, b) => {
            let a = emit_ttree(g, cse, leaves, a, hole);
            let b = emit_ttree(g, cse, leaves, b, hole);
            let (lo, hi) = (a.0.min(b.0), a.0.max(b.0));
            match op {
                BinOp::And => keyed(g, cse, CseKey::And(lo, hi), |g| g.and(a, b)),
                BinOp::Or => keyed(g, cse, CseKey::Or(lo, hi), |g| g.or(a, b)),
                BinOp::Xor => keyed(g, cse, CseKey::Xor(lo, hi), |g| g.xor(a, b)),
                BinOp::Add => keyed(g, cse, CseKey::Add(lo, hi), |g| g.add(a, b)),
            }
        }
    }
}

// --- Pretty printing ---

fn term_expr(leaves: &[Leaf], templates: &[Template], bank: &[Entry], term: &Term) -> String {
    match *term {
        Term::Leaf(li) => leaves[li as usize].name.clone(),
        Term::Un(op, a) => {
            format!("{}({})", un_name(op), term_expr(leaves, templates, bank, &bank[a as usize].term))
        }
        Term::Bin(op, a, b) => {
            format!(
                "{}({}, {})",
                bin_name(op),
                term_expr(leaves, templates, bank, &bank[a as usize].term),
                term_expr(leaves, templates, bank, &bank[b as usize].term)
            )
        }
        Term::Tpl(t, a) => ttree_expr(
            &templates[t as usize].body,
            leaves,
            &term_expr(leaves, templates, bank, &bank[a as usize].term),
        ),
    }
}

/// Print a template body with the hole replaced by `hole_expr`.
fn ttree_expr(body: &TTree, leaves: &[Leaf], hole_expr: &str) -> String {
    match body {
        TTree::Hole => hole_expr.into(),
        TTree::Leaf(li) => leaves[*li as usize].name.clone(),
        TTree::Un(op, a) => format!("{}({})", un_name(*op), ttree_expr(a, leaves, hole_expr)),
        TTree::Bin(op, a, b) => format!(
            "{}({}, {})",
            bin_name(*op),
            ttree_expr(a, leaves, hole_expr),
            ttree_expr(b, leaves, hole_expr)
        ),
    }
}

/// Structural key for sharing nodes during reconstruction (commutative
/// operands canonicalized by the caller).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CseKey {
    Class([u64; 4]),
    Const(u64),
    Not(u32),
    Shl(u32),
    ShlSeed(u32),
    PXor(u32),
    And(u32, u32),
    Or(u32, u32),
    Xor(u32, u32),
    Add(u32, u32),
    Regions(u32, u32, u32),
}

fn keyed(
    g: &mut Graph,
    cse: &mut HashMap<CseKey, NodeId>,
    key: CseKey,
    make: impl FnOnce(&mut Graph) -> NodeId,
) -> NodeId {
    if let Some(&id) = cse.get(&key) {
        return id;
    }
    let id = make(g);
    cse.insert(key, id);
    id
}

/// Copy a subgraph into `g` node by node, sharing through the CSE map, and
/// return the node corresponding to the subgraph's output.
fn splice(g: &mut Graph, cse: &mut HashMap<CseKey, NodeId>, sub: &Graph) -> NodeId {
    let mut map: Vec<NodeId> = Vec::with_capacity(sub.nodes().len());
    for op in sub.nodes() {
        let id = match *op {
            Op::Class(class) => keyed(g, cse, CseKey::Class(class.words()), |g| g.class(class)),
            Op::Const(pattern) => keyed(g, cse, CseKey::Const(pattern), |g| g.constant(pattern)),
            Op::Not(a) => {
                let a = map[a.0 as usize];
                keyed(g, cse, CseKey::Not(a.0), |g| g.not(a))
            }
            Op::ShiftLeft1(a) => {
                let a = map[a.0 as usize];
                keyed(g, cse, CseKey::Shl(a.0), |g| g.shift_left1(a))
            }
            Op::ShiftLeft1Seeded(a) => {
                let a = map[a.0 as usize];
                keyed(g, cse, CseKey::ShlSeed(a.0), |g| g.shift_left1_seeded(a))
            }
            Op::PrefixXor(a) => {
                let a = map[a.0 as usize];
                keyed(g, cse, CseKey::PXor(a.0), |g| g.prefix_xor(a))
            }
            Op::And(a, b) => {
                let (a, b) = (map[a.0 as usize], map[b.0 as usize]);
                keyed(g, cse, CseKey::And(a.0.min(b.0), a.0.max(b.0)), |g| g.and(a, b))
            }
            Op::Or(a, b) => {
                let (a, b) = (map[a.0 as usize], map[b.0 as usize]);
                keyed(g, cse, CseKey::Or(a.0.min(b.0), a.0.max(b.0)), |g| g.or(a, b))
            }
            Op::Xor(a, b) => {
                let (a, b) = (map[a.0 as usize], map[b.0 as usize]);
                keyed(g, cse, CseKey::Xor(a.0.min(b.0), a.0.max(b.0)), |g| g.xor(a, b))
            }
            Op::Add(a, b) => {
                let (a, b) = (map[a.0 as usize], map[b.0 as usize]);
                keyed(g, cse, CseKey::Add(a.0.min(b.0), a.0.max(b.0)), |g| g.add(a, b))
            }
            Op::Regions(q, s, n) => {
                let (q, s, n) = (map[q.0 as usize], map[s.0 as usize], map[n.0 as usize]);
                keyed(g, cse, CseKey::Regions(q.0, s.0, n.0), |g| g.regions(q, s, n))
            }
        };
        map.push(id);
    }
    map[sub.output().0 as usize]
}

// --- Complete equivalence proof via product reachability ---------------------

/// An explicit byte-level serial machine: the formal counterpart of a
/// reference closure. `step` consumes one byte and returns the next state
/// and the output bit for this byte's position.
pub struct Fsm<'a> {
    pub initial: u32,
    pub step: &'a dyn Fn(u32, u8) -> (u32, bool),
}

#[derive(Debug)]
pub struct Proof {
    /// Distinct (graph carries, position mod 64, machine state) triples.
    pub product_states: usize,
    pub transitions: u64,
}

pub enum ProveOutcome {
    /// The graph equals the machine on EVERY input of every length.
    Proven(Proof),
    /// A shortest mismatching input, found breadth-first.
    Refuted(Vec<u8>),
    /// The product state space exceeded the exploration cap.
    Aborted { explored: usize },
}

/// Prove (not test) that `graph` computes the same stream as `fsm`.
///
/// Every IR op except `Regions` has an exact byte-serial form carrying at
/// most one bit of state (`Shl1` keeps the previous input bit, `PrefixXor`
/// its running parity, `Add` its carry — the bit-serial full adder), so a
/// graph IS a finite automaton over bytes whose state is one bit per
/// stateful node plus the position mod 64 (block constants are positional).
/// Equivalence against the machine is then plain product reachability:
/// finite, exact, and complete over all inputs — block boundaries included,
/// because the byte-serial semantics is what the blockwise interpreter
/// implements (asserted by differential tests).
///
/// Panics on graphs containing `Regions` (sequential, out of scope).
pub fn prove(graph: &Graph, fsm: &Fsm) -> ProveOutcome {
    const MAX_STATES: usize = 1 << 22;
    let initial = ByteState { carries: seed_carries(graph), pos: 0, machine: fsm.initial };
    let mut parents: HashMap<ByteState, Option<(ByteState, u8)>> = HashMap::new();
    parents.insert(initial, None);
    let mut queue = std::collections::VecDeque::from([initial]);
    let mut transitions = 0u64;
    let mut values = vec![false; graph.nodes().len()];
    while let Some(state) = queue.pop_front() {
        for byte in 0..=255u8 {
            transitions += 1;
            let (next_carries, graph_out) = step_byte(graph, state.carries, state.pos, byte, &mut values);
            let (next_machine, machine_out) = (fsm.step)(state.machine, byte);
            if graph_out != machine_out {
                // Rebuild the shortest input reaching this disagreement.
                let mut input = vec![byte];
                let mut cursor = state;
                while let Some(&Some((prev, b))) = parents.get(&cursor) {
                    input.push(b);
                    cursor = prev;
                }
                input.reverse();
                return ProveOutcome::Refuted(input);
            }
            let next = ByteState {
                carries: next_carries,
                pos: (state.pos + 1) % 64,
                machine: next_machine,
            };
            let occupancy = parents.len();
            if let std::collections::hash_map::Entry::Vacant(slot) = parents.entry(next) {
                if occupancy >= MAX_STATES {
                    return ProveOutcome::Aborted { explored: occupancy };
                }
                slot.insert(Some((state, byte)));
                queue.push_back(next);
            }
        }
    }
    ProveOutcome::Proven(Proof { product_states: parents.len(), transitions })
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ByteState {
    /// One bit per node (stateful nodes use theirs; others stay 0).
    carries: u64,
    /// Stream position mod 64: block constants are positional.
    pos: u8,
    machine: u32,
}

fn seed_carries(graph: &Graph) -> u64 {
    assert!(graph.nodes().len() <= 64, "prover packs node carries into a u64");
    let mut carries = 0u64;
    for (i, op) in graph.nodes().iter().enumerate() {
        if matches!(op, Op::ShiftLeft1Seeded(_)) {
            carries |= 1 << i;
        }
    }
    carries
}

/// One byte of byte-serial graph execution. `values` is caller-provided
/// scratch (one bool per node).
fn step_byte(
    graph: &Graph,
    carries: u64,
    pos: u8,
    byte: u8,
    values: &mut [bool],
) -> (u64, bool) {
    let mut next = carries;
    for (i, op) in graph.nodes().iter().enumerate() {
        let bit = 1u64 << i;
        values[i] = match *op {
            Op::Class(class) => class.contains(byte),
            Op::Const(pattern) => (pattern >> pos) & 1 != 0,
            Op::Not(a) => !values[a.0 as usize],
            Op::And(a, b) => values[a.0 as usize] & values[b.0 as usize],
            Op::Or(a, b) => values[a.0 as usize] | values[b.0 as usize],
            Op::Xor(a, b) => values[a.0 as usize] ^ values[b.0 as usize],
            Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) => {
                let out = carries & bit != 0;
                if values[a.0 as usize] {
                    next |= bit;
                } else {
                    next &= !bit;
                }
                out
            }
            Op::PrefixXor(a) => {
                let out = (carries & bit != 0) ^ values[a.0 as usize];
                if out {
                    next |= bit;
                } else {
                    next &= !bit;
                }
                out
            }
            Op::Add(a, b) => {
                let (x, y, c) =
                    (values[a.0 as usize], values[b.0 as usize], carries & bit != 0);
                let carry_out = (x & y) | (x & c) | (y & c);
                if carry_out {
                    next |= bit;
                } else {
                    next &= !bit;
                }
                x ^ y ^ c
            }
            Op::Regions(..) => panic!("prove() does not support the sequential Regions op"),
        };
    }
    (next, values[graph.output().0 as usize])
}

/// Byte-serial execution over a whole input — the prover's semantics,
/// exposed so tests can differentially pin it against [`interp::run`].
pub fn byte_serial_masks(graph: &Graph, input: &[u8]) -> Vec<u64> {
    let mut masks = vec![0u64; input.len().div_ceil(64)];
    let mut carries = seed_carries(graph);
    let mut values = vec![false; graph.nodes().len()];
    for (i, &byte) in input.iter().enumerate() {
        let (next, out) = step_byte(graph, carries, (i % 64) as u8, byte, &mut values);
        carries = next;
        if out {
            masks[i / 64] |= 1 << (i % 64);
        }
    }
    masks
}

// --- Verification input generation ------------------------------------------

/// xorshift64*; avoids a dev-dependency, same generator the tests use.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Random input over the corpus alphabet: half byte-at-a-time uniform, half
/// run-structured (runs are where carry bugs and parity tricks live).
fn random_input(alphabet: &[u8], rng: &mut Rng) -> Vec<u8> {
    let len = (rng.next() % 384) as usize;
    let mut input = Vec::with_capacity(len);
    if rng.next().is_multiple_of(2) {
        while input.len() < len {
            input.push(alphabet[(rng.next() % alphabet.len() as u64) as usize]);
        }
    } else {
        while input.len() < len {
            let byte = alphabet[(rng.next() % alphabet.len() as u64) as usize];
            let run = 1 + (rng.next() % 12) as usize;
            for _ in 0..run.min(len - input.len()) {
                input.push(byte);
            }
        }
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(max_level: usize) -> Budget {
        Budget {
            max_level,
            max_candidates: 2_000_000,
            max_bank: 200_000,
            settle_levels: 0,
            cost: CostModel::nodes(),
            order: Order::TreeSize,
            progress: false,
        }
    }

    fn corpus(alphabet: &[u8], seed: u64, inputs: usize, blocks: usize) -> Vec<Vec<u8>> {
        let mut rng = Rng(seed);
        (0..inputs)
            .map(|_| {
                (0..blocks * 64)
                    .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                    .collect()
            })
            .collect()
    }

    fn mask_ref(data: &[u8], mut f: impl FnMut(u8) -> bool) -> Vec<u64> {
        let mut masks = vec![0u64; data.len().div_ceil(64)];
        for (i, &b) in data.iter().enumerate() {
            if f(b) {
                masks[i / 64] |= 1 << (i % 64);
            }
        }
        masks
    }

    /// The canonical rediscovery: in-string context from a serial
    /// quote-parity machine must come back as a single PrefixXor.
    #[test]
    fn rediscovers_prefix_xor_for_quote_parity() {
        let leaves = [Leaf::class("Q", b"\"")];
        let corpus = corpus(b"\"x,", 0xDEAD_BEEF, 4, 2);
        let reference = |data: &[u8]| {
            let mut parity = false;
            mask_ref(data, |b| {
                if b == b'"' {
                    parity = !parity;
                }
                parity
            })
        };
        match synthesize(&leaves, &corpus, &Spec::exact(&reference), &budget(4)) {
            Outcome::Found(sol) => {
                assert_eq!(sol.expr, "PrefixXor(Q)");
                assert_eq!(sol.tree_size, 2);
            }
            Outcome::NotFound(stats) => panic!("not found: {stats:?}"),
        }
    }

    /// Line starts need the seeded shift (position 0 counts), not the
    /// plain one — the synthesizer must tell them apart.
    #[test]
    fn rediscovers_seeded_shift_for_line_starts() {
        let leaves = [Leaf::class("N", b"\n")];
        let corpus = corpus(b"\nxy", 0xFACE_FEED, 4, 2);
        let reference = |data: &[u8]| {
            let mut at_start = true;
            mask_ref(data, |b| {
                let out = at_start;
                at_start = b == b'\n';
                out
            })
        };
        match synthesize(&leaves, &corpus, &Spec::exact(&reference), &budget(4)) {
            Outcome::Found(sol) => assert_eq!(sol.expr, "Shl1Seeded(N)"),
            Outcome::NotFound(stats) => panic!("not found: {stats:?}"),
        }
    }

    /// The auto driver must behave like plain synthesis when the target is
    /// directly reachable: found in round 1, no promotions made.
    #[test]
    fn auto_finds_direct_targets_in_round_one() {
        let leaves = [Leaf::class("Q", b"\"")];
        let corpus = corpus(b"\"x,", 0xDEAD_BEEF, 4, 2);
        let reference = |data: &[u8]| {
            let mut parity = false;
            mask_ref(data, |b| {
                if b == b'"' {
                    parity = !parity;
                }
                parity
            })
        };
        let auto = AutoBudget {
            rounds: 2,
            per_round: budget(4),
            promotions: 4,
            max_leaves: 8,
        };
        match synthesize_auto(&leaves, &corpus, &Spec::exact(&reference), &auto) {
            AutoOutcome::Found(sol, reports) => {
                assert_eq!(sol.expr, "PrefixXor(Q)");
                assert!(reports.is_empty(), "no promotion round should have run");
            }
            AutoOutcome::NotFound(reports) => panic!("not found: {reports:?}"),
        }
    }

    /// Cost-ordered enumeration must surface the cheapest care form
    /// directly — no settling through expensive equivalents. The known
    /// cheapest escape form under quote-only care costs 7 (avx2 model).
    #[test]
    fn cost_order_finds_cheapest_care_form_directly() {
        let leaves =
            [Leaf::class("B", b"\\"), Leaf::constant("EVEN", 0x5555_5555_5555_5555)];
        let mut inputs = corpus(b"\\x\"", 0x1357_9BDF_2468_ACE0, 6, 2);
        inputs.push(vec![b'\\'; 128]);
        let reference = |data: &[u8]| {
            let mut run_odd = false;
            mask_ref(data, |b| {
                if b == b'\\' {
                    run_odd = !run_odd;
                    false
                } else {
                    let out = run_odd;
                    run_odd = false;
                    out
                }
            })
        };
        let care = |data: &[u8]| mask_ref(data, |b| b == b'"');
        let budget = Budget {
            max_level: 9,
            max_candidates: 20_000_000,
            max_bank: 1_000_000,
            settle_levels: 0,
            cost: CostModel::avx2(),
            order: Order::Cost,
            progress: false,
        };
        match synthesize(&leaves, &inputs, &Spec::with_care(&reference, &care), &budget) {
            Outcome::Found(sol) => {
                assert!(sol.cost <= 7, "expected cost <= 7, got {} ({})", sol.cost, sol.expr);
            }
            Outcome::NotFound(stats) => panic!("not found: {stats:?}"),
        }
    }

    /// Multi-output: the second spec should be expressed via the first
    /// (`O0`), and the merged graph must be cheaper than separate ones.
    #[test]
    fn multi_output_reuses_earlier_streams() {
        let leaves = [Leaf::class("Q", b"\"")];
        let corpus = corpus(b"\"x,", 0xBADC_0FFE, 4, 2);
        let inside_ref = |data: &[u8]| {
            let mut inside = false;
            mask_ref(data, |b| {
                if b == b'"' {
                    inside = !inside;
                }
                inside
            })
        };
        let outside_ref = |data: &[u8]| {
            let mut inside = false;
            mask_ref(data, |b| {
                if b == b'"' {
                    inside = !inside;
                }
                !inside
            })
        };
        match synthesize_multi(
            &corpus,
            &[
                MultiSpec { leaves: &leaves, spec: Spec::exact(&inside_ref) },
                MultiSpec { leaves: &[], spec: Spec::exact(&outside_ref) },
            ],
            &budget(4),
        ) {
            MultiOutcome::Found(multi) => {
                assert_eq!(multi.exprs[0], "PrefixXor(Q)");
                assert_eq!(multi.exprs[1], "Not(O0)");
                assert_eq!(multi.outputs.len(), 2);
                assert!(multi.shared_cost < multi.separate_cost);
            }
            MultiOutcome::NotFound { failed_spec, stats } => {
                panic!("spec {failed_spec} not found: {stats:?}")
            }
        }
    }

    /// The prover's byte-serial executor and the blockwise interpreter are
    /// two implementations of one semantics; they must agree bit-for-bit,
    /// carries across block seams included.
    #[test]
    fn byte_serial_executor_matches_interp() {
        // A graph exercising every supported op kind.
        let mut g = Graph::new();
        let b = g.class_byte(b'\\');
        let nb = g.not(b);
        let seeded = g.shift_left1_seeded(nb);
        let starts = g.and(b, seeded);
        let even = g.constant(0x5555_5555_5555_5555);
        let es = g.and(starts, even);
        let carries = g.add(b, es);
        let toggled = g.xor(carries, even);
        let shifted = g.shift_left1(b);
        let f = g.and(nb, shifted);
        let gated = g.and(f, toggled);
        let parity = g.prefix_xor(b);
        let out = g.or(gated, parity);
        g.set_output(out);

        let mut rng = Rng(0xA5A5_5A5A_A5A5_5A5A);
        for _ in 0..500 {
            let len = (rng.next() % 300) as usize;
            let input: Vec<u8> =
                (0..len).map(|_| [b'\\', b'x', b'"'][(rng.next() % 3) as usize]).collect();
            let serial = byte_serial_masks(&g, &input);
            let mut positions = Vec::new();
            interp::run(&g, &input, &mut positions);
            let mut blockwise = vec![0u64; input.len().div_ceil(64)];
            for p in positions {
                blockwise[(p / 64) as usize] |= 1 << (p % 64);
            }
            assert_eq!(serial, blockwise, "executor divergence on {input:?}");
        }
    }

    /// PrefixXor(Q) against the two-state quote-parity machine: a complete
    /// proof, not a sample.
    #[test]
    fn proves_quote_parity_for_all_inputs() {
        let mut g = Graph::new();
        let q = g.class_byte(b'"');
        let parity = g.prefix_xor(q);
        g.set_output(parity);
        let step = |state: u32, byte: u8| -> (u32, bool) {
            let next = if byte == b'"' { state ^ 1 } else { state };
            (next, next == 1)
        };
        match prove(&g, &Fsm { initial: 0, step: &step }) {
            ProveOutcome::Proven(proof) => assert!(proof.product_states <= 256),
            ProveOutcome::Refuted(witness) => panic!("refuted by {witness:?}"),
            ProveOutcome::Aborted { explored } => panic!("aborted at {explored}"),
        }
    }

    /// A wrong graph must be refuted with a concrete shortest witness.
    #[test]
    fn prover_refutes_wrong_graphs() {
        let mut g = Graph::new();
        let q = g.class_byte(b'"');
        let shifted = g.shift_left1(q); // not parity: wrong on the quote itself
        g.set_output(shifted);
        let step = |state: u32, byte: u8| -> (u32, bool) {
            let next = if byte == b'"' { state ^ 1 } else { state };
            (next, next == 1)
        };
        match prove(&g, &Fsm { initial: 0, step: &step }) {
            ProveOutcome::Refuted(witness) => {
                assert!(!witness.is_empty() && witness.len() <= 2, "witness {witness:?}")
            }
            ProveOutcome::Proven(_) => panic!("wrong graph proven equal"),
            ProveOutcome::Aborted { explored } => panic!("aborted at {explored}"),
        }
    }

    /// CR-before-LF needs one byte of lookahead; every IR op is causal, so
    /// no graph of any size computes it. The search must come up empty.
    #[test]
    fn lookahead_is_unsynthesizable() {
        let leaves = [Leaf::class("CR", b"\r"), Leaf::class("LF", b"\n")];
        let corpus = corpus(b"\r\nx", 0xCAFE_F00D, 4, 2);
        let reference = |data: &[u8]| {
            let mut masks = vec![0u64; data.len().div_ceil(64)];
            for i in 0..data.len().saturating_sub(1) {
                if data[i] == b'\r' && data[i + 1] == b'\n' {
                    masks[i / 64] |= 1 << (i % 64);
                }
            }
            masks
        };
        match synthesize(&leaves, &corpus, &Spec::exact(&reference), &budget(5)) {
            Outcome::Found(sol) => panic!("impossible target was 'solved': {}", sol.expr),
            Outcome::NotFound(stats) => assert_eq!(stats.completed_level, 5),
        }
    }
}
