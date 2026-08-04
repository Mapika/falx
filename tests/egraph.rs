//! Equality-saturation optimizer (`falx::egraph`) tests: soundness against the
//! interpreter, cost parity with the two-candidate optimizer it generalizes,
//! the per-subterm mixing and factoring wins it unlocks, and determinism.

use falx::codegen::{self, CodegenOptions, GraphOptimizer};
use falx::formats::{self, DelimitedParts, Dialect, Escape};
use falx::ir::{Graph, NodeId, Op};
use falx::synth::{CostModel, graph_cost};
use falx::{egraph, graph_opt};

fn run_graph(graph: &Graph, data: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    falx::interp::run(graph, data, &mut out);
    out
}

fn run_node(graph: &Graph, node: NodeId, data: &[u8]) -> Vec<u32> {
    let mut graph = graph.clone();
    graph.set_output(node);
    run_graph(&graph, data)
}

fn corpus_for(dialect: &Dialect) -> Vec<Vec<u8>> {
    let mut alphabet = dialect.structural.clone();
    if let Some(quote) = dialect.quote {
        alphabet.extend([quote, quote]);
    }
    if let Escape::Backslash(escape) = dialect.escape {
        alphabet.extend([escape, escape, escape]);
    }
    alphabet.extend_from_slice(b"xy \t\r");

    let mut cases = vec![
        Vec::new(),
        alphabet.iter().copied().cycle().take(192).collect(),
        b"alpha,beta\n\"quo,ted\",tail\n".to_vec(),
        b"{\"a\":[1,{\"b\":\"c,d\"}],\"e\":2}".to_vec(),
        b"key=\"escaped \\\" quote\" x=y\n".to_vec(),
    ];
    cases.push((0..320).map(|i| alphabet[i % alphabet.len()]).collect());
    cases
}

/// Every output node of `optimized` must compute the same stream as the
/// corresponding node of `original` on every corpus input, and the optimizer
/// must never grow the graph.
fn assert_parts_equivalent(
    original: &DelimitedParts,
    optimized: &DelimitedParts,
    dialect: &Dialect,
) {
    assert!(
        optimized.graph.nodes().len() <= original.graph.nodes().len(),
        "optimizer should not grow the graph: original={}, optimized={}",
        original.graph.nodes().len(),
        optimized.graph.nodes().len()
    );

    for data in corpus_for(dialect) {
        assert_eq!(
            run_graph(&optimized.graph, &data),
            run_graph(&original.graph, &data),
            "structural output changed for input {data:?}"
        );
        assert_eq!(
            run_node(&optimized.graph, optimized.terminators, &data),
            run_node(&original.graph, original.terminators, &data),
            "terminator output changed for input {data:?}"
        );
        match (original.nest, optimized.nest) {
            (None, None) => {}
            (Some((o_open, o_close)), Some((n_open, n_close))) => {
                assert_eq!(
                    run_node(&optimized.graph, n_open, &data),
                    run_node(&original.graph, o_open, &data),
                    "open-bracket output changed for input {data:?}"
                );
                assert_eq!(
                    run_node(&optimized.graph, n_close, &data),
                    run_node(&original.graph, o_close, &data),
                    "close-bracket output changed for input {data:?}"
                );
            }
            _ => panic!("nesting role changed"),
        }
    }
}

fn dialects() -> Vec<Dialect> {
    vec![
        formats::csv_dialect(),
        formats::tsv_dialect(),
        formats::logfmt_dialect(),
        formats::ndjson_dialect(),
        formats::json_dialect(),
        formats::multi_dialect(),
        formats::csv_hash_dialect(),
    ]
}

#[test]
fn eqsat_preserves_delimited_part_roles() {
    for dialect in dialects() {
        let original = formats::delimited_parts(&dialect);
        let optimized = egraph::optimize_parts(original.clone(), CostModel::avx2()).parts;
        assert_parts_equivalent(&original, &optimized, &dialect);
    }
}

#[test]
fn eqsat_never_costs_more_than_cost_weighted() {
    // Equality saturation explores a superset of the two-candidate rewrites, so
    // its extracted cost can never exceed the cost-weighted optimizer's.
    for dialect in dialects() {
        let parts = formats::delimited_parts(&dialect);
        let cw = graph_opt::optimize_parts(parts.clone(), CostModel::avx2());
        let es = egraph::optimize_parts(parts, CostModel::avx2());
        assert!(
            es.stats.optimized_cost <= cw.stats.optimized_cost,
            "eqsat regressed on {:?}: cw={} eqsat={}",
            dialect.structural,
            cw.stats.optimized_cost,
            es.stats.optimized_cost
        );
    }
}

#[test]
fn eqsat_collapses_odd_backslash_escape_chain() {
    // The simdjson odd-backslash chain ends in `!(EVEN ^ !Add(..))`; the
    // Not-through-Xor rule collapses it to `EVEN ^ Add(..)`. Every
    // backslash-escape dialect carries it.
    for dialect in [
        formats::logfmt_dialect(),
        formats::ndjson_dialect(),
        formats::json_dialect(),
    ] {
        let original = formats::delimited_parts(&dialect);
        let optimized = egraph::optimize_parts(original.clone(), CostModel::avx2());
        assert!(
            optimized.stats.applied,
            "escape-chain dialect should optimize, stats={:?}",
            optimized.stats
        );
        assert!(
            optimized.stats.optimized_cost + 2 <= optimized.stats.original_cost,
            "expected at least the two collapsed Not ops, stats={:?}",
            optimized.stats
        );
        assert_parts_equivalent(&original, &optimized.parts, &dialect);
    }
}

#[test]
fn eqsat_mixes_speculative_and_conservative_per_subterm() {
    // The payoff over the global two-candidate optimizer. Subterm A is two
    // single-use classes joined by `Or` — fusion (speculative) wins. Subterm B
    // is `Xor(x, !z)` whose `!z` is shared by an outer `And`, so Not-extraction
    // (speculative) loses. The two-candidate optimizer must pick one global
    // mode: fusing A forces B's losing speculation (or vice-versa). The
    // e-graph fuses A *and* keeps B conservative, beating both whole-graph
    // candidates — so it strictly undercuts whatever the two-candidate
    // optimizer extracted.
    let dialect = formats::csv_dialect();
    let mut parts = formats::delimited_parts(&dialect);
    let g = &mut parts.graph;
    let a1 = g.class_byte(b'a');
    let a2 = g.class_byte(b'b');
    let sub_a = g.or(a1, a2); // fusable -> Class{a,b}
    let x = g.class_byte(b'x');
    let z = g.class_byte(b'z');
    let not_z = g.not(z);
    let xor = g.xor(x, not_z);
    let sub_b = g.and(xor, not_z); // keeps not_z shared
    let out = g.or(sub_a, sub_b);
    parts.graph.set_output(out);

    let original = parts.clone();
    let cw = graph_opt::optimize_parts(parts.clone(), CostModel::avx2());
    let es = egraph::optimize_parts(parts, CostModel::avx2());

    assert!(
        es.stats.optimized_cost < cw.stats.optimized_cost,
        "e-graph should beat the global two-candidate choice: cw={} eqsat={}",
        cw.stats.optimized_cost,
        es.stats.optimized_cost
    );
    assert_parts_equivalent(&original, &es.parts, &dialect);
}

#[test]
fn eqsat_factors_shared_subterm_that_cost_weighted_cannot() {
    // Distributive factoring `Or(And(a,b), And(a,c)) = And(a, Or(b,c))` shares
    // `a` and drops one `And`. The two-candidate optimizer has no
    // distributivity rule, so it leaves the expanded form; the e-graph factors.
    let dialect = formats::csv_dialect();
    let mut parts = formats::delimited_parts(&dialect);
    let g = &mut parts.graph;
    // ShiftLeft1 streams so class fusion can't fold the `And`s (two distinct
    // single-byte classes would intersect to the empty class and collapse).
    let a = g.class_byte(b'a');
    let a = g.shift_left1(a);
    let b = g.class_byte(b'b');
    let b = g.shift_left1(b);
    let c = g.class_byte(b'c');
    let c = g.shift_left1(c);
    let ab = g.and(a, b);
    let ac = g.and(a, c);
    let out = g.or(ab, ac);
    parts.graph.set_output(out);

    let original = parts.clone();
    let cw = graph_opt::optimize_parts(parts.clone(), CostModel::avx2());
    let es = egraph::optimize_parts(parts, CostModel::avx2());

    assert!(
        es.stats.optimized_cost < cw.stats.optimized_cost,
        "factoring should beat the two-candidate optimizer: cw={} eqsat={}",
        cw.stats.optimized_cost,
        es.stats.optimized_cost
    );
    assert_parts_equivalent(&original, &es.parts, &dialect);
}

#[test]
fn eqsat_is_deterministic() {
    // Byte-identical codegen demands a deterministic optimizer: the same input
    // must extract the identical graph every run.
    for dialect in dialects() {
        let parts = formats::delimited_parts(&dialect);
        let a = egraph::optimize_parts(parts.clone(), CostModel::avx2()).parts;
        let b = egraph::optimize_parts(parts, CostModel::avx2()).parts;
        assert_eq!(
            format!("{:?}", a.graph.nodes()),
            format!("{:?}", b.graph.nodes()),
            "non-deterministic graph for {:?}",
            dialect.structural
        );
        assert_eq!(a.terminators, b.terminators);
        assert_eq!(a.nest, b.nest);
    }
}

#[test]
fn eqsat_extraction_matches_reported_cost() {
    // The stats the optimizer reports must equal the cost of the graph it
    // actually returns.
    for dialect in dialects() {
        let parts = formats::delimited_parts(&dialect);
        let opt = egraph::optimize_parts(parts, CostModel::avx2());
        assert_eq!(
            graph_cost(&opt.parts.graph, &CostModel::avx2()),
            opt.stats.optimized_cost,
            "reported cost disagrees with returned graph for {:?}",
            dialect.structural
        );
    }
}

#[test]
fn codegen_with_eqsat_optimizer_succeeds() {
    // The EqSat optimizer is a drop-in codegen pass: it must emit a complete
    // native-SIMD parser, same as the cost-weighted path.
    let code = codegen::emit_parser_with_columns_options(
        &formats::csv_dialect(),
        "csv_eqsat_test",
        &[],
        CodegenOptions {
            graph_optimizer: GraphOptimizer::EqSat,
            ..CodegenOptions::default()
        },
    )
    .expect("eqsat codegen should succeed");
    assert!(code.contains("mod avx512"));
    assert!(!code.contains("pub mod fallback"));
}

#[test]
fn absorption_removes_the_redundant_arm() {
    // And(x, Or(x, y)) = x and Or(x, And(x, y)) = x. Neither AC-flattening nor
    // factoring covers this: both rewrite within one operator level, while
    // absorption spans two. The in-tree dialect graphs happen not to contain
    // the pattern, so it is exercised directly here.
    for outer_is_and in [true, false] {
        let mut g = Graph::new();
        let x = g.class_byte(b',');
        let y = g.class_byte(b'|');
        let inner = if outer_is_and {
            g.or(x, y)
        } else {
            g.and(x, y)
        };
        let root = if outer_is_and {
            g.and(x, inner)
        } else {
            g.or(x, inner)
        };
        g.set_output(root);
        let terminators = x;
        let parts = DelimitedParts {
            graph: g,
            terminators,
            nest: None,
        };
        let opt = egraph::optimize_parts(parts, CostModel::avx2());

        // The absorbed form is a single class node: the other operand and the
        // inner combiner are both gone.
        let live = live_node_count(&opt.parts.graph);
        assert_eq!(
            live, 1,
            "absorption should reduce the output cone to one class node, got {live} \
             (outer_is_and = {outer_is_and})"
        );
        assert!(
            matches!(
                opt.parts.graph.nodes()[opt.parts.graph.output().index()],
                Op::Class(_)
            ),
            "the surviving output node should be the absorbed class"
        );
    }
}

/// Count nodes reachable from the graph's output — the cone that actually
/// costs anything at runtime.
fn live_node_count(graph: &Graph) -> usize {
    let mut live = vec![false; graph.nodes().len()];
    let mut stack = vec![graph.output().index()];
    while let Some(i) = stack.pop() {
        if std::mem::replace(&mut live[i], true) {
            continue;
        }
        let mut push = |id: falx::ir::NodeId| stack.push(id.index());
        match graph.nodes()[i] {
            Op::Class(_) | Op::Const(_) => {}
            Op::Not(a) | Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) | Op::PrefixXor(a) => push(a),
            Op::And(a, b) | Op::Or(a, b) | Op::Xor(a, b) | Op::Add(a, b) => {
                push(a);
                push(b);
            }
            Op::Regions(a, b, c) => {
                push(a);
                push(b);
                push(c);
            }
        }
    }
    live.iter().filter(|&&l| l).count()
}
