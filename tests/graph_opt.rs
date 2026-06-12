use falx::formats::{self, DelimitedParts, Dialect, Escape};
use falx::interp;
use falx::ir::{Graph, NodeId};
use falx::synth::CostModel;

fn run_graph(graph: &Graph, data: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    interp::run(graph, data, &mut out);
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
            "structural output changed for input {:?}",
            data
        );
        assert_eq!(
            run_node(&optimized.graph, optimized.terminators, &data),
            run_node(&original.graph, original.terminators, &data),
            "terminator output changed for input {:?}",
            data
        );

        match (original.nest, optimized.nest) {
            (None, None) => {}
            (Some((orig_open, orig_close)), Some((opt_open, opt_close))) => {
                assert_eq!(
                    run_node(&optimized.graph, opt_open, &data),
                    run_node(&original.graph, orig_open, &data),
                    "open-bracket output changed for input {:?}",
                    data
                );
                assert_eq!(
                    run_node(&optimized.graph, opt_close, &data),
                    run_node(&original.graph, orig_close, &data),
                    "close-bracket output changed for input {:?}",
                    data
                );
            }
            _ => panic!("nesting role changed"),
        }
    }
}

#[test]
fn optimizer_preserves_delimited_part_roles() {
    let dialects = [
        formats::csv_dialect(),
        formats::tsv_dialect(),
        formats::logfmt_dialect(),
        formats::ndjson_dialect(),
        formats::json_dialect(),
        formats::multi_dialect(),
        formats::csv_hash_dialect(),
    ];

    for dialect in dialects {
        let original = formats::delimited_parts(&dialect);
        let optimized = falx::graph_opt::optimize_parts(original.clone(), CostModel::avx2()).parts;
        assert_parts_equivalent(&original, &optimized, &dialect);
    }
}

#[test]
fn optimizer_reports_removed_nodes_for_redundant_graph() {
    let dialect = formats::csv_dialect();
    let mut parts = formats::delimited_parts(&dialect);
    let dead = parts.graph.class_byte(b'z');
    let dead_not = parts.graph.not(dead);
    let _dead_back = parts.graph.not(dead_not);
    let original_nodes = parts.graph.nodes().len();

    let optimized = falx::graph_opt::optimize_parts(parts, CostModel::avx2());

    assert!(optimized.stats.applied);
    assert!(
        optimized.stats.removed_nodes >= 3,
        "expected at least the unreachable redundant chain to be removed, stats={:?}",
        optimized.stats
    );
    assert!(optimized.parts.graph.nodes().len() < original_nodes);
}
