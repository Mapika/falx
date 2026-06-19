use falx::codegen::{self, CodegenOptions, GraphOptimizer, GraphSource};
use falx::formats;
use falx::interp;
use falx::ir::{Graph, NodeId};
use falx::synth_formats::{self, SynthProfile};

mod common;
use common::Rng;

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

fn cases(alphabet: &[u8]) -> Vec<Vec<u8>> {
    let mut rng = Rng(0xA17E_5A11_D15C_0DED);
    let mut cases = vec![
        Vec::new(),
        b"a,b,c\n".to_vec(),
        b"\"a,b\",c\n".to_vec(),
        b"\"multi\nline\",x\n".to_vec(),
        b"\t\t\nx\ty\n".to_vec(),
    ];
    for _ in 0..64 {
        let len = (rng.next() % 256) as usize;
        cases.push(
            (0..len)
                .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                .collect(),
        );
    }
    cases
}

#[test]
fn weighted_synth_csv_and_tsv_match_manual_graphs() {
    let formats = [
        (formats::csv_dialect(), b"\",\nxy".as_slice()),
        (formats::tsv_dialect(), b"\t\nxy".as_slice()),
    ];

    for (dialect, alphabet) in formats {
        let manual = formats::delimited_parts(&dialect);
        let synthesized =
            synth_formats::synthesize_delimited_parts_with_profile(&dialect, SynthProfile::Fast)
                .expect("fast weighted synthesis should solve this dialect");

        for data in cases(alphabet) {
            assert_eq!(
                run_graph(&synthesized.graph, &data),
                run_graph(&manual.graph, &data),
                "structural stream diverged for input {data:?}"
            );
            assert_eq!(
                run_node(&synthesized.graph, synthesized.terminators, &data),
                run_node(&manual.graph, manual.terminators, &data),
                "raw terminator stream diverged for input {data:?}"
            );
        }
    }
}

#[test]
fn weighted_synth_json_nesting_matches_manual_outputs() {
    let dialect = formats::json_dialect();
    let manual = formats::delimited_parts(&dialect);
    let synthesized =
        synth_formats::synthesize_delimited_parts_with_profile(&dialect, SynthProfile::Fast)
            .expect("fast weighted synthesis should solve JSON nesting");
    let (manual_opens, manual_closes) = manual.nest.expect("JSON has nesting nodes");
    let (synth_opens, synth_closes) = synthesized.nest.expect("JSON has nesting nodes");

    let cases = [
        b"{}".as_slice(),
        b"{\"a\":[1,2,{\"b\":3}]}",
        b"{\"quoted\":\"[not structural]\",\"escaped\":\"\\\"}\"}",
        b"[\n{\"a\":1}, {\"b\":[true,false,null]}]",
        b"{\"multi\\nline\":\"x\", \"arr\":[{\"k\":\"v\"}]}",
    ];

    for data in cases {
        assert_eq!(
            run_graph(&synthesized.graph, data),
            run_graph(&manual.graph, data),
            "JSON structural stream diverged for input {data:?}"
        );
        assert_eq!(
            run_node(&synthesized.graph, synthesized.terminators, data),
            run_node(&manual.graph, manual.terminators, data),
            "JSON raw terminator stream diverged for input {data:?}"
        );
        assert_eq!(
            run_node(&synthesized.graph, synth_opens, data),
            run_node(&manual.graph, manual_opens, data),
            "JSON live open stream diverged for input {data:?}"
        );
        assert_eq!(
            run_node(&synthesized.graph, synth_closes, data),
            run_node(&manual.graph, manual_closes, data),
            "JSON live close stream diverged for input {data:?}"
        );
    }
}

#[test]
fn weighted_synth_codegen_emits_native_simd_without_fallback() {
    let code = codegen::emit_parser_with_columns_options(
        &formats::csv_dialect(),
        "csv_synth_test",
        &[],
        CodegenOptions {
            graph_source: GraphSource::SynthWeighted(SynthProfile::Fast),
            ..CodegenOptions::default()
        },
    )
    .expect("synth-weighted codegen should succeed");

    assert!(!code.contains("pub mod fallback"));
    assert!(!code.contains("fallback::"));
    assert!(code.contains("mod avx512"));
    assert!(
        code.find("avx512::").expect("AVX-512 dispatch present")
            < code.find("avx2::").expect("AVX2 dispatch present")
    );
}

#[test]
fn codegen_default_uses_weighted_when_supported() {
    let dialect = formats::csv_dialect();
    let default = codegen::emit_parser_with_columns(&dialect, "csv_default_test", &[])
        .expect("default codegen should succeed");
    let weighted = codegen::emit_parser_with_columns_options(
        &dialect,
        "csv_default_test",
        &[],
        CodegenOptions {
            graph_source: GraphSource::SynthWeighted(SynthProfile::Weighted),
            ..CodegenOptions::default()
        },
    )
    .expect("weighted codegen should succeed");

    assert_eq!(default, weighted);
}

#[test]
fn codegen_default_uses_manual_for_unsupported_dialects() {
    let dialect = formats::csv_hash_dialect();
    let columns = [
        codegen::Column {
            index: 0,
            name: Some("key".into()),
            ty: codegen::ColumnType::Str,
            info_key: None,
        },
        codegen::Column {
            index: 1,
            name: Some("amount".into()),
            ty: codegen::ColumnType::I64,
            info_key: None,
        },
    ];
    let default = codegen::emit_parser_with_columns(&dialect, "csv_hash_default_test", &columns)
        .expect("default codegen should succeed");
    let manual = codegen::emit_parser_with_columns_options(
        &dialect,
        "csv_hash_default_test",
        &columns,
        CodegenOptions {
            graph_source: GraphSource::Manual,
            ..CodegenOptions::default()
        },
    )
    .expect("manual codegen should succeed");

    assert_eq!(default, manual);
}

#[test]
fn codegen_default_uses_eqsat_graph_optimizer() {
    let dialect = formats::csv_dialect();
    let default = codegen::emit_parser_with_columns(&dialect, "csv_opt_default_test", &[])
        .expect("default codegen should succeed");
    let explicit = codegen::emit_parser_with_columns_options(
        &dialect,
        "csv_opt_default_test",
        &[],
        CodegenOptions {
            graph_optimizer: GraphOptimizer::EqSat,
            ..CodegenOptions::default()
        },
    )
    .expect("explicit optimizer codegen should succeed");

    assert_eq!(default, explicit);
}

#[test]
fn disabled_graph_optimizer_still_emits_native_simd_without_fallback() {
    let code = codegen::emit_parser_with_columns_options(
        &formats::csv_dialect(),
        "csv_unoptimized_test",
        &[],
        CodegenOptions {
            graph_optimizer: GraphOptimizer::Disabled,
            ..CodegenOptions::default()
        },
    )
    .expect("unoptimized codegen should succeed");

    assert!(!code.contains("pub mod fallback"));
    assert!(!code.contains("fallback::"));
    assert!(code.contains("mod avx512"));
    assert!(
        code.find("avx512::").expect("AVX-512 dispatch present")
            < code.find("avx2::").expect("AVX2 dispatch present")
    );
}

#[test]
fn weighted_synth_rejects_comment_region_dialects() {
    let dialect = formats::csv_hash_dialect();
    assert!(!synth_formats::supports_weighted(&dialect));

    let err = match synth_formats::synthesize_delimited_parts_with_profile(
        &dialect,
        SynthProfile::Fast,
    ) {
        Ok(_) => panic!("comment regions are not synth-supported yet"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("comment"));
}

#[test]
fn weighted_synth_rejects_quote_escape_conflicts() {
    let dialect = formats::Dialect {
        structural: vec![b',', b'\n'],
        quote: Some(b'\\'),
        escape: formats::Escape::Backslash(b'\\'),
        comment: None,
        nesting: vec![],
        lines_per_record: 1,
    };

    assert!(!synth_formats::supports_weighted(&dialect));

    let err = match synth_formats::synthesize_delimited_parts_with_profile(
        &dialect,
        SynthProfile::Fast,
    ) {
        Ok(_) => panic!("quote/escape conflicts are not synth-supported"),
        Err(err) => err,
    };
    let message = err.to_string();
    assert!(message.contains("quote"));
    assert!(message.contains("escape"));
    assert!(message.contains("conflict"));
}

#[test]
fn weighted_synth_handles_large_structural_sets_in_boundary_corpus() {
    let dialect = formats::Dialect {
        structural: (1..=32).collect(),
        quote: None,
        escape: formats::Escape::None,
        comment: None,
        nesting: vec![],
        lines_per_record: 1,
    };

    let synthesized =
        synth_formats::synthesize_delimited_parts_with_profile(&dialect, SynthProfile::Fast)
            .expect("large structural class should synthesize without corpus panic");

    assert_eq!(
        run_graph(&synthesized.graph, &[1, 32, 33, b'\n']),
        vec![0, 1, 3]
    );
}

#[test]
fn weighted_synth_rejects_invalid_nesting_dialects() {
    let dialect = formats::Dialect {
        structural: vec![b'[', b']'],
        quote: None,
        escape: formats::Escape::None,
        comment: None,
        nesting: vec![(b'{', b'}')],
        lines_per_record: 1,
    };

    assert!(!synth_formats::supports_weighted(&dialect));

    let err = match synth_formats::synthesize_delimited_parts_with_profile(
        &dialect,
        SynthProfile::Fast,
    ) {
        Ok(_) => panic!("invalid nesting should not be synth-supported"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("nesting"));
}
