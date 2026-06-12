use falx::codegen::{self, CodegenOptions, GraphSource};
use falx::formats;
use falx::interp;
use falx::ir::{Graph, NodeId};
use falx::synth_formats::{self, SynthProfile};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

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
fn weighted_synth_codegen_emits_native_simd_without_fallback() {
    let code = codegen::emit_parser_with_columns_options(
        &formats::csv_dialect(),
        "csv_synth_test",
        &[],
        CodegenOptions {
            graph_source: GraphSource::SynthWeighted(SynthProfile::Fast),
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
    };

    let synthesized =
        synth_formats::synthesize_delimited_parts_with_profile(&dialect, SynthProfile::Fast)
            .expect("large structural class should synthesize without corpus panic");

    assert_eq!(
        run_graph(&synthesized.graph, &[1, 32, 33, b'\n']),
        vec![0, 1, 3]
    );
}
