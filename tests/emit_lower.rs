//! Milestone 1 of emitter↔`codegen` parity: the typed-AST `lower`ing emits a
//! scalar structural indexer for any dialect graph. We check its shape, then
//! actually **compile the generated Rust with `rustc` and run it**, asserting
//! byte-identical structural indices to the reference interpreter
//! (`interp::run`) across every built-in dialect — and, for CSV, to the crate's
//! generated production kernel (`falx::kernels::csv`, the synth+codegen output).
//! That last check is the concrete "new emitter matches the checked-in kernel" proof.

use falx::emit::{emit_c, emit_rust, lower};
use falx::ir::Graph;
use falx::{formats, interp};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

mod common;
use common::Rng;

fn dialects() -> Vec<(&'static str, formats::Dialect)> {
    vec![
        ("csv", formats::csv_dialect()),
        ("tsv", formats::tsv_dialect()),
        ("lines", formats::lines_dialect()),
        ("ndjson", formats::ndjson_dialect()),
        ("logfmt", formats::logfmt_dialect()),
        ("vcf", formats::vcf_dialect()),
        ("json", formats::json_dialect()),
        ("csv_hash", formats::csv_hash_dialect()),
        ("multi", formats::multi_dialect()),
    ]
}

#[test]
fn lines_indexer_has_no_quote_machinery() {
    let src = emit_rust(&lower::lower_indexer(&formats::lines_dialect())).unwrap();
    assert!(src.contains("fn class_mask"));
    assert!(!src.contains("fn prefix_xor"), "lines has no quotes");
    assert!(src.contains("pub fn index_structurals(data: &[u8], out: &mut Vec<u32>)"));
}

#[test]
fn csv_indexer_has_expected_shape() {
    let src = emit_rust(&lower::lower_indexer(&formats::csv_dialect())).unwrap();
    assert!(src.contains("fn class_mask"));
    assert!(src.contains("fn prefix_xor"));
    assert!(src.contains("class_mask(block, &["));
    assert!(src.contains("while offset + 64 <= data.len()"));
    // The output drain is the unrolled `push_indexes` scatter, not per-bit push.
    assert!(src.contains("fn push_indexes"));
    assert!(
        src.contains("offset as u32, out)"),
        "push_indexes drain call"
    );
    assert!(!src.contains("out.push(offset as u32 + mask.trailing_zeros());"));
    // M2: the class_mask seam is SIMD (AVX2) with a scalar fallback.
    assert!(src.contains("_mm256_cmpeq_epi8"), "AVX2 classify emitted");
    assert!(
        src.contains("fn class_mask_scalar"),
        "scalar fallback emitted"
    );
    assert!(src.contains("#[target_feature(enable = \"avx2\")]"));
    assert!(
        src.contains("_mm_clmulepi64_si128"),
        "PCLMULQDQ parity emitted"
    );
    assert!(
        src.contains("_mm512_cmpeq_epi8_mask"),
        "AVX-512 classify emitted"
    );
    assert!(src.contains("vceqq_u8"), "NEON classify emitted");
    // Visible with `cargo test --test emit_lower -- --nocapture`.
    eprintln!("\n===== generated csv index_structurals =====\n{src}");
}

/// One program: shared helpers + an `index_<dialect>` per dialect + a `main`
/// that dispatches on argv[1], reads bytes from stdin, prints the indices.
fn build_program() -> String {
    let ds = dialects();
    let graphs: Vec<Graph> = ds.iter().map(|(_, d)| formats::delimited(d)).collect();
    let refs: Vec<&Graph> = graphs.iter().collect();
    let mut items = lower::needed_helpers(&refs);
    for ((name, _), g) in ds.iter().zip(&graphs) {
        items.extend(lower::rust_index_items(g, &format!("index_{name}")));
    }
    let kernels = emit_rust(&items).unwrap();
    let arms: String = ds
        .iter()
        .map(|(name, _)| format!("\"{name}\" => index_{name}(&data, &mut out),\n"))
        .collect();
    format!(
        "{kernels}\n\
         fn main() {{\n\
         use std::io::Read;\n\
         let which = std::env::args().nth(1).unwrap();\n\
         let mut data = Vec::new();\n\
         std::io::stdin().read_to_end(&mut data).unwrap();\n\
         let mut out: Vec<u32> = Vec::new();\n\
         match which.as_str() {{\n\
         {arms}_ => panic!(\"unknown dialect\"),\n\
         }}\n\
         let s: Vec<String> = out.iter().map(|x| x.to_string()).collect();\n\
         println!(\"{{}}\", s.join(\",\"));\n\
         }}\n"
    )
}

fn compile(program: &str, stem: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("falx_emit_lower");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(format!("{stem}.rs"));
    std::fs::write(&src, program).unwrap();
    let bin = dir.join(if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    });
    let out = Command::new("rustc")
        .args(["-O", "--edition", "2021", "--crate-name", stem])
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("failed to spawn rustc");
    assert!(
        out.status.success(),
        "rustc failed to compile the generated kernel:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    bin
}

fn run(bin: &Path, which: &str, input: &[u8]) -> Vec<u32> {
    let mut child = Command::new(bin)
        .arg(which)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    let s = s.trim();
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(|x| x.parse().unwrap()).collect()
    }
}

fn test_inputs() -> Vec<Vec<u8>> {
    // Structural bytes for every dialect, plus quotes, escapes, comments.
    let alphabet: &[u8] = b"\",\n\t #{}[]:=|;/&\\abx\"";
    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    let mut inputs = vec![
        Vec::new(),
        b"a,b,c\n".to_vec(),
        b"\"x,y\",z\n".to_vec(),
        b"# comment, not split\nreal,row\n".to_vec(),
        b"{\"k\": [1, 2], \"s\": \"a,b\"}\n".to_vec(),
        b"a\\\"b\",c\n".to_vec(),
    ];
    // Block-seam stressers (carry bugs live at the 64-byte boundary).
    for len in [1usize, 63, 64, 65, 127, 128, 129, 256] {
        inputs.push(random_bytes(&mut rng, alphabet, len));
    }
    for _ in 0..24 {
        let len = (rng.next() % 320) as usize;
        inputs.push(random_bytes(&mut rng, alphabet, len));
    }
    inputs
}

fn random_bytes(rng: &mut Rng, alphabet: &[u8], len: usize) -> Vec<u8> {
    (0..len)
        .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
        .collect()
}

#[test]
fn lowered_indexers_match_interp_and_simd() {
    // Skip gracefully where a standalone rustc isn't on PATH; CI always has it.
    if Command::new("rustc").arg("--version").output().is_err() {
        eprintln!("skipping: no `rustc` on PATH");
        return;
    }

    let bin = compile(&build_program(), "index");
    let inputs = test_inputs();

    // Every dialect: generated scalar kernel == reference interpreter.
    for (name, d) in dialects() {
        let graph = formats::delimited(&d);
        for input in &inputs {
            let got = run(&bin, name, input);
            let mut want = Vec::new();
            interp::run(&graph, input, &mut want);
            assert_eq!(
                got,
                want,
                "dialect {name}: generated kernel != interp on {:?}",
                String::from_utf8_lossy(input)
            );
        }
    }

    // CSV: typed-AST scalar kernel == the crate's GENERATED production kernel
    // (synth+codegen, drift-locked) — the concrete emitter↔checked-in-kernel check.
    for input in &inputs {
        let got = run(&bin, "csv", input);
        let mut production = Vec::new();
        falx::kernels::csv::index_structurals(input, &mut production);
        assert_eq!(
            got,
            production,
            "csv: typed-AST kernel != generated kernels::csv on {:?}",
            String::from_utf8_lossy(input)
        );
    }
}

#[cfg(target_arch = "x86_64")]
fn run_variant(bin: &Path, name: &str, variant: &str, input: &[u8]) -> Vec<u32> {
    let mut child = Command::new(bin)
        .arg(name)
        .arg(variant)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    let s = s.trim();
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(|x| x.parse().unwrap()).collect()
    }
}

/// Each baked x86 whole-loop variant, called *directly*, must match the
/// interpreter. The runtime dispatch only exercises the best path, so this pins
/// the others (e.g. AVX2 on an AVX-512 box, which the dispatcher would skip).
/// Variants are gated on feature detection so we never execute an unsupported one.
#[test]
fn lowered_baked_x86_variants_match_interp() {
    #[cfg(not(target_arch = "x86_64"))]
    eprintln!("skipping: baked x86 variants are x86_64-only");
    #[cfg(target_arch = "x86_64")]
    baked_x86_variants_check();
}

/// The x86 body of [`lowered_baked_x86_variants_match_interp`], isolated so the
/// `is_x86_feature_detected!` calls are only *compiled* on x86_64 (the macro
/// does not exist on other targets).
#[cfg(target_arch = "x86_64")]
fn baked_x86_variants_check() {
    if Command::new("rustc").arg("--version").output().is_err() {
        eprintln!("skipping: no `rustc` on PATH");
        return;
    }
    // Non-Regions dialects, with and without quote machinery.
    let ds: Vec<(&str, formats::Dialect)> = vec![
        ("csv", formats::csv_dialect()),
        ("lines", formats::lines_dialect()),
        ("ndjson", formats::ndjson_dialect()),
        ("json", formats::json_dialect()),
    ];
    let graphs: Vec<Graph> = ds.iter().map(|(_, d)| formats::delimited(d)).collect();
    let refs: Vec<&Graph> = graphs.iter().collect();
    let mut items = lower::needed_helpers(&refs);
    for ((name, _), g) in ds.iter().zip(&graphs) {
        items.extend(lower::rust_index_items(g, &format!("index_{name}")));
    }
    let kernels = emit_rust(&items).unwrap();
    let arms: String = ds
        .iter()
        .flat_map(|(name, _)| {
            [
                format!(
                    "(\"{name}\", \"avx512\") => unsafe {{ index_{name}_avx512(&data, &mut out) }},\n"
                ),
                format!(
                    "(\"{name}\", \"avx2\") => unsafe {{ index_{name}_avx2(&data, &mut out) }},\n"
                ),
                format!("(\"{name}\", \"portable\") => index_{name}_portable(&data, &mut out),\n"),
            ]
        })
        .collect();
    let program = format!(
        "{kernels}\n\
         fn main() {{\n\
         use std::io::Read;\n\
         let name = std::env::args().nth(1).unwrap();\n\
         let variant = std::env::args().nth(2).unwrap();\n\
         let mut data = Vec::new();\n\
         std::io::stdin().read_to_end(&mut data).unwrap();\n\
         let mut out: Vec<u32> = Vec::new();\n\
         match (name.as_str(), variant.as_str()) {{\n\
         {arms}_ => panic!(\"unknown\"),\n\
         }}\n\
         let s: Vec<String> = out.iter().map(|x| x.to_string()).collect();\n\
         println!(\"{{}}\", s.join(\",\"));\n\
         }}\n"
    );
    let bin = compile(&program, "baked_variants");
    let inputs = test_inputs();

    // Only run variants this CPU actually supports.
    let mut variants: Vec<&str> = vec!["portable"];
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("pclmulqdq") {
        variants.push("avx2");
    }
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("pclmulqdq")
    {
        variants.push("avx512");
    }

    for (name, d) in &ds {
        let graph = formats::delimited(d);
        for input in &inputs {
            let mut want = Vec::new();
            interp::run(&graph, input, &mut want);
            for variant in &variants {
                let got = run_variant(&bin, name, variant, input);
                assert_eq!(
                    got,
                    want,
                    "{name}/{variant}: baked kernel != interp on {:?}",
                    String::from_utf8_lossy(input)
                );
            }
        }
    }
}

/// A CSV `parse` program: it prints, per record, the `offset:len` of each raw
/// field span (one record per line) so the test can compare spans exactly.
fn build_parse_program() -> String {
    let g = formats::delimited(&formats::csv_dialect());
    let mut items = lower::needed_helpers(&[&g]);
    items.push(lower::index_function(&g, "index_structurals"));
    items.push(lower::parse_function("index_structurals"));
    let kernels = emit_rust(&items).unwrap();
    format!(
        "{kernels}\n\
         fn main() {{\n\
         use std::io::Read;\n\
         let mut data = Vec::new();\n\
         std::io::stdin().read_to_end(&mut data).unwrap();\n\
         let recs = parse(&data);\n\
         let base = data.as_ptr() as usize;\n\
         let mut out = String::new();\n\
         for rec in &recs {{\n\
         let parts: Vec<String> = rec.iter().map(|f| format!(\"{{}}:{{}}\", f.as_ptr() as usize - base, f.len())).collect();\n\
         out.push_str(&parts.join(\",\"));\n\
         out.push('\\n');\n\
         }}\n\
         print!(\"{{}}\", out);\n\
         }}\n"
    )
}

fn run_parse(bin: &Path, input: &[u8]) -> Vec<Vec<(usize, usize)>> {
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            line.split(',')
                .map(|p| {
                    let (o, l) = p.split_once(':').unwrap();
                    (o.parse().unwrap(), l.parse().unwrap())
                })
                .collect()
        })
        .collect()
}

#[test]
fn lowered_csv_parser_matches_production_spans() {
    if Command::new("rustc").arg("--version").output().is_err() {
        eprintln!("skipping: no `rustc` on PATH");
        return;
    }
    let bin = compile(&build_parse_program(), "parse");
    for input in &test_inputs() {
        let got = run_parse(&bin, input);
        let parsed = falx::kernels::csv::parse(input);
        let want: Vec<Vec<(usize, usize)>> = parsed
            .records()
            .map(|rec| {
                (0..rec.field_count())
                    .map(|i| {
                        let (from, to) = rec.field_span(i).unwrap();
                        (from as usize, (to - from) as usize)
                    })
                    .collect()
            })
            .collect();
        assert_eq!(
            got,
            want,
            "csv parse: typed-AST spans != kernels::csv on {:?}",
            String::from_utf8_lossy(input)
        );
    }
}

/// A `column_i64` program: prints one line per record — the i64 value of field
/// `argv[1]`, or `none`.
fn build_columns_program() -> String {
    let g = formats::delimited(&formats::csv_dialect());
    let mut items = lower::needed_helpers(&[&g]);
    items.push(lower::index_function(&g, "index_structurals"));
    items.push(lower::parse_function("index_structurals"));
    items.push(lower::parse_i64_helper());
    items.push(lower::column_i64_function());
    let kernels = emit_rust(&items).unwrap();
    format!(
        "{kernels}\n\
         fn main() {{\n\
         use std::io::Read;\n\
         let col: usize = std::env::args().nth(1).unwrap().parse().unwrap();\n\
         let mut data = Vec::new();\n\
         std::io::stdin().read_to_end(&mut data).unwrap();\n\
         let vals = column_i64(&data, col);\n\
         let mut out = String::new();\n\
         for v in &vals {{\n\
         match v {{ Some(x) => out.push_str(&x.to_string()), None => out.push_str(\"none\") }}\n\
         out.push('\\n');\n\
         }}\n\
         print!(\"{{}}\", out);\n\
         }}\n"
    )
}

fn run_column(bin: &Path, col: usize, input: &[u8]) -> Vec<Option<i64>> {
    let mut child = Command::new(bin)
        .arg(col.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| {
            if l == "none" {
                None
            } else {
                Some(l.parse().unwrap())
            }
        })
        .collect()
}

#[test]
fn lowered_csv_column_i64_matches_production() {
    if Command::new("rustc").arg("--version").output().is_err() {
        eprintln!("skipping: no `rustc` on PATH");
        return;
    }
    let bin = compile(&build_columns_program(), "columns");
    let col = 1usize;
    let mut inputs = test_inputs();
    inputs.push(b"a,42,b\nc,-7,d\nx,,y\nq,999999999999,z\n".to_vec());
    inputs.push(b"1,2\n3,4\n5,6".to_vec());
    inputs.push(b"only_one_field\n".to_vec());
    for input in &inputs {
        let got = run_column(&bin, col, input);
        let parsed = falx::kernels::csv::parse(input);
        let want: Vec<Option<i64>> = parsed
            .records()
            .map(|rec| {
                if col < rec.field_count() {
                    let raw = rec.field_raw(col).unwrap();
                    std::str::from_utf8(raw)
                        .ok()
                        .and_then(|s| s.parse::<i64>().ok())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            got,
            want,
            "column_i64: typed-AST != production on {:?}",
            String::from_utf8_lossy(input)
        );
    }
}

fn compile_c(program: &str, stem: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("falx_emit_lower");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(format!("{stem}.c"));
    std::fs::write(&src, program).unwrap();
    let bin = dir.join(if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    });
    let out = Command::new("cc")
        .args(["-O2", "-std=c11"])
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("failed to spawn cc");
    assert!(
        out.status.success(),
        "cc failed to compile the generated CUDA-C kernel:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    bin
}

fn build_c_index_program(dialect: &formats::Dialect) -> String {
    let c = emit_c(&lower::lower_index_c(dialect)).unwrap();
    format!(
        "#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n\
         {c}\n\
         int main(void) {{\n\
         size_t cap = 1024, len = 0;\n\
         uint8_t* data = (uint8_t*)malloc(cap);\n\
         int ch;\n\
         while ((ch = getchar()) != EOF) {{ if (len == cap) {{ cap *= 2; data = (uint8_t*)realloc(data, cap); }} data[len++] = (uint8_t)ch; }}\n\
         uint32_t* out = (uint32_t*)malloc((len + 1) * sizeof(uint32_t));\n\
         size_t out_count = 0;\n\
         index_structurals(data, len, out, &out_count);\n\
         for (size_t i = 0; i < out_count; i++) {{ if (i) putchar(','); printf(\"%u\", out[i]); }}\n\
         putchar('\\n');\n\
         return 0;\n\
         }}\n"
    )
}

/// M5: the typed AST lowers the structural index to CUDA-C; the `cc`-compiled
/// kernel matches the reference interpreter across dialects.
#[test]
fn lowered_index_c_matches_interp() {
    if Command::new("cc").arg("--version").output().is_err() {
        eprintln!("skipping: no `cc` on PATH");
        return;
    }
    let dialects: [(&str, formats::Dialect); 6] = [
        ("csv", formats::csv_dialect()),
        ("tsv", formats::tsv_dialect()),
        ("lines", formats::lines_dialect()),
        ("ndjson", formats::ndjson_dialect()),
        ("json", formats::json_dialect()),
        ("csv_hash", formats::csv_hash_dialect()),
    ];
    let inputs = test_inputs();
    for (name, d) in &dialects {
        let g = formats::delimited(d);
        let bin = compile_c(&build_c_index_program(d), &format!("c_{name}"));
        for input in &inputs {
            let got = run(&bin, "x", input);
            let mut want = Vec::new();
            interp::run(&g, input, &mut want);
            assert_eq!(
                got,
                want,
                "CUDA-C index {name} != interp on {:?}",
                String::from_utf8_lossy(input)
            );
        }
    }
}

/// A JSON `parse_nested` program: prints `open:close` of each matched bracket
/// pair, comma-separated.
fn build_nested_program() -> String {
    let g = formats::delimited(&formats::json_dialect());
    let mut items = lower::needed_helpers(&[&g]);
    items.push(lower::index_function(&g, "index_structurals"));
    items.push(lower::nested_function("index_structurals"));
    let kernels = emit_rust(&items).unwrap();
    format!(
        "{kernels}\n\
         fn main() {{\n\
         use std::io::Read;\n\
         let mut data = Vec::new();\n\
         std::io::stdin().read_to_end(&mut data).unwrap();\n\
         let pairs = parse_nested(&data);\n\
         let s: Vec<String> = pairs.iter().map(|(o, c)| format!(\"{{}}:{{}}\", o, c)).collect();\n\
         println!(\"{{}}\", s.join(\",\"));\n\
         }}\n"
    )
}

fn run_pairs(bin: &Path, input: &[u8]) -> Vec<(usize, usize)> {
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    let s = s.trim();
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',')
            .map(|p| {
                let (o, c) = p.split_once(':').unwrap();
                (o.parse().unwrap(), c.parse().unwrap())
            })
            .collect()
    }
}

/// M6: the typed AST lowers a JSON nested-bracket tape; the rustc-compiled
/// `parse_nested` matches a stack-matcher over the trusted structural index.
#[test]
fn lowered_json_nested_matches_reference() {
    if Command::new("rustc").arg("--version").output().is_err() {
        eprintln!("skipping: no `rustc` on PATH");
        return;
    }
    let bin = compile(&build_nested_program(), "nested");
    let g = formats::delimited(&formats::json_dialect());
    for input in &test_inputs() {
        let got = run_pairs(&bin, input);
        let mut idx = Vec::new();
        interp::run(&g, input, &mut idx);
        let mut want = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        for &pp in &idx {
            let p = pp as usize;
            let b = input[p];
            if b == b'{' || b == b'[' {
                stack.push(p);
            } else if (b == b'}' || b == b']') && !stack.is_empty() {
                let o = stack.pop().unwrap();
                want.push((o, p));
            }
        }
        assert_eq!(
            got,
            want,
            "parse_nested: typed-AST != reference on {:?}",
            String::from_utf8_lossy(input)
        );
    }
}

/// A stats program: `argv[1]` selects `fastq_read_count` (newlines/4) or
/// `logfmt_pair_count` (`=` count); prints the count.
fn build_stats_program() -> String {
    let gf = formats::delimited(&formats::fastq_dialect());
    let gl = formats::delimited(&formats::logfmt_dialect());
    let mut items = lower::needed_helpers(&[&gf, &gl]);
    items.push(lower::index_function(&gf, "index_fastq"));
    items.push(lower::index_function(&gl, "index_logfmt"));
    items.push(lower::count_structural_function(
        "index_fastq",
        "b'\\n'",
        4,
        "fastq_read_count",
    ));
    items.push(lower::count_structural_function(
        "index_logfmt",
        "b'='",
        1,
        "logfmt_pair_count",
    ));
    let kernels = emit_rust(&items).unwrap();
    format!(
        "{kernels}\n\
         fn main() {{\n\
         use std::io::Read;\n\
         let which = std::env::args().nth(1).unwrap();\n\
         let mut data = Vec::new();\n\
         std::io::stdin().read_to_end(&mut data).unwrap();\n\
         let n = match which.as_str() {{ \"fastq\" => fastq_read_count(&data), \"logfmt\" => logfmt_pair_count(&data), _ => panic!() }};\n\
         println!(\"{{}}\", n);\n\
         }}\n"
    )
}

fn run_count(bin: &Path, which: &str, input: &[u8]) -> usize {
    let mut child = Command::new(bin)
        .arg(which)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

/// M7: stat sinks. The rustc-compiled `fastq_read_count` / `logfmt_pair_count`
/// match counts over the trusted structural index.
#[test]
fn lowered_stat_sinks_match_reference() {
    if Command::new("rustc").arg("--version").output().is_err() {
        eprintln!("skipping: no `rustc` on PATH");
        return;
    }
    let bin = compile(&build_stats_program(), "stats");
    let gf = formats::delimited(&formats::fastq_dialect());
    let gl = formats::delimited(&formats::logfmt_dialect());
    for input in &test_inputs() {
        let mut fi = Vec::new();
        interp::run(&gf, input, &mut fi);
        let want_reads = fi.iter().filter(|&&p| input[p as usize] == b'\n').count() / 4;
        assert_eq!(
            run_count(&bin, "fastq", input),
            want_reads,
            "fastq_read_count on {:?}",
            String::from_utf8_lossy(input)
        );

        let mut li = Vec::new();
        interp::run(&gl, input, &mut li);
        let want_pairs = li.iter().filter(|&&p| input[p as usize] == b'=').count();
        assert_eq!(
            run_count(&bin, "logfmt", input),
            want_pairs,
            "logfmt_pair_count on {:?}",
            String::from_utf8_lossy(input)
        );
    }
}
