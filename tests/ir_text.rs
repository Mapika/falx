//! The textual-IR contract.
//!
//! The promise is that textual IR is a lossless, stable interchange format:
//! anything falx lowers can be written out, read back, and emitted to
//! byte-identical code. That makes the IR a real target — a producer outside
//! this repo can emit it and get every backend — so these tests are the
//! contract, not incidental coverage.

use falx::codegen::{self, CodegenOptions};
use falx::ir_text;

/// Every checked-in format: lower it, print the IR, parse it back, and emit
/// from both. The generated code must be byte-identical, which means the text
/// preserved everything the backend reads.
#[test]
fn round_trip_regenerates_identical_code_for_every_format() {
    for (name, dialect, columns) in falx::kernels::targets() {
        let module = codegen::lower(&dialect, name, &columns, CodegenOptions::default())
            .expect("lowering should succeed");
        let direct = codegen::emit_module(&module).expect("emit should succeed");

        let text = ir_text::print(&module);
        let reparsed = ir_text::parse(&text)
            .unwrap_or_else(|e| panic!("{name}: parsing printed IR failed: {e}"));
        let round_tripped = codegen::emit_module(&reparsed).expect("emit should succeed");

        assert_eq!(
            direct, round_tripped,
            "{name}: code generated from re-parsed IR differs from the original"
        );
    }
}

/// Printing is deterministic and idempotent: the text of a re-parsed module is
/// the text it was parsed from. Without this, "stable format" would only mean
/// "parses back", and diffs of checked-in IR would be noise.
#[test]
fn printing_is_idempotent() {
    for (name, dialect, columns) in falx::kernels::targets() {
        let module = codegen::lower(&dialect, name, &columns, CodegenOptions::default())
            .expect("lowering should succeed");
        let once = ir_text::print(&module);
        let twice = ir_text::print(&ir_text::parse(&once).expect("parse"));
        assert_eq!(once, twice, "{name}: printing is not idempotent");
    }
}

/// The full pipeline entry point and the two-phase (lower, then emit) form
/// must agree — otherwise the IR would be a side channel rather than the
/// actual path code generation takes.
#[test]
fn lower_then_emit_matches_the_one_shot_api() {
    for (name, dialect, columns) in falx::kernels::targets() {
        let one_shot = codegen::emit_parser_with_columns(&dialect, name, &columns)
            .expect("codegen should succeed");
        let module = codegen::lower(&dialect, name, &columns, CodegenOptions::default())
            .expect("lowering should succeed");
        let two_phase = codegen::emit_module(&module).expect("emit should succeed");
        assert_eq!(
            one_shot, two_phase,
            "{name}: lower+emit diverges from emit_parser_with_columns"
        );
    }
}

/// A module written by hand — not produced by `lower` — generates a working
/// parser. This is the actual third-party story: no dialect builder, no spec
/// file, just IR text.
#[test]
fn handwritten_ir_generates_a_working_parser() {
    let text = "\
falx-ir 1
format handwritten_csv
structural 2c,0a
quote 22
escape none
%0 = class 0a
%1 = class 2c,0a
%2 = class 22
%3 = prefix-xor %2
%4 = not %3
%5 = and %1 %4
output %5
terminators %0
";
    let module = ir_text::parse(text).expect("handwritten IR should parse");
    let code = codegen::emit_module(&module).expect("emit should succeed");
    assert!(code.contains("pub fn index_structurals"));
    assert!(code.contains("pub fn parse("));
    assert!(
        code.contains("mod generated"),
        "expected a self-contained generated module"
    );
}

/// Comments and blank lines are ignored, so IR can be annotated.
#[test]
fn comments_and_blank_lines_are_ignored() {
    let text = "\
falx-ir 1
; a leading comment

format commented
structural 0a   ; trailing comment

%0 = class 0a
output %0
terminators %0
";
    let module = ir_text::parse(text).expect("annotated IR should parse");
    assert_eq!(module.name, "commented");
    assert_eq!(module.dialect.structural, vec![b'\n']);
}

/// Malformed IR is rejected with a located, actionable message rather than
/// panicking or silently producing a wrong parser.
#[test]
fn malformed_ir_is_rejected() {
    let cases: &[(&str, &str)] = &[
        ("missing header", "format x\nstructural 0a\n"),
        (
            "unknown opcode",
            "falx-ir 1\nformat x\nstructural 0a\n%0 = frobnicate\noutput %0\nterminators %0\n",
        ),
        (
            "forward reference",
            "falx-ir 1\nformat x\nstructural 0a\n%0 = not %1\noutput %0\nterminators %0\n",
        ),
        (
            "self reference",
            "falx-ir 1\nformat x\nstructural 0a\n%0 = not %0\noutput %0\nterminators %0\n",
        ),
        (
            "non-consecutive numbering",
            "falx-ir 1\nformat x\nstructural 0a\n%1 = class 0a\noutput %1\nterminators %1\n",
        ),
        (
            "bad arity",
            "falx-ir 1\nformat x\nstructural 0a\n%0 = class 0a\n%1 = and %0\noutput %1\nterminators %0\n",
        ),
        (
            "dangling output",
            "falx-ir 1\nformat x\nstructural 0a\n%0 = class 0a\noutput %7\nterminators %0\n",
        ),
        (
            "missing output",
            "falx-ir 1\nformat x\nstructural 0a\n%0 = class 0a\nterminators %0\n",
        ),
        (
            "unsupported version",
            "falx-ir 99\nformat x\nstructural 0a\n%0 = class 0a\noutput %0\nterminators %0\n",
        ),
        (
            "unknown directive",
            "falx-ir 1\nformat x\nstructural 0a\nfrobnicate 1\n%0 = class 0a\noutput %0\nterminators %0\n",
        ),
    ];
    for (label, text) in cases {
        let result = ir_text::parse(text);
        assert!(
            result.is_err(),
            "{label}: expected a parse error, got a module"
        );
    }
}

/// Byte-set syntax: ranges, singletons, and the empty class all survive a
/// round trip, including a full 256-byte class.
#[test]
fn class_syntax_round_trips() {
    let text = "\
falx-ir 1
format classes
structural 0a
%0 = class 0a
%1 = class 30-39
%2 = class 00-ff
%3 = class 09,20,7c
%4 = or %1 %3
%5 = and %4 %2
%6 = or %5 %0
output %6
terminators %0
";
    let module = ir_text::parse(text).expect("class syntax should parse");
    let printed = ir_text::print(&module);
    let reparsed = ir_text::parse(&printed).expect("reprint should parse");
    assert_eq!(
        printed,
        ir_text::print(&reparsed),
        "class syntax is not stable across a round trip"
    );
    assert!(
        printed.contains("class 00-ff"),
        "a full class should print as one range, got:\n{printed}"
    );
    assert!(
        printed.contains("class 30-39"),
        "a digit class should print as a range, got:\n{printed}"
    );
}

/// Every opcode survives a round trip. Guards against an operation being
/// added to the IR without a textual form — the failure mode that would
/// silently make the format lossy.
#[test]
fn every_opcode_round_trips() {
    let text = "\
falx-ir 1
format opcodes
structural 2c,0a
quote 22
escape backslash 5c
comment 23
%0 = class 0a
%1 = class 2c
%2 = class 22
%3 = class 5c
%4 = class 23
%5 = const 0x5555555555555555
%6 = not %2
%7 = and %1 %6
%8 = or %7 %0
%9 = xor %8 %5
%10 = shl1 %9
%11 = shl1-seeded %4
%12 = prefix-xor %2
%13 = add %3 %5
%14 = regions %12 %11 %0
%15 = and %10 %13
%16 = or %15 %14
output %16
terminators %0
";
    let module = ir_text::parse(text).expect("all opcodes should parse");
    let printed = ir_text::print(&module);
    let reparsed = ir_text::parse(&printed).expect("reprint should parse");
    assert_eq!(printed, ir_text::print(&reparsed));
    for opcode in [
        "class",
        "const",
        "not",
        "and",
        "or",
        "xor",
        "shl1 ",
        "shl1-seeded",
        "prefix-xor",
        "add",
        "regions",
    ] {
        assert!(
            printed.contains(opcode),
            "opcode `{opcode}` vanished across the round trip:\n{printed}"
        );
    }
}

/// Columns, nesting pairs, and record grouping survive the round trip — the
/// attributes that drive the span and columnar APIs, not just the graph.
#[test]
fn surface_attributes_round_trip() {
    let module = codegen::lower(
        &falx::formats::json_dialect(),
        "nested_fmt",
        &[],
        CodegenOptions::default(),
    )
    .expect("lowering should succeed");
    let text = ir_text::print(&module);
    assert!(text.contains("nesting "), "nesting pairs should print");
    assert!(
        text.contains("nest-open ") && text.contains("nest-close "),
        "nesting role nodes should print"
    );
    let reparsed = ir_text::parse(&text).expect("parse");
    assert_eq!(reparsed.dialect.nesting, module.dialect.nesting);
    assert_eq!(reparsed.nest.is_some(), module.nest.is_some());

    let fastq = codegen::lower(
        &falx::formats::fastq_dialect(),
        "grouped",
        &[],
        CodegenOptions::default(),
    )
    .expect("lowering should succeed");
    let text = ir_text::print(&fastq);
    assert!(text.contains("lines-per-record 4"));
    assert_eq!(
        ir_text::parse(&text)
            .expect("parse")
            .dialect
            .lines_per_record,
        4
    );
}
