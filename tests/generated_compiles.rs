//! Compile generated parsers with `rustc`.
//!
//! Every other test inspects generated code as a *string* — it can tell that an
//! item is present, not that the file is valid Rust. That gap is real: an
//! emitted template with a syntax error passes every content check and reaches
//! users intact. (It has happened: `RECORD_TERMINATOR` was once emitted as a
//! literal newline inside a byte literal, caught only by compiling the output.)
//!
//! The in-tree kernels are already covered, because they live in `src/kernels/`
//! and the crate builds them — the drift test then ties them to what codegen
//! currently emits. What that misses is everything with no checked-in kernel,
//! which today means **framed** modules: the container layer, and the
//! record-aware driver that comes with it. Those are what this compiles.
//!
//! Generated parsers are meant to be std-only and dependency-free, so each is
//! compiled as its own crate with warnings denied — that also pins the
//! "generated code is warning-clean" property a consumer would otherwise
//! discover for us.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Compile `source` as a standalone crate. Returns rustc's stderr on failure.
fn compile(name: &str, source: &str, crate_type: &str, extra: &[&str]) -> Result<PathBuf, String> {
    // CARGO_TARGET_TMPDIR is cargo's per-test-binary scratch directory.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let src = dir.join(format!("{name}.rs"));
    std::fs::write(&src, source).expect("write generated source");
    let out = dir.join(format!("{name}.out"));

    let output = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .arg(&src)
        .args(["--crate-type", crate_type])
        .args(["--edition", "2021"])
        // Generated code must be clean, not merely valid.
        .args(["-D", "warnings"])
        .arg("-o")
        .arg(&out)
        .args(extra)
        .output()
        .expect("failed to run rustc — it must be on PATH");

    if output.status.success() {
        Ok(out)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

fn emit(ir: &str) -> String {
    let module = falx::ir_text::parse(ir).expect("IR should parse");
    falx::codegen::emit_module(&module).expect("emit should succeed")
}

/// Every shape of framed module compiles as a standalone, dependency-free
/// crate with warnings denied.
#[test]
fn framed_modules_compile_standalone() {
    let cases: &[(&str, &str)] = &[
        (
            // Fixed-width, payload-counted, with typed columns.
            "framed_columns",
            "falx-ir 1
format framed_columns
structural 2c,0a
quote 22
escape none
frame header=4 length-at=0 width=u32 endian=be counts=payload adjust=0 trailer=0
column 0 id i64
column 1 label string
column 2 score f64
%0 = class 0a
%1 = class 2c,0a
%2 = class 22
%3 = prefix-xor %2
%4 = not %3
%5 = and %1 %4
output %5
terminators %0
",
        ),
        (
            // bgzf-shaped: total-counted with adjust, trailer, magic, ISIZE.
            "framed_bgzf",
            "falx-ir 1
format framed_bgzf
structural 0a
frame header=18 length-at=16 width=u16 endian=le counts=total adjust=1 trailer=8 magic=0:1f,8b uncompressed=-4:u32:le skip-empty=true
%0 = class 0a
output %0
terminators %0
",
        ),
        (
            // Varint lengths (protobuf-style length-delimited).
            "framed_varint",
            "falx-ir 1
format framed_varint
structural 09,0a
frame header=0 length-at=0 width=varint endian=le counts=payload adjust=0 trailer=0
%0 = class 0a
%1 = class 09,0a
output %1
terminators %0
",
        ),
        (
            // Grouped-line format: the record-aware driver is deliberately
            // omitted, so this pins that the *remainder* still compiles.
            "framed_grouped",
            "falx-ir 1
format framed_grouped
structural 0a
lines-per-record 4
frame header=4 length-at=0 width=u32 endian=be counts=payload adjust=0 trailer=0
%0 = class 0a
output %0
terminators %0
",
        ),
        (
            // Nesting plus framing, so the nested-tape API is emitted too.
            "framed_nested",
            "falx-ir 1
format framed_nested
structural 2c,3a,0a,7b,7d,5b,5d
quote 22
escape backslash 5c
nesting 7b:7d,5b:5d
frame header=4 length-at=0 width=u32 endian=be counts=payload adjust=0 trailer=0
%0 = class 0a
%1 = class 2c,3a,0a,7b,7d,5b,5d
%2 = class 22
%3 = prefix-xor %2
%4 = not %3
%5 = and %1 %4
%6 = class 7b,5b
%7 = and %6 %5
%8 = class 7d,5d
%9 = and %8 %5
output %5
terminators %0
nest-open %7
nest-close %9
",
        ),
    ];

    for (name, ir) in cases {
        let code = emit(ir);
        if let Err(stderr) = compile(name, &code, "lib", &[]) {
            panic!("generated `{name}` does not compile:\n{stderr}");
        }
    }
}

/// The generated record-aware driver is not just syntactically valid — it
/// reassembles records that straddle frames, running as a standalone binary
/// with no dependency on falx.
#[test]
fn generated_record_driver_runs_standalone() {
    let code = emit(
        "falx-ir 1
format framed_csv
structural 2c,0a
frame header=4 length-at=0 width=u32 endian=be counts=payload adjust=0 trailer=0
column 0 id i64
%0 = class 0a
%1 = class 2c,0a
output %1
terminators %0
",
    );

    // The generated module, plus a driver that frames records at deliberately
    // record-misaligned offsets and checks they come back whole and in order.
    let program = format!(
        r####"{code}

fn main() {{
    let mut payload = Vec::new();
    for row in 0..2_000u64 {{
        payload.extend_from_slice(format!("{{row}},label{{row}}\n").as_bytes());
    }}
    for chunk_size in [3usize, 11, 97, 1024] {{
        let mut data = Vec::new();
        for chunk in payload.chunks(chunk_size) {{
            data.extend_from_slice(&(chunk.len() as u32).to_be_bytes());
            data.extend_from_slice(chunk);
        }}
        let frames = scan_frames(&data).expect("scan");
        for threads in [1usize, 2, 7, 16] {{
            let states: Vec<Vec<i64>> = parse_records_par(
                &data,
                &frames,
                threads,
                Vec::new,
                || |f: &Frame, d: &[u8], out: &mut Vec<u8>| -> Result<(), ()> {{
                    out.extend_from_slice(&d[f.payload.clone()]);
                    Ok(())
                }},
                |state: &mut Vec<i64>, records: &[u8]| {{
                    let cols = parse_columns(records);
                    for r in 0..cols.rows {{
                        state.push(cols.id[r]);
                    }}
                }},
            )
            .expect("parse");
            let flat: Vec<i64> = states.concat();
            let expected: Vec<i64> = (0..2_000).collect();
            assert_eq!(flat, expected, "chunk {{chunk_size}}, {{threads}} threads");
        }}
    }}
}}
"####
    );

    let binary = match compile("framed_runner", &program, "bin", &[]) {
        Ok(path) => path,
        Err(stderr) => panic!("generated record driver does not compile:\n{stderr}"),
    };
    let run = Command::new(&binary)
        .output()
        .expect("run generated binary");
    assert!(
        run.status.success(),
        "generated record driver failed at runtime:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
}
