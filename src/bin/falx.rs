//! The falx command line: spec → IR → parser.
//!
//! The pipeline is deliberately splittable. `build` runs it end to end, but
//! `emit-ir` stops at the IR and `build-ir` starts from it, so the textual IR
//! is a real interchange point rather than an internal detail: anything that
//! can write that text reaches every backend falx has.

use std::fs;
use std::io::{self, Write};
use std::process;

use falx::codegen::{self, CodegenOptions, GraphOptimizer};
use falx::ir_text;

const USAGE: &str = "\
falx — a parser generator for high-throughput record and column parsers

USAGE:
    falx build     <spec.toml>  [-o <out.rs>]   Generate a parser from a spec
    falx emit-ir   <spec.toml>  [-o <out.fxir>] Lower a spec to textual IR
    falx opt       <module.fxir>[-o <out.fxir>] Optimize textual IR
    falx build-ir  <module.fxir>[-o <out.rs>]   Generate a parser from textual IR
    falx check     <spec.toml|module.fxir>      Validate without writing output

OPTIONS:
    -o <path>     Write to <path> instead of stdout
    --no-opt      Skip the graph optimizer (emit-ir, build)
    -h, --help    Show this help

The IR is the contract: `emit-ir` output is accepted by `build-ir`, and any
producer that emits it gets every backend without going through a spec file.
See docs/ir.md for the format.";

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("error: {}", message.as_ref());
    process::exit(1)
}

fn usage_error(message: impl AsRef<str>) -> ! {
    eprintln!("error: {}", message.as_ref());
    eprintln!("\n{USAGE}");
    process::exit(2)
}

struct Args {
    input: String,
    output: Option<String>,
    optimize: bool,
}

/// Parse `<input> [-o out] [--no-opt]` for a subcommand.
fn parse_args(mut rest: Vec<String>, command: &str) -> Args {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut optimize = true;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "-o" => {
                if i + 1 >= rest.len() {
                    usage_error("-o requires a path");
                }
                output = Some(std::mem::take(&mut rest[i + 1]));
                i += 2;
            }
            "--no-opt" => {
                optimize = false;
                i += 1;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                process::exit(0);
            }
            other if other.starts_with('-') => usage_error(format!("unknown option '{other}'")),
            _ => {
                if input.is_some() {
                    usage_error(format!("{command}: unexpected extra argument"));
                }
                input = Some(std::mem::take(&mut rest[i]));
                i += 1;
            }
        }
    }
    Args {
        input: input.unwrap_or_else(|| usage_error(format!("{command}: missing input path"))),
        output,
        optimize,
    }
}

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| fail(format!("reading '{path}': {e}")))
}

fn write_out(output: Option<String>, contents: &str) {
    match output {
        Some(path) => {
            fs::write(&path, contents).unwrap_or_else(|e| fail(format!("writing '{path}': {e}")))
        }
        None => io::stdout()
            .write_all(contents.as_bytes())
            .unwrap_or_else(|e| fail(format!("writing to stdout: {e}"))),
    }
}

/// Lower a spec file to an IR module, honouring `--no-opt`.
fn lower_spec(path: &str, optimize: bool) -> ir_text::Module {
    let spec = falx::spec::parse(&read(path))
        .unwrap_or_else(|e| fail(format!("parsing spec '{path}': {e}")));
    let options = CodegenOptions {
        graph_optimizer: if optimize {
            CodegenOptions::default().graph_optimizer
        } else {
            GraphOptimizer::Disabled
        },
        ..CodegenOptions::default()
    };
    codegen::lower(&spec.dialect, &spec.name, &spec.columns, options)
        .unwrap_or_else(|e| fail(format!("lowering '{path}': {}", e.0)))
}

fn load_ir(path: &str) -> ir_text::Module {
    ir_text::parse(&read(path)).unwrap_or_else(|e| fail(format!("reading IR '{path}': {e}")))
}

fn emit(module: &ir_text::Module, path: &str) -> String {
    codegen::emit_module(module).unwrap_or_else(|e| fail(format!("generating '{path}': {}", e.0)))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| {
        println!("{USAGE}");
        process::exit(2)
    });
    let rest: Vec<String> = args.collect();

    match command.as_str() {
        "-h" | "--help" | "help" => println!("{USAGE}"),
        "build" => {
            let args = parse_args(rest, "build");
            let module = lower_spec(&args.input, args.optimize);
            let code = emit(&module, &args.input);
            write_out(args.output, &code);
        }
        "emit-ir" => {
            let args = parse_args(rest, "emit-ir");
            let module = lower_spec(&args.input, args.optimize);
            write_out(args.output, &ir_text::print(&module));
        }
        "opt" => {
            let args = parse_args(rest, "opt");
            let module = load_ir(&args.input);
            let optimizer = if args.optimize {
                CodegenOptions::default().graph_optimizer
            } else {
                GraphOptimizer::Disabled
            };
            let optimized = codegen::optimize_module(&module, optimizer);
            eprintln!(
                "{}: {} -> {} nodes",
                optimized.name,
                module.graph.nodes().len(),
                optimized.graph.nodes().len()
            );
            write_out(args.output, &ir_text::print(&optimized));
        }
        "build-ir" => {
            let args = parse_args(rest, "build-ir");
            let module = load_ir(&args.input);
            let code = emit(&module, &args.input);
            write_out(args.output, &code);
        }
        "check" => {
            let args = parse_args(rest, "check");
            // Route by content, not extension: an IR file always starts with
            // its version header.
            let text = read(&args.input);
            let module = if text.trim_start().starts_with("falx-ir") {
                load_ir(&args.input)
            } else {
                lower_spec(&args.input, args.optimize)
            };
            // Emission is the real validator — it is where column names,
            // nesting pairs, and dialect combinations are checked.
            let code = emit(&module, &args.input);
            eprintln!(
                "ok: {} — {} nodes, {} column(s), {} bytes of generated Rust",
                module.name,
                module.graph.nodes().len(),
                module.columns.len(),
                code.len()
            );
        }
        other => usage_error(format!("unknown subcommand '{other}'")),
    }
}
