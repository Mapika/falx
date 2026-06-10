use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

use falx::codegen;
use falx::spec;

fn print_usage(prog: &str) {
    eprintln!("Usage: {} build <spec.toml> [-o <output.rs>]", prog);
}

fn main() {
    let mut args = env::args();
    let prog = args.next().unwrap_or_else(|| "falx".to_string());

    let subcommand = match args.next() {
        Some(s) => s,
        None => {
            print_usage(&prog);
            process::exit(2);
        }
    };

    if subcommand != "build" {
        eprintln!("Error: unknown subcommand '{}'", subcommand);
        print_usage(&prog);
        process::exit(2);
    }

    let spec_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Error: missing argument <spec.toml>");
            print_usage(&prog);
            process::exit(2);
        }
    };

    let mut output_path: Option<String> = None;
    let mut remaining = args.collect::<Vec<_>>();
    while !remaining.is_empty() {
        match remaining[0].as_str() {
            "-o" => {
                if remaining.len() < 2 {
                    eprintln!("Error: -o requires an argument");
                    print_usage(&prog);
                    process::exit(2);
                }
                output_path = Some(remaining[1].clone());
                remaining = remaining.drain(2..).collect();
            }
            other => {
                eprintln!("Error: unknown argument '{}'", other);
                print_usage(&prog);
                process::exit(2);
            }
        }
    }

    // Read the spec file.
    let spec_content = match fs::read_to_string(&spec_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading '{}': {}", spec_path, e);
            process::exit(1);
        }
    };

    // Parse the spec.
    let spec = match spec::parse(&spec_content) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error parsing spec: {}", e);
            process::exit(1);
        }
    };

    // Generate the full parser (indexer + span API).
    let generated = match codegen::emit_parser(&spec.dialect, &spec.name) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error generating code: {}", e);
            process::exit(1);
        }
    };

    // Write output.
    if let Some(out_path) = output_path {
        if let Err(e) = fs::write(&out_path, &generated) {
            eprintln!("Error writing to '{}': {}", out_path, e);
            process::exit(1);
        }
    } else {
        if let Err(e) = io::stdout().write_all(generated.as_bytes()) {
            eprintln!("Error writing to stdout: {}", e);
            process::exit(1);
        }
    }
}
