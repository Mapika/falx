use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

use vexel::codegen;
use vexel::formats::{delimited, Dialect, Escape};

#[derive(Debug)]
struct Spec {
    name: String,
    structural: Vec<u8>,
    quote: Option<u8>,
    escape: Escape,
}

fn parse_string_as_byte(s: &str) -> Result<u8, String> {
    // Interpret a TOML string as a single byte using Rust's escape sequences.
    let bytes = s.as_bytes();

    // Handle escape sequences manually.
    let unescaped = if s.starts_with('\\') && s.len() > 1 {
        match &s[1..2] {
            "n" => b'\n',
            "t" => b'\t',
            "r" => b'\r',
            "\\" => b'\\',
            "\"" => b'"',
            _ => return Err(format!("Unknown escape sequence in string: {}", s)),
        }
    } else if bytes.len() == 1 {
        bytes[0]
    } else {
        return Err(format!(
            "String '{}' is not a single byte (interpreted as {} bytes)",
            s, bytes.len()
        ));
    };

    Ok(unescaped)
}

fn parse_spec(content: &str) -> Result<Spec, String> {
    // Parse TOML without serde: manually extract fields from toml::Value.
    let parsed: toml::Table = content
        .parse()
        .map_err(|e| format!("Failed to parse TOML: {}", e))?;

    // Extract `name` (required, string).
    let name = parsed
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: 'name' (must be a string)")?
        .to_string();

    // Extract `structural` (required, array of strings).
    let structural_arr = parsed
        .get("structural")
        .and_then(|v| v.as_array())
        .ok_or("Missing required field: 'structural' (must be an array of strings)")?;

    let mut structural = Vec::new();
    for (idx, item) in structural_arr.iter().enumerate() {
        let s = item
            .as_str()
            .ok_or(format!(
                "structural[{}]: expected string, got {}",
                idx,
                value_type_name(item)
            ))?;
        let byte = parse_string_as_byte(s).map_err(|e| {
            format!("structural[{}]: {}", idx, e)
        })?;
        structural.push(byte);
    }

    // Extract `quote` (optional, string, single byte).
    let quote = if let Some(q_val) = parsed.get("quote") {
        Some(
            q_val
                .as_str()
                .ok_or("'quote' must be a string")?
                .parse::<u8>()
                .map_err(|_| {
                    let s = q_val.as_str().unwrap();
                    parse_string_as_byte(s).map(|b| b)
                })
                .or_else(|_| {
                    let s = q_val.as_str().unwrap();
                    parse_string_as_byte(s)
                })?,
        )
    } else {
        None
    };

    // Extract `escape` (optional, string; default "none"; valid: "none", "doubled", "backslash").
    let escape_str = parsed
        .get("escape")
        .and_then(|v| v.as_str())
        .unwrap_or("none");

    // Extract `escape_char` (optional, string, single byte; default "\\").
    let escape_char_str = parsed
        .get("escape_char")
        .and_then(|v| v.as_str())
        .unwrap_or("\\");
    let escape_char = parse_string_as_byte(escape_char_str)
        .map_err(|e| format!("'escape_char': {}", e))?;

    let escape = match escape_str {
        "none" | "doubled" => Escape::None,
        "backslash" => Escape::Backslash(escape_char),
        other => {
            return Err(format!(
                "Invalid 'escape' value: '{}' (must be 'none', 'doubled', or 'backslash')",
                other
            ))
        }
    };

    Ok(Spec {
        name,
        structural,
        quote,
        escape,
    })
}

fn value_type_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::String(_) => "string",
        toml::Value::Integer(_) => "integer",
        toml::Value::Float(_) => "float",
        toml::Value::Boolean(_) => "boolean",
        toml::Value::Datetime(_) => "datetime",
        toml::Value::Array(_) => "array",
        toml::Value::Table(_) => "table",
    }
}

fn print_usage(prog: &str) {
    eprintln!("Usage: {} build <spec.toml> [-o <output.rs>]", prog);
}

fn main() {
    let mut args = env::args();
    let prog = args.next().unwrap_or_else(|| "vexel".to_string());

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
    let spec = match parse_spec(&spec_content) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error parsing spec: {}", e);
            process::exit(1);
        }
    };

    // Build the dialect.
    let dialect = Dialect {
        structural: spec.structural,
        quote: spec.quote,
        escape: spec.escape,
    };

    // Build the graph and generate code.
    let graph = delimited(&dialect);
    let generated = match codegen::emit(&graph, &spec.name) {
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
