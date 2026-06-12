//! Regenerate the checked-in kernels in src/kernels/ from their dialects.
//! Run after changing the IR, codegen, or a format definition:
//! `cargo run --example generate`
//!
//! The default uses weighted synthesized graphs for supported targets.
//! Use `cargo run --example generate -- --manual` to force handwritten graphs.

use falx::codegen::{self, CodegenOptions, GraphSource};
use falx::kernels;
use falx::synth_formats;

const USAGE: &str = "usage: generate [--manual|--synth weighted]";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerateMode {
    Manual,
    SynthWeighted,
}

fn parse_mode<I>(args: I) -> Result<GenerateMode, String>
where
    I: IntoIterator,
    I::Item: Into<String>,
{
    let mut mode = GenerateMode::SynthWeighted;
    let mut args = args.into_iter().map(Into::into).skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manual" => mode = GenerateMode::Manual,
            "--synth" => {
                let Some(value) = args.next() else {
                    return Err("--synth requires a value".into());
                };
                match value.as_str() {
                    "weighted" => mode = GenerateMode::SynthWeighted,
                    other => return Err(format!("unknown --synth value '{other}'")),
                }
            }
            "--help" | "-h" => {
                return Err(USAGE.into());
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(mode)
}

fn wants_help<I>(args: I) -> bool
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut args = args.into_iter().skip(1);
    let Some(arg) = args.next() else {
        return false;
    };
    matches!(arg.as_ref(), "--help" | "-h") && args.next().is_none()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if wants_help(&args) {
        println!("{USAGE}");
        return;
    }

    let mode = match parse_mode(args) {
        Ok(mode) => mode,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    for (name, dialect, columns) in kernels::targets() {
        let source =
            if mode == GenerateMode::SynthWeighted && synth_formats::supports_weighted(&dialect) {
                "synth-weighted"
            } else {
                "manual"
            };
        let options = match mode {
            GenerateMode::Manual => CodegenOptions {
                graph_source: GraphSource::Manual,
                ..CodegenOptions::default()
            },
            GenerateMode::SynthWeighted => CodegenOptions::default(),
        };

        let code = codegen::emit_parser_with_columns_options(&dialect, name, &columns, options)
            .expect("dialect should be emittable");
        let path = format!("{}/src/kernels/{name}.rs", env!("CARGO_MANIFEST_DIR"));
        std::fs::write(&path, code).expect("write generated kernel");
        println!("wrote {path} [{source}]");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_to_weighted_synth() {
        assert_eq!(
            parse_mode(["generate"]).unwrap(),
            GenerateMode::SynthWeighted
        );
    }

    #[test]
    fn parse_manual_mode() {
        assert_eq!(
            parse_mode(["generate", "--manual"]).unwrap(),
            GenerateMode::Manual
        );
    }

    #[test]
    fn parse_weighted_synth_mode() {
        assert_eq!(
            parse_mode(["generate", "--synth", "weighted"]).unwrap(),
            GenerateMode::SynthWeighted
        );
    }

    #[test]
    fn parse_rejects_unknown_synth_mode() {
        let err = parse_mode(["generate", "--synth", "tree"]).unwrap_err();
        assert!(err.contains("unknown --synth value"));
    }

    #[test]
    fn parse_rejects_missing_synth_value() {
        let err = parse_mode(["generate", "--synth"]).unwrap_err();
        assert!(err.contains("--synth requires a value"));
    }

    #[test]
    fn detects_help_flag() {
        assert!(wants_help(["generate", "--help"]));
        assert!(wants_help(["generate", "-h"]));
        assert!(!wants_help(["generate", "--manual"]));
        assert!(!wants_help(["generate", "--synth", "weighted"]));
        assert!(!wants_help(["generate", "--unknown", "--help"]));
    }
}
