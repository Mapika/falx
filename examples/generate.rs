//! Regenerate the checked-in kernels in src/kernels/ from their dialects.
//! Run after changing the IR, codegen, or a format definition:
//! `cargo run --example generate`
//!
//! To opt into weighted synthesized graphs for supported targets:
//! `cargo run --example generate -- --synth weighted`

use falx::codegen::{self, CodegenOptions, GraphSource};
use falx::kernels;
use falx::synth_formats::{self, SynthProfile};

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
    let mut mode = GenerateMode::Manual;
    let mut args = args.into_iter().map(Into::into).skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
                return Err("usage: generate [--synth weighted]".into());
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(mode)
}

fn main() {
    let mode = match parse_mode(std::env::args()) {
        Ok(mode) => mode,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    for (name, dialect, columns) in kernels::targets() {
        let (options, source) =
            if mode == GenerateMode::SynthWeighted && synth_formats::supports_weighted(&dialect) {
                (
                    CodegenOptions {
                        graph_source: GraphSource::SynthWeighted(SynthProfile::Weighted),
                    },
                    "synth-weighted",
                )
            } else {
                (CodegenOptions::default(), "manual")
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
    fn parse_defaults_to_manual() {
        assert_eq!(parse_mode(["generate"]).unwrap(), GenerateMode::Manual);
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
}
