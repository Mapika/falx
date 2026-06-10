use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let spec_path = "spec.toml";
    println!("cargo::rerun-if-changed={}", spec_path);

    let spec_content = fs::read_to_string(spec_path)
        .expect("Failed to read spec.toml");

    let spec = falx::spec::parse(&spec_content)
        .expect("Failed to parse spec.toml");

    let generated = falx::codegen::emit_parser(&spec.dialect, &spec.name)
        .expect("Failed to generate parser");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("parser.rs");

    fs::write(&out_path, generated)
        .expect("Failed to write generated parser");
}
