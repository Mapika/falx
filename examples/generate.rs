//! Regenerate the checked-in kernels in src/kernels/ from their dialects.
//! Run after changing the IR, codegen, or a format definition:
//! `cargo run --example generate`

use falx::{codegen, formats};

fn main() {
    let targets = [
        ("csv", formats::csv_dialect()),
        ("tsv", formats::tsv_dialect()),
        ("logfmt", formats::logfmt_dialect()),
        ("ndjson", formats::ndjson_dialect()),
    ];
    for (name, dialect) in targets {
        let code = codegen::emit_parser(&dialect, name).expect("dialect should be emittable");
        let path = format!("{}/src/kernels/{name}.rs", env!("CARGO_MANIFEST_DIR"));
        std::fs::write(&path, code).expect("write generated kernel");
        println!("wrote {path}");
    }
}
