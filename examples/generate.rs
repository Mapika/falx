//! Regenerate the checked-in kernels in src/kernels/ from their dialects.
//! Run after changing the IR, codegen, or a format definition:
//! `cargo run --example generate`

use falx::{codegen, kernels};

fn main() {
    for (name, dialect, columns) in kernels::targets() {
        let code = codegen::emit_parser_with_columns(&dialect, name, &columns)
            .expect("dialect should be emittable");
        let path = format!("{}/src/kernels/{name}.rs", env!("CARGO_MANIFEST_DIR"));
        std::fs::write(&path, code).expect("write generated kernel");
        println!("wrote {path}");
    }
}
