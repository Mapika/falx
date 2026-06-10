//! Checked-in kernels produced by the code generator — regenerate with
//! `cargo run --example generate`. A drift test asserts these files match
//! what current codegen emits.

pub mod csv;
pub mod logfmt;
pub mod ndjson;
pub mod tsv;
