//! Checked-in kernels produced by the code generator — regenerate with
//! `cargo run --example generate`. A drift test asserts these files match
//! what current codegen emits.

use crate::codegen::{Column, ColumnType};
use crate::formats::{self, Dialect};

pub mod csv;
pub mod csv_geo;
pub mod csv_geo_text;
pub mod csv_typed;
pub mod logfmt;
pub mod multi;
pub mod ndjson;
pub mod tsv;

/// The registry of checked-in kernels: name, dialect, and projected
/// columns. The generator example and the drift test both consume this, so
/// the files on disk can never silently diverge from how they were made.
pub fn targets() -> Vec<(&'static str, Dialect, Vec<Column>)> {
    vec![
        ("csv", formats::csv_dialect(), vec![]),
        ("tsv", formats::tsv_dialect(), vec![]),
        ("logfmt", formats::logfmt_dialect(), vec![]),
        ("ndjson", formats::ndjson_dialect(), vec![]),
        // 9 structural bytes: classified via PSHUFB nibble tables.
        ("multi", formats::multi_dialect(), vec![]),
        // Typed projection demo: non-adjacent indexes, one column of every
        // type, so tests exercise skipped fields between requested ones.
        (
            "csv_typed",
            formats::csv_dialect(),
            vec![
                Column { index: 0, name: Some("id".into()), ty: ColumnType::I64 },
                Column { index: 1, name: Some("title".into()), ty: ColumnType::Str },
                Column { index: 2, name: Some("value".into()), ty: ColumnType::F64 },
                Column { index: 4, name: Some("label".into()), ty: ColumnType::Bytes },
            ],
        ),
        // worldcitiespop schema (Country,City,AccentCity,Region,Population,
        // Latitude,Longitude): the two-typed-columns benchmark kernel.
        (
            "csv_geo",
            formats::csv_dialect(),
            vec![
                Column { index: 5, name: Some("latitude".into()), ty: ColumnType::F64 },
                Column { index: 6, name: Some("longitude".into()), ty: ColumnType::F64 },
            ],
        ),
        // Same schema with the City column materialized as a string: the
        // text + numbers benchmark kernel.
        (
            "csv_geo_text",
            formats::csv_dialect(),
            vec![
                Column { index: 1, name: Some("city".into()), ty: ColumnType::Str },
                Column { index: 5, name: Some("latitude".into()), ty: ColumnType::F64 },
                Column { index: 6, name: Some("longitude".into()), ty: ColumnType::F64 },
            ],
        ),
    ]
}
