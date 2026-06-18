//! Python bindings for falx's fused `.vcf.gz` → Apache Arrow pipeline.
//!
//! `read_vcf_gz_columns(path)` decompresses and parses a bgzipped VCF straight
//! into typed Arrow columns (POS/QUAL/DP/AF) using `bgzf::parse_gz_par` — block-
//! parallel, no full-file decompressed buffer — and returns a `pyarrow.RecordBatch`
//! that pandas/polars/duckdb consume zero-copy.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::IntoPyArrow;
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use falx::bgzf;
use falx::kernels::vcf_typed;

/// Owned, mergeable view of the numeric typed columns from one parsed chunk.
#[derive(Default)]
struct Cols {
    pos: Vec<i64>,
    quality: Vec<f64>,
    quality_v: Vec<bool>,
    dp: Vec<i64>,
    dp_v: Vec<bool>,
    af: Vec<f64>,
    af_v: Vec<bool>,
}

/// Parse one record-aligned chunk into owned columns (the fusion driver drops
/// the decompressed buffer right after, so nothing may borrow it).
fn parse_chunk(s: &[u8]) -> Cols {
    let c = vcf_typed::parse_columns(s);
    let bits = |bm: &[u64]| (0..c.rows).map(|r| vcf_typed::bitmap_get(bm, r)).collect();
    Cols {
        pos: c.pos.clone(),
        quality: c.quality.clone(),
        quality_v: bits(&c.quality_valid),
        dp: c.dp.clone(),
        dp_v: bits(&c.dp_valid),
        af: c.af.clone(),
        af_v: bits(&c.af_valid),
    }
}

fn merge(parts: Vec<Cols>) -> Cols {
    let mut m = Cols::default();
    for p in parts {
        m.pos.extend(p.pos);
        m.quality.extend(p.quality);
        m.quality_v.extend(p.quality_v);
        m.dp.extend(p.dp);
        m.dp_v.extend(p.dp_v);
        m.af.extend(p.af);
        m.af_v.extend(p.af_v);
    }
    m
}

fn build_batch(c: Cols) -> Result<RecordBatch, arrow::error::ArrowError> {
    let opt_i = |vals: Vec<i64>, v: Vec<bool>| {
        Int64Array::from_iter(vals.into_iter().zip(v).map(|(x, ok)| ok.then_some(x)))
    };
    let opt_f = |vals: Vec<f64>, v: Vec<bool>| {
        Float64Array::from_iter(vals.into_iter().zip(v).map(|(x, ok)| ok.then_some(x)))
    };
    let schema = Schema::new(vec![
        Field::new("pos", DataType::Int64, false),
        Field::new("quality", DataType::Float64, true),
        Field::new("dp", DataType::Int64, true),
        Field::new("af", DataType::Float64, true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from(c.pos)) as ArrayRef,
            Arc::new(opt_f(c.quality, c.quality_v)),
            Arc::new(opt_i(c.dp, c.dp_v)),
            Arc::new(opt_f(c.af, c.af_v)),
        ],
    )
}

/// Read a bgzipped VCF and return its typed columns as a `pyarrow.RecordBatch`.
///
/// Columns: `pos` (int64), `quality`/`dp`/`af` (nullable, null where the field
/// or INFO key is absent). Parsing runs with the GIL released.
#[pyfunction]
#[pyo3(signature = (path, threads = 0))]
fn read_vcf_gz_columns<'py>(
    py: Python<'py>,
    path: String,
    threads: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let comp = std::fs::read(&path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(8, |n| n.get())
    } else {
        threads
    };
    let batch = py
        .detach(|| -> Result<RecordBatch, String> {
            let parts = bgzf::parse_gz_par(&comp, threads, b'\n', parse_chunk)
                .map_err(|e| e.to_string())?;
            build_batch(merge(parts)).map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)?;
    batch.into_pyarrow(py)
}

#[pymodule]
fn falx_genomics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_vcf_gz_columns, m)?)?;
    Ok(())
}
