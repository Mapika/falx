//! A/B the two graph optimizers — the cost-weighted two-candidate simplifier
//! (`graph_opt`) and equality saturation (`egraph`) — over the manual
//! `formats::delimited_parts()` builder for every built-in format.
//!
//! Equality saturation explores a superset of the two-candidate rewrites, so
//! `eqsat` cost should never exceed `cw` cost. This table is the record to
//! check before considering a codegen default flip.
//!
//! ```text
//! cargo run --release --example opt_probe
//! ```

use falx::formats::{self, Dialect};
use falx::synth::CostModel;
use falx::{egraph, graph_opt};

fn probe(name: &str, dialect: &Dialect) {
    let cw = graph_opt::optimize_parts(formats::delimited_parts(dialect), CostModel::avx2()).stats;
    let es = egraph::optimize_parts(formats::delimited_parts(dialect), CostModel::avx2()).stats;
    let flag = if es.optimized_cost <= cw.optimized_cost {
        "ok"
    } else {
        "REGRESSED"
    };
    println!(
        "{name:10} orig nodes {:3} cost {:4} | cw nodes {:3} cost {:4} applied={:5} \
         | eqsat nodes {:3} cost {:4} applied={:5} [{flag}]",
        cw.original_nodes,
        cw.original_cost,
        cw.optimized_nodes,
        cw.optimized_cost,
        cw.applied,
        es.optimized_nodes,
        es.optimized_cost,
        es.applied,
    );
}

fn main() {
    probe("csv", &formats::csv_dialect());
    probe("tsv", &formats::tsv_dialect());
    probe("logfmt", &formats::logfmt_dialect());
    probe("multi", &formats::multi_dialect());
    probe("csv_hash", &formats::csv_hash_dialect());
    probe("ndjson", &formats::ndjson_dialect());
    probe("json", &formats::json_dialect());
}
