//! Report what the cost-weighted graph optimizer does to every built-in
//! format, for both graph sources codegen can draw from: the manual
//! `formats::delimited_parts()` builder and the weighted synthesizer.
//!
//! ```text
//! cargo run --release --example opt_probe
//! ```

use falx::formats::{self, Dialect};
use falx::graph_opt::optimize_parts;
use falx::synth::CostModel;
use falx::synth_formats::{self, SynthProfile};

fn probe(name: &str, dialect: &Dialect) {
    let manual = formats::delimited_parts(dialect);
    let m = optimize_parts(manual, CostModel::avx2()).stats;
    print!(
        "{name:10} manual: nodes {:3} -> {:3} cost {:4} -> {:4} applied={}",
        m.original_nodes, m.optimized_nodes, m.original_cost, m.optimized_cost, m.applied
    );
    if synth_formats::supports_weighted(dialect) {
        let synth =
            synth_formats::synthesize_delimited_parts_with_profile(dialect, SynthProfile::Weighted)
                .expect("supported dialect must synthesize");
        let s = optimize_parts(synth, CostModel::avx2()).stats;
        println!(
            " | synth: nodes {:3} -> {:3} cost {:4} -> {:4} applied={}",
            s.original_nodes, s.optimized_nodes, s.original_cost, s.optimized_cost, s.applied
        );
    } else {
        println!(" | synth: unsupported");
    }
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
