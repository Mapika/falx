//! Does the e-graph's advantage over the two-candidate optimizer grow with
//! graph complexity? Generate random bitstream circuits of increasing size,
//! optimize each with both `graph_opt` (cost-weighted two-candidate) and
//! `egraph` (equality saturation), and curve the cost advantage vs graph size.
//!
//! Every generated graph is differentially checked on random inputs: a cheaper
//! graph that computes something different is a bug, not a win.
//!
//! `cargo run --release --example egraph_scaling [--smoke]`

use falx::formats::DelimitedParts;
use falx::ir::{Graph, NodeId};
use falx::synth::CostModel;
use falx::{egraph, graph_opt, interp};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick(&mut self, v: &[NodeId]) -> NodeId {
        v[self.below(v.len() as u64) as usize]
    }
}

/// Build a connected random circuit with roughly `internal` internal nodes over
/// a small leaf pool. The small pool forces operand reuse (CSE / factoring
/// opportunities); inline `Not`s on operands seed Not-extraction, De Morgan and
/// the speculative/conservative mixing the e-graph exploits.
fn random_parts(seed: u64, internal: usize) -> DelimitedParts {
    let mut rng = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut g = Graph::new();
    let mut pool: Vec<NodeId> = Vec::new();
    for b in [b',', b';', b'|', b'\t', b':', b' '] {
        pool.push(g.class_byte(b));
    }
    pool.push(g.constant(0x5555_5555_5555_5555)); // even-position mask
    pool.push(g.constant(0xAAAA_AAAA_AAAA_AAAA)); // odd-position mask

    let mut acc = rng.pick(&pool);
    for _ in 0..internal {
        let other_raw = rng.pick(&pool);
        let other = if rng.below(100) < 30 {
            g.not(other_raw)
        } else {
            other_raw
        };
        acc = match rng.below(100) {
            0..25 => g.not(acc),
            25..50 => g.and(acc, other),
            50..75 => g.or(acc, other),
            _ => g.xor(acc, other),
        };
        pool.push(acc);
    }
    let mut parts = DelimitedParts {
        graph: g,
        terminators: pool[0],
        nest: None,
    };
    parts.graph.set_output(acc);
    parts
}

fn run_out(graph: &Graph, data: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    interp::run(graph, data, &mut out);
    out
}

/// Differential check: optimized output stream must match the original on every
/// probe input. Returns true if equivalent.
fn equivalent(original: &Graph, optimized: &Graph) -> bool {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..6 {
        let len = 64 * (1 + rng.below(5)) as usize;
        let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        if run_out(original, &data) != run_out(optimized, &data) {
            return false;
        }
    }
    true
}

struct Row {
    nodes: usize,
    cw_cost: f64,
    es_cost: f64,
    advantage: f64,
    cw_us: f64,
    es_us: f64,
    mismatches: usize,
}

fn sweep(sizes: &[usize], k: u64) -> Vec<Row> {
    let model = CostModel::avx2();
    let mut rows = Vec::new();
    for &internal in sizes {
        let (mut nodes, mut cw_sum, mut es_sum, mut adv_sum) = (0usize, 0.0, 0.0, 0.0);
        let (mut cw_t, mut es_t) = (0.0, 0.0);
        let mut mismatches = 0;
        for seed in 0..k {
            let parts = random_parts(seed, internal);

            let t = Instant::now();
            let cw = graph_opt::optimize_parts(parts.clone(), model);
            cw_t += t.elapsed().as_micros() as f64;

            let t = Instant::now();
            let es = egraph::optimize_parts(parts.clone(), model);
            es_t += t.elapsed().as_micros() as f64;

            nodes += es.stats.original_nodes;

            if !equivalent(&parts.graph, &es.parts.graph)
                || !equivalent(&parts.graph, &cw.parts.graph)
            {
                mismatches += 1;
            }

            let (cwc, esc) = (
                cw.stats.optimized_cost as f64,
                es.stats.optimized_cost as f64,
            );
            cw_sum += cwc;
            es_sum += esc;
            adv_sum += if cwc > 0.0 {
                (cwc - esc) / cwc * 100.0
            } else {
                0.0
            };
        }
        let kf = k as f64;
        rows.push(Row {
            nodes: nodes / k as usize,
            cw_cost: cw_sum / kf,
            es_cost: es_sum / kf,
            advantage: adv_sum / kf,
            cw_us: cw_t / kf,
            es_us: es_t / kf,
            mismatches,
        });
    }
    rows
}

fn main() {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let (sizes, k): (Vec<usize>, u64) = if smoke {
        (vec![8, 16, 32], 4)
    } else {
        // Past ~300 nodes the cheap-phase De Morgan/not-extraction expansion can
        // hit the e-node cap on dense random circuits (incomplete saturation) —
        // far beyond any real kernel, so the default sweep stops at 256.
        (vec![8, 16, 32, 48, 64, 96, 128, 192, 256], 16)
    };

    let rows = sweep(&sizes, k);

    println!(
        "{:>6}  {:>8}  {:>8}  {:>9}  {:>9}  {:>9}  {:>5}",
        "nodes", "cw_cost", "es_cost", "adv// %", "cw_us", "es_us", "mism"
    );
    for r in &rows {
        println!(
            "{:>6}  {:>8.1}  {:>8.1}  {:>+8.2}%  {:>9.1}  {:>9.1}  {:>5}",
            r.nodes, r.cw_cost, r.es_cost, r.advantage, r.cw_us, r.es_us, r.mismatches
        );
    }

    let max_adv = rows.iter().map(|r| r.advantage).fold(0.0_f64, f64::max);
    let scale = if max_adv > 0.0 { 50.0 / max_adv } else { 0.0 };
    println!("\ne-graph cost advantage vs two-candidate optimizer (bar = advantage%):");
    for r in &rows {
        let bar = "#".repeat((r.advantage * scale).round().max(0.0) as usize);
        println!("  n={:>4} | {:<50} {:+.2}%", r.nodes, bar, r.advantage);
    }
    let total_mism: usize = rows.iter().map(|r| r.mismatches).sum();
    println!(
        "\nsoundness: {} differential mismatches across all graphs",
        total_mism
    );
}
