//! The synthesizer demo: a ladder of structural-parsing subproblems, each
//! specified only as the serial byte-at-a-time machine a person would
//! naively write, solved by search over the bitstream algebra.
//!
//! The ladder ends at the simdjson odd-backslash-run escape trick — the
//! hardest hand-derived piece of falx's format graphs — and at a target
//! that is provably out of the algebra's reach (one byte of lookahead),
//! where exhaustive search corroborates the impossibility argument.
//!
//! Run with: cargo run --release --example synth_demo

use falx::interp;
use falx::ir::Graph;
use falx::synth::{
    AutoBudget, AutoOutcome, Budget, CostModel, Fsm, Leaf, MultiOutcome, Order, Outcome,
    ProveOutcome, Solution, Spec, prove, synthesize, synthesize_auto, synthesize_multi,
};

const EVEN: u64 = 0x5555_5555_5555_5555;

/// xorshift64*; same generator the crate's tests use.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn uniform(alphabet: &[u8], blocks: usize, rng: &mut Rng) -> Vec<u8> {
    (0..blocks * 64)
        .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
        .collect()
}

/// Run-structured input: random bytes emitted in runs of 1..=10, the shape
/// where carry- and parity-based tricks earn their keep.
fn runs(alphabet: &[u8], blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let len = blocks * 64;
    let mut input = Vec::with_capacity(len);
    while input.len() < len {
        let byte = alphabet[(rng.next() % alphabet.len() as u64) as usize];
        for _ in 0..(1 + rng.next() % 10).min((len - input.len()) as u64) {
            input.push(byte);
        }
    }
    input
}

/// Backslash runs of every length 1..=8 ending right at the 64-byte seam,
/// each followed by a quote — the block-boundary carry cases.
fn seam_runs() -> Vec<u8> {
    let mut input = vec![b'x'; 192];
    let mut pos = 10;
    for run in 1..=8 {
        for k in 0..run {
            input[pos + k] = b'\\';
        }
        input[pos + run] = b'"';
        pos += run + 3;
    }
    // A run straddling the second seam: bytes 124..=130 are backslashes.
    for byte in &mut input[124..=130] {
        *byte = b'\\';
    }
    input[131] = b'"';
    input
}

fn mask_ref(data: &[u8], mut f: impl FnMut(u8) -> bool) -> Vec<u64> {
    let mut masks = vec![0u64; data.len().div_ceil(64)];
    for (i, &b) in data.iter().enumerate() {
        if f(b) {
            masks[i / 64] |= 1 << (i % 64);
        }
    }
    masks
}

fn budget(max_level: usize, max_candidates: u64, progress: bool) -> Budget {
    Budget {
        max_level,
        max_candidates,
        max_bank: 4_000_000,
        settle_levels: 1,
        cost: CostModel::avx2(),
        order: Order::TreeSize,
        progress,
    }
}

fn report(name: &str, spec: &str, outcome: Outcome) -> Option<Solution> {
    println!("rung: {name}");
    println!("  serial spec: {spec}");
    match outcome {
        Outcome::Found(sol) => {
            println!("  FOUND  {}", sol.expr);
            println!(
                "         cost {} (avx2 model), tree {}, {} graph nodes, {} candidates, {} distinct terms, {:.1}s, verified on {} fresh inputs",
                sol.cost,
                sol.tree_size,
                sol.dag_nodes,
                sol.stats.candidates,
                sol.stats.bank_unique,
                sol.stats.elapsed_ms as f64 / 1000.0,
                sol.verified_inputs,
            );
            if sol.alternates.len() > 1 {
                println!("         equivalent forms seen (cost, cheapest first):");
                for (cost, expr) in &sol.alternates {
                    println!("           {cost:>4}  {expr}");
                }
            }
            if sol.stats.restarts > 0 {
                println!("         ({} CEGIS corpus extensions)", sol.stats.restarts);
            }
            println!();
            Some(*sol)
        }
        Outcome::NotFound(stats) => {
            println!(
                "  NOT FOUND through level {} — {} candidates, {} distinct terms, {:.1}s",
                stats.completed_level,
                stats.candidates,
                stats.bank_unique,
                stats.elapsed_ms as f64 / 1000.0,
            );
            println!();
            None
        }
    }
}

/// The serial spec for escaped positions, falx convention: a non-escape
/// byte is marked iff the run of escape bytes ending just before it has
/// odd length.
fn escaped_reference(data: &[u8]) -> Vec<u64> {
    let mut run_odd = false;
    mask_ref(data, |b| {
        if b == b'\\' {
            run_odd = !run_odd;
            false
        } else {
            let out = run_odd;
            run_odd = false;
            out
        }
    })
}

/// Differential check of a synthesized escape graph against the
/// hand-derived one on random escape-heavy inputs.
fn check_vs_hand(sol: &Solution) {
    let hand = hand_escape_graph();
    println!(
        "  hand-derived graph: {} nodes   synthesized: {} nodes",
        hand.nodes().len(),
        sol.dag_nodes
    );
    let mut rng = Rng(0xBEEF_BEEF_BEEF_BEEF);
    let alphabet = b"\\\"x\n";
    let mut agreed = 0usize;
    for _ in 0..50_000 {
        let len = (rng.next() % 384) as usize;
        let input: Vec<u8> = (0..len)
            .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
            .collect();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        interp::run(&sol.graph, &input, &mut a);
        interp::run(&hand, &input, &mut b);
        assert_eq!(a, b, "synthesized and hand graphs diverged on {input:?}");
        agreed += 1;
    }
    println!("  differential check vs hand-derived graph: {agreed} random inputs, all agree");
    println!();
}

/// Run the product-automaton prover and print the verdict: a complete
/// equivalence proof over ALL inputs, not a sample.
fn prove_report(graph: &Graph, fsm: &Fsm) {
    match prove(graph, fsm) {
        ProveOutcome::Proven(proof) => println!(
            "  PROVEN equal to the serial machine for ALL inputs ({} product states, {} transitions explored)",
            proof.product_states, proof.transitions
        ),
        ProveOutcome::Refuted(witness) => {
            println!("  !! REFUTED by shortest witness {witness:?}")
        }
        ProveOutcome::Aborted { explored } => {
            println!("  proof aborted: state space over {explored} states")
        }
    }
    println!();
}

/// The escaped-positions serial machine, as an explicit FSM for the prover.
fn escaped_fsm_step(state: u32, byte: u8) -> (u32, bool) {
    if byte == b'\\' { (state ^ 1, false) } else { (0, state == 1) }
}

/// falx's hand-derived `escaped_positions` graph (formats.rs history),
/// replicated through the public builders so the demo can count nodes and
/// differentially compare against whatever the synthesizer finds.
fn hand_escape_graph() -> Graph {
    let mut g = Graph::new();
    let backslashes = g.class_byte(b'\\');
    let shifted = g.shift_left1(backslashes);
    let not_shifted = g.not(shifted);
    let starts = g.and(backslashes, not_shifted);
    let even_positions = g.constant(EVEN);
    let odd_positions = g.constant(!EVEN);
    let not_backslashes = g.not(backslashes);
    let even_starts = g.and(starts, even_positions);
    let even_carries = g.add(backslashes, even_starts);
    let even_run_ends = g.and(even_carries, not_backslashes);
    let odd_len_from_even = g.and(even_run_ends, odd_positions);
    let odd_starts = g.and(starts, odd_positions);
    let odd_carries = g.add(backslashes, odd_starts);
    let odd_run_ends = g.and(odd_carries, not_backslashes);
    let odd_len_from_odd = g.and(odd_run_ends, even_positions);
    let output = g.or(odd_len_from_even, odd_len_from_odd);
    g.set_output(output);
    g
}

fn main() {
    println!("falx-synth: discovering bit-parallel kernels from serial specifications");
    println!("=========================================================================");
    println!();

    let mut rng = Rng(0x0123_4567_89AB_CDEF);

    // --- Rung 1: in-string context -----------------------------------------
    let corpus: Vec<Vec<u8>> = vec![
        uniform(b"\"x,", 2, &mut rng),
        uniform(b"\"\"x", 2, &mut rng), // doubled-quote heavy
        runs(b"\"x", 2, &mut rng),
    ];
    let rung1 = report(
        "in-string mask (the quote-context trick)",
        "toggle a flag on every '\"', output the flag",
        synthesize(
            &[Leaf::class("Q", b"\"")],
            &corpus,
            &Spec::exact(&|data| {
                let mut inside = false;
                mask_ref(data, |b| {
                    if b == b'"' {
                        inside = !inside;
                    }
                    inside
                })
            }),
            &budget(4, 1_000_000, false),
        ),
    );
    if let Some(sol) = &rung1 {
        let step = |state: u32, byte: u8| -> (u32, bool) {
            let next = if byte == b'"' { state ^ 1 } else { state };
            (next, next == 1)
        };
        prove_report(&sol.graph, &Fsm { initial: 0, step: &step });
    }

    // --- Rung 2: unquoted structurals (the CSV core) ------------------------
    let corpus: Vec<Vec<u8>> = vec![
        uniform(b"\",\nx", 2, &mut rng),
        uniform(b"\",", 2, &mut rng),
        runs(b"\",\nx", 2, &mut rng),
    ];
    report(
        "unquoted structurals (CSV stage 1)",
        "track quote parity; mark ',' and '\\n' while outside quotes",
        synthesize(
            &[Leaf::class("Struct", b",\n"), Leaf::class("Q", b"\"")],
            &corpus,
            &Spec::exact(&|data| {
                let mut inside = false;
                mask_ref(data, |b| {
                    if b == b'"' {
                        inside = !inside;
                    }
                    (b == b',' || b == b'\n') && !inside
                })
            }),
            &budget(6, 5_000_000, false),
        ),
    );

    // --- Rung 2b: multi-output — the CSV mask pair from one shared DAG ------
    println!("rung: MULTI-OUTPUT — CSV structural + record-terminator masks, one DAG");
    println!("  Real kernels emit several streams from one block pass. Specs are");
    println!("  solved in order and each solved stream joins the next one's leaf");
    println!("  library, so later outputs reuse earlier ones (named O0, O1, ...).");
    let corpus: Vec<Vec<u8>> = vec![
        uniform(b"\",\nx", 2, &mut rng),
        uniform(b"\",\n", 2, &mut rng),
        runs(b"\",\nx", 2, &mut rng),
    ];
    let structural_ref = |data: &[u8]| {
        let mut inside = false;
        mask_ref(data, |b| {
            if b == b'"' {
                inside = !inside;
            }
            (b == b',' || b == b'\n') && !inside
        })
    };
    let terminator_ref = |data: &[u8]| {
        let mut inside = false;
        mask_ref(data, |b| {
            if b == b'"' {
                inside = !inside;
            }
            b == b'\n' && !inside
        })
    };
    match synthesize_multi(
        &[
            Leaf::class("Struct", b",\n"),
            Leaf::class("N", b"\n"),
            Leaf::class("Q", b"\""),
        ],
        &corpus,
        &[Spec::exact(&structural_ref), Spec::exact(&terminator_ref)],
        &budget(6, 5_000_000, false),
    ) {
        MultiOutcome::Found(multi) => {
            for (k, expr) in multi.exprs.iter().enumerate() {
                println!("  O{k} = {expr}");
            }
            println!(
                "  shared graph: {} nodes, cost {} — separate kernels would cost {}",
                multi.graph.nodes().len(),
                multi.shared_cost,
                multi.separate_cost,
            );
            println!();
        }
        MultiOutcome::NotFound { failed_spec, stats } => {
            println!("  NOT FOUND for spec {failed_spec}: {stats:?}");
            println!();
        }
    }

    // --- Rung 3: line starts -------------------------------------------------
    let corpus: Vec<Vec<u8>> = vec![uniform(b"\nxy", 2, &mut rng), runs(b"\nx", 2, &mut rng)];
    report(
        "line starts",
        "mark position 0 and every byte following '\\n'",
        synthesize(
            &[Leaf::class("N", b"\n")],
            &corpus,
            &Spec::exact(&|data| {
                let mut at_start = true;
                mask_ref(data, |b| {
                    let out = at_start;
                    at_start = b == b'\n';
                    out
                })
            }),
            &budget(4, 1_000_000, false),
        ),
    );

    // --- Rungs 4 and 5: escape-run vocabulary --------------------------------
    let escape_corpus: Vec<Vec<u8>> = vec![
        uniform(b"\\x\"", 2, &mut rng),
        runs(b"\\x\"", 2, &mut rng),
        runs(b"\\x", 3, &mut rng),
        seam_runs(),
        vec![b'\\'; 128],
        b"\\x".repeat(64),
        b"\\\\x".iter().copied().cycle().take(192).collect(),
    ];

    let run_starts = report(
        "backslash-run starts",
        "mark every '\\' whose previous byte is not '\\'",
        synthesize(
            &[Leaf::class("B", b"\\")],
            &escape_corpus,
            &Spec::exact(&|data| {
                let mut prev = false;
                mask_ref(data, |b| {
                    let is = b == b'\\';
                    let out = is && !prev;
                    prev = is;
                    out
                })
            }),
            &budget(6, 5_000_000, false),
        ),
    )
    .expect("run starts should be a size-5 find");

    let follows = report(
        "follows-a-run (landing positions)",
        "mark every non-'\\' whose previous byte is '\\'",
        synthesize(
            &[Leaf::class("B", b"\\")],
            &escape_corpus,
            &Spec::exact(&|data| {
                let mut prev = false;
                mask_ref(data, |b| {
                    let is = b == b'\\';
                    let out = !is && prev;
                    prev = is;
                    out
                })
            }),
            &budget(6, 5_000_000, false),
        ),
    )
    .expect("landing positions should be a size-5 find");

    // --- Rung 6: the odd-run escape trick -------------------------------------
    println!("rung: ESCAPED POSITIONS — the simdjson odd-backslash-run trick");
    println!("  serial spec: flip run parity on '\\'; on a non-'\\', mark it iff parity");
    println!("  was odd, then clear. (What escaped_positions in formats.rs computes");
    println!("  with 16 graph nodes, derived by hand from the simdjson paper.)");
    println!("  leaves: B, EVEN, plus the two streams synthesized above (library learning)");
    let outcome = synthesize(
        &[
            Leaf::class("B", b"\\"),
            Leaf::constant("EVEN", EVEN),
            Leaf::derived("S", run_starts.graph.clone()),
            Leaf::derived("F", follows.graph.clone()),
        ],
        &escape_corpus,
        &Spec::exact(&escaped_reference),
        &budget(9, 600_000_000, true),
    );
    if let Some(sol) = report("escaped positions (continued)", "as above", outcome) {
        check_vs_hand(&sol);
        prove_report(&sol.graph, &Fsm { initial: 0, step: &escaped_fsm_step });
    }

    // --- Rung 6b: same target, but the system invents its own vocabulary ---
    println!("rung: ESCAPED POSITIONS FROM SCRATCH — automatic abstraction discovery");
    println!("  leaves: B and EVEN only. When a round of enumeration exhausts, banked");
    println!("  terms are scored (gate = precision x recall vs target, generativity =");
    println!("  novel terms built on them, near-miss subterm harvest) and the best are");
    println!("  promoted to leaves for the next round. No human-chosen subgoals.");
    let auto = AutoBudget {
        rounds: 4,
        per_round: Budget {
            max_level: 9,
            max_candidates: 250_000_000,
            max_bank: 4_000_000,
            settle_levels: 1,
            cost: CostModel::avx2(),
            order: Order::TreeSize,
            progress: false,
        },
        promotions: 8,
        max_leaves: 24,
    };
    let outcome = synthesize_auto(
        &[Leaf::class("B", b"\\"), Leaf::constant("EVEN", EVEN)],
        &escape_corpus,
        &Spec::exact(&escaped_reference),
        &auto,
    );
    let (solution, reports) = match outcome {
        AutoOutcome::Found(sol, reports) => (Some(sol), reports),
        AutoOutcome::NotFound(reports) => (None, reports),
    };
    for r in &reports {
        println!(
            "  round {}: exhausted at level {} ({} candidates, {} distinct terms, {:.1}s); promoted:",
            r.round,
            r.stats.completed_level,
            r.stats.candidates,
            r.stats.bank_unique,
            r.stats.elapsed_ms as f64 / 1000.0,
        );
        for name in &r.promoted {
            println!("    + {name}");
        }
    }
    match solution {
        Some(sol) => {
            println!(
                "  FOUND in round {}: {}",
                reports.len() + 1,
                sol.expr
            );
            println!(
                "         tree {}, {} graph nodes, verified on {} fresh inputs, {:.1}s total",
                sol.tree_size,
                sol.dag_nodes,
                sol.verified_inputs,
                sol.stats.elapsed_ms as f64 / 1000.0,
            );
            check_vs_hand(&sol);
            prove_report(&sol.graph, &Fsm { initial: 0, step: &escaped_fsm_step });
        }
        None => {
            println!("  NOT FOUND within {} rounds — honest frontier above.", auto.rounds);
            println!();
        }
    }

    // --- Rung 6c: don't-care synthesis -----------------------------------------
    println!("rung: ESCAPED POSITIONS UNDER DON'T-CARE — only quote bytes matter");
    println!("  The only consumer of this stream is `quotes & !escaped`, so the spec");
    println!("  now constrains it at '\"' bytes ONLY; everywhere else is a free bit.");
    println!("  Don't-cares admit smaller circuits than any exact-equality form.");
    println!("  This rung runs COST-ORDERED (Dijkstra over tree cost): cheap forms");
    println!("  surface first, expensive subtrees are implicitly deprioritized.");
    let care_outcome = synthesize(
        &[
            Leaf::class("B", b"\\"),
            Leaf::constant("EVEN", EVEN),
            Leaf::derived("S", run_starts.graph.clone()),
            Leaf::derived("F", follows.graph.clone()),
        ],
        &escape_corpus,
        &Spec::with_care(&escaped_reference, &|data| mask_ref(data, |b| b == b'"')),
        &Budget {
            max_level: 14,
            max_candidates: 50_000_000,
            max_bank: 2_000_000,
            settle_levels: 2,
            cost: CostModel::avx2(),
            order: Order::Cost,
            progress: false,
        },
    );
    if let Some(sol) = report("escaped positions, quote-only care (continued)", "as above", care_outcome)
    {
        // Equality vs the hand graph holds only at quote bytes; compare there.
        let hand = hand_escape_graph();
        let mut rng = Rng(0xC0FF_EE00_C0FF_EE00);
        let alphabet = b"\\\"x\n";
        for _ in 0..50_000 {
            let len = (rng.next() % 384) as usize;
            let input: Vec<u8> = (0..len)
                .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                .collect();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            interp::run(&sol.graph, &input, &mut a);
            interp::run(&hand, &input, &mut b);
            let quotes_only = |v: &[u32]| -> Vec<u32> {
                v.iter().copied().filter(|&p| input[p as usize] == b'"').collect()
            };
            assert_eq!(
                quotes_only(&a),
                quotes_only(&b),
                "care-synthesized and hand graphs diverged at a quote in {input:?}"
            );
        }
        println!("  differential check at quote bytes vs hand graph: 50000 inputs, all agree");
        println!();
    }

    // --- Rung 7: provably out of reach ----------------------------------------
    let corpus: Vec<Vec<u8>> = vec![
        uniform(b"\r\nx", 2, &mut rng),
        runs(b"\r\nx", 2, &mut rng),
        b"\r\n".repeat(32),
    ];
    report(
        "CR immediately before LF (CRLF terminators, issue #3)",
        "mark every '\\r' whose NEXT byte is '\\n' — one byte of lookahead",
        synthesize(
            &[Leaf::class("CR", b"\r"), Leaf::class("LF", b"\n")],
            &corpus,
            &Spec::exact(&|data| {
                let mut masks = vec![0u64; data.len().div_ceil(64)];
                for i in 0..data.len().saturating_sub(1) {
                    if data[i] == b'\r' && data[i + 1] == b'\n' {
                        masks[i / 64] |= 1 << (i % 64);
                    }
                }
                masks
            }),
            &budget(8, 40_000_000, false),
        ),
    );
    println!("  Expected: every IR op is causal (bit i depends only on bytes <= i), and");
    println!("  composition preserves causality, so NO graph of ANY size computes this —");
    println!("  the exhaustive search corroborates what the argument proves. Multi-byte");
    println!("  terminators need either a lookahead op (ShiftRight1) or falx's existing");
    println!("  convention: mark the LF, trim the CR in the span layer.");
}
