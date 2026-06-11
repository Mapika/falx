//! Integration tests pinning the synthesizer's flagship discoveries.
//!
//! These tests enforce that future heuristic tuning cannot silently break
//! the core rediscovery and proving capabilities.

use falx::ir::Graph;
use falx::synth::{
    AutoBudget, AutoOutcome, Budget, CostModel, Fsm, Leaf, MultiOutcome, MultiSpec, Order,
    Outcome, ProveOutcome, Spec, prove, synthesize, synthesize_auto, synthesize_multi,
};

const EVEN: u64 = 0x5555_5555_5555_5555;

/// xorshift64* RNG; same generator the crate's tests use.
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

/// Test 1: Assisted escape rediscovery with human-derived vocabulary.
/// Verifies that the synthesizer rediscovers the escape kernel using
/// manually-constructed "run starts" and "landing positions" leaves,
/// and that the solution is efficiently small.
#[test]
fn assisted_escape_rediscovery_stays_green() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);

    let escape_corpus: Vec<Vec<u8>> = vec![
        uniform(b"\\x\"", 2, &mut rng),
        runs(b"\\x\"", 2, &mut rng),
        runs(b"\\x", 3, &mut rng),
        seam_runs(),
        vec![b'\\'; 128],
        b"\\x".repeat(64),
        b"\\\\x".iter().copied().cycle().take(192).collect(),
    ];

    // Build the "run starts" leaf manually.
    let mut g_starts = Graph::new();
    let b = g_starts.class_byte(b'\\');
    let shifted = g_starts.shift_left1(b);
    let not_shifted = g_starts.not(shifted);
    let starts = g_starts.and(b, not_shifted);
    g_starts.set_output(starts);

    // Build the "landing positions" leaf manually.
    let mut g_follows = Graph::new();
    let b = g_follows.class_byte(b'\\');
    let shifted = g_follows.shift_left1(b);
    let not_b = g_follows.not(b);
    let follows = g_follows.and(not_b, shifted);
    g_follows.set_output(follows);

    let outcome = synthesize(
        &[
            Leaf::class("B", b"\\"),
            Leaf::constant("EVEN", EVEN),
            Leaf::derived("S", g_starts),
            Leaf::derived("F", g_follows),
        ],
        &escape_corpus,
        &Spec::exact(&escaped_reference),
        &Budget {
            max_level: 9,
            max_candidates: 50_000_000,
            max_bank: 2_000_000,
            settle_levels: 0,
            cost: CostModel::avx2(),
        order: Order::TreeSize,
            progress: false,
        },
    );

    match outcome {
        Outcome::Found(solution) => {
            // Must succeed.
            assert!(solution.dag_nodes <= 12,
                "solution has {} nodes, expected <= 12",
                solution.dag_nodes);
        }
        Outcome::NotFound(stats) => {
            panic!(
                "synthesizer failed: exhausted to tree size {}, {} candidates, {} bank terms",
                stats.completed_level, stats.candidates, stats.bank_unique
            );
        }
    }
}

/// Test 2: Verify the discovered escape kernel against the serial FSM.
/// Proves that a synthesized solution is correct for ALL inputs, not just
/// the corpus.
#[test]
fn found_escape_kernel_is_proven() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);

    let escape_corpus: Vec<Vec<u8>> = vec![
        uniform(b"\\x\"", 2, &mut rng),
        runs(b"\\x\"", 2, &mut rng),
        runs(b"\\x", 3, &mut rng),
        seam_runs(),
        vec![b'\\'; 128],
        b"\\x".repeat(64),
        b"\\\\x".iter().copied().cycle().take(192).collect(),
    ];

    // Build leaves.
    let mut g_starts = Graph::new();
    let b = g_starts.class_byte(b'\\');
    let shifted = g_starts.shift_left1(b);
    let not_shifted = g_starts.not(shifted);
    let starts = g_starts.and(b, not_shifted);
    g_starts.set_output(starts);

    let mut g_follows = Graph::new();
    let b = g_follows.class_byte(b'\\');
    let shifted = g_follows.shift_left1(b);
    let not_b = g_follows.not(b);
    let follows = g_follows.and(not_b, shifted);
    g_follows.set_output(follows);

    let outcome = synthesize(
        &[
            Leaf::class("B", b"\\"),
            Leaf::constant("EVEN", EVEN),
            Leaf::derived("S", g_starts),
            Leaf::derived("F", g_follows),
        ],
        &escape_corpus,
        &Spec::exact(&escaped_reference),
        &Budget {
            max_level: 9,
            max_candidates: 50_000_000,
            max_bank: 2_000_000,
            settle_levels: 0,
            cost: CostModel::avx2(),
        order: Order::TreeSize,
            progress: false,
        },
    );

    if let Outcome::Found(solution) = outcome {
        // The escape FSM: state toggles on backslash, outputs on non-backslash
        // iff state was 1.
        let escape_fsm = Fsm {
            initial: 0,
            step: &|state: u32, byte: u8| -> (u32, bool) {
                if byte == b'\\' {
                    (state ^ 1, false)
                } else {
                    (0, state == 1)
                }
            },
        };

        match prove(&solution.graph, &escape_fsm) {
            ProveOutcome::Proven(_) => {
                // Success: proved on all inputs.
            }
            ProveOutcome::Refuted(witness) => {
                panic!("proof failed: solution refuted by witness {:?}", witness);
            }
            ProveOutcome::Aborted { explored } => {
                panic!("proof aborted after exploring {} states", explored);
            }
        }
    } else {
        panic!("synthesis failed, cannot run proof");
    }
}

/// Test 3: Automatic abstraction discovery from scratch.
/// Verifies that the synthesizer can find the escape kernel using only
/// backslash and even-position classes as base leaves, automatically
/// discovering the needed vocabulary via library learning.
#[test]
#[ignore = "expensive: run with cargo test --release -- --ignored"]
fn from_scratch_discovery_stays_green() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);

    let escape_corpus: Vec<Vec<u8>> = vec![
        uniform(b"\\x\"", 2, &mut rng),
        runs(b"\\x\"", 2, &mut rng),
        runs(b"\\x", 3, &mut rng),
        seam_runs(),
        vec![b'\\'; 128],
        b"\\x".repeat(64),
        b"\\\\x".iter().copied().cycle().take(192).collect(),
    ];

    let outcome = synthesize_auto(
        &[Leaf::class("B", b"\\"), Leaf::constant("EVEN", EVEN)],
        &escape_corpus,
        &Spec::exact(&escaped_reference),
        &AutoBudget {
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
        },
    );

    match outcome {
        AutoOutcome::Found(solution, _reports) => {
            assert!(solution.dag_nodes <= 12,
                "automatic solution has {} nodes, expected <= 12",
                solution.dag_nodes);
        }
        AutoOutcome::NotFound(_reports) => {
            panic!("automatic abstraction discovery failed");
        }
    }
}

/// The 6-node don't-care escape form is exact at EVERY non-escape
/// position, not just quote bytes: masking it by `Not(B)` must equal the
/// serial escape machine for all inputs. This is the proof that makes it
/// safe as `escaped_positions` for any consumer that reads non-escape
/// bytes (the quote class, whatever the quote byte is).
#[test]
fn dont_care_escape_form_is_exact_on_non_escape_bytes() {
    let mut g = Graph::new();
    let b = g.class_byte(b'\\');
    let not_b = g.not(b);
    let even = g.constant(EVEN);
    let x = g.xor(b, even);
    let sum = g.add(even, x);
    let inv = g.not(sum);
    let form = g.xor(even, inv);
    let masked = g.and(not_b, form);
    g.set_output(masked);
    let step = |state: u32, byte: u8| -> (u32, bool) {
        if byte == b'\\' { (state ^ 1, false) } else { (0, state == 1) }
    };
    match prove(&g, &Fsm { initial: 0, step: &step }) {
        ProveOutcome::Proven(proof) => assert!(proof.product_states <= 1024),
        ProveOutcome::Refuted(witness) => panic!("refuted by {witness:?}"),
        ProveOutcome::Aborted { explored } => panic!("aborted at {explored}"),
    }
}

/// CSV stage 1 as a three-output multi synthesis — in-string, structural,
/// terminator masks from per-spec leaves, later outputs reusing earlier
/// ones. Fast enough to run unconditionally.
#[test]
fn csv_trio_multi_output_stays_green() {
    let mut rng = Rng(0x2222_3333_4444_5555);
    let corpus: Vec<Vec<u8>> = vec![
        uniform(b"\",\nx", 2, &mut rng),
        uniform(b"\",", 2, &mut rng),
        runs(b"\",\nx", 2, &mut rng),
    ];
    let in_string_ref = |data: &[u8]| {
        let mut inside = false;
        mask_ref(data, |b| {
            if b == b'"' {
                inside = !inside;
            }
            inside
        })
    };
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
        &corpus,
        &[
            MultiSpec { leaves: &[Leaf::class("Q", b"\"")], spec: Spec::exact(&in_string_ref) },
            MultiSpec {
                leaves: &[Leaf::class("Struct", b",\n")],
                spec: Spec::exact(&structural_ref),
            },
            MultiSpec { leaves: &[Leaf::class("N", b"\n")], spec: Spec::exact(&terminator_ref) },
        ],
        &Budget {
            max_level: 6,
            max_candidates: 5_000_000,
            max_bank: 500_000,
            settle_levels: 0,
            cost: CostModel::avx2(),
            order: Order::TreeSize,
            progress: false,
        },
    ) {
        MultiOutcome::Found(multi) => {
            assert_eq!(multi.exprs[0], "PrefixXor(Q)");
            assert!(multi.shared_cost < multi.separate_cost);
            // The synthesized structural mask must agree with the
            // production CSV graph everywhere.
            let production = falx::formats::csv();
            let mut synthesized = multi.graph.clone();
            synthesized.set_output(multi.outputs[1]);
            let mut check = Rng(0x6666_7777_8888_9999);
            let alphabet = b"\",\nx";
            for _ in 0..2_000 {
                let len = (check.next() % 300) as usize;
                let input: Vec<u8> = (0..len)
                    .map(|_| alphabet[(check.next() % alphabet.len() as u64) as usize])
                    .collect();
                let (mut a, mut b) = (Vec::new(), Vec::new());
                falx::interp::run(&synthesized, &input, &mut a);
                falx::interp::run(&production, &input, &mut b);
                assert_eq!(a, b, "CSV stage 1 diverged on {input:?}");
            }
        }
        MultiOutcome::NotFound { failed_spec, stats } => {
            panic!("spec {failed_spec} not found: {stats:?}")
        }
    }
}
