//! Integration tests pinning the synthesizer's flagship discoveries.
//!
//! These tests enforce that future heuristic tuning cannot silently break
//! the core rediscovery and proving capabilities.

use falx::ir::Graph;
use falx::synth::{
    Order,
    AutoBudget, AutoOutcome, Budget, CostModel, Fsm, Leaf, Outcome, ProveOutcome, Spec,
    prove, synthesize, synthesize_auto,
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
