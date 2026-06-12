//! Weighted synthesis of format graphs from [`crate::formats::Dialect`]
//! descriptions.

use crate::formats::{DelimitedParts, Dialect, Escape};
use crate::synth::{
    synthesize_multi, Budget, CostModel, Leaf, MultiOutcome, MultiSpec, Order, Spec, Stats,
};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const EVEN: u64 = 0x5555_5555_5555_5555;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SynthProfile {
    /// Bounded search profile for CI and smoke tests.
    Fast,
    /// Full weighted profile for opt-in kernel generation.
    Weighted,
}

#[derive(Debug)]
pub enum SynthFormatError {
    Unsupported(&'static str),
    NotFound { stage: &'static str, stats: Stats },
}

impl std::fmt::Display for SynthFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(f, "unsupported dialect: {reason}"),
            Self::NotFound { stage, stats } => write!(
                f,
                "no solution for {stage}: completed level {}, {} candidates, {} bank terms",
                stats.completed_level, stats.candidates, stats.bank_unique
            ),
        }
    }
}

impl std::error::Error for SynthFormatError {}

pub fn supports_weighted(dialect: &Dialect) -> bool {
    validate_dialect(dialect).is_ok()
}

pub fn synthesize_delimited_parts_with_profile(
    dialect: &Dialect,
    profile: SynthProfile,
) -> Result<DelimitedParts, SynthFormatError> {
    validate_dialect(dialect)?;
    let key = SynthKey::new(dialect, profile);
    if let Some(parts) = cache()
        .lock()
        .expect("synth cache poisoned")
        .get(&key)
        .cloned()
    {
        return Ok(parts);
    }

    let parts = synthesize_delimited_parts_uncached(dialect, profile)?;
    cache()
        .lock()
        .expect("synth cache poisoned")
        .insert(key, parts.clone());
    Ok(parts)
}

fn synthesize_delimited_parts_uncached(
    dialect: &Dialect,
    profile: SynthProfile,
) -> Result<DelimitedParts, SynthFormatError> {
    let corpus = corpus_for(dialect);
    let budget = budget(profile);
    let structural_bytes = dialect.structural.clone();
    let quote_byte = dialect.quote;
    let escape_byte = match dialect.escape {
        Escape::None => None,
        Escape::Backslash(byte) => Some(byte),
    };
    let opens: Vec<u8> = dialect.nesting.iter().map(|&(open, _)| open).collect();
    let closes: Vec<u8> = dialect.nesting.iter().map(|&(_, close)| close).collect();
    let opens_for_ref = opens.clone();
    let closes_for_ref = closes.clone();

    let escaped_ref = move |data: &[u8]| escaped_reference(data, escape_byte.unwrap_or(b'\\'));
    let escaped_care = move |data: &[u8]| match quote_byte {
        Some(quote) => mask_ref(data, |byte| byte == quote),
        None => vec![0; data.len().div_ceil(64)],
    };
    let real_quotes_ref = move |data: &[u8]| real_quotes_reference(data, quote_byte, escape_byte);
    let in_string_ref = move |data: &[u8]| in_string_reference(data, quote_byte, escape_byte);
    let structural_ref =
        move |data: &[u8]| live_class_reference(data, &structural_bytes, quote_byte, escape_byte);
    let terminator_ref = move |data: &[u8]| mask_ref(data, |byte| byte == b'\n');
    let opens_ref =
        move |data: &[u8]| live_class_reference(data, &opens_for_ref, quote_byte, escape_byte);
    let closes_ref =
        move |data: &[u8]| live_class_reference(data, &closes_for_ref, quote_byte, escape_byte);

    let escape_leaves = vec![
        Leaf::class("B", &[escape_byte.unwrap_or(b'\\')]),
        Leaf::constant("EVEN", EVEN),
    ];
    let quote_leaves = quote_byte.map(|quote| vec![Leaf::class("Q", &[quote])]);
    let structural_leaves = vec![Leaf::class("Struct", &dialect.structural)];
    let terminator_leaves = vec![Leaf::class("N", b"\n")];
    let open_leaves = (!dialect.nesting.is_empty()).then(|| vec![Leaf::class("Open", &opens)]);
    let close_leaves = (!dialect.nesting.is_empty()).then(|| vec![Leaf::class("Close", &closes)]);

    let mut specs = Vec::new();
    let mut stage_names = Vec::new();
    let mut opens_idx = None;
    let mut closes_idx = None;

    if escape_byte.is_some() {
        stage_names.push("escaped positions");
        specs.push(MultiSpec {
            leaves: &escape_leaves,
            spec: Spec::with_care(&escaped_ref, &escaped_care),
        });
    }
    if let Some(leaves) = quote_leaves.as_ref() {
        stage_names.push("real quotes");
        specs.push(MultiSpec {
            leaves,
            spec: Spec::exact(&real_quotes_ref),
        });

        stage_names.push("in-string mask");
        specs.push(MultiSpec {
            leaves: &[],
            spec: Spec::exact(&in_string_ref),
        });
    }

    let structural_idx = specs.len();
    stage_names.push("structural mask");
    specs.push(MultiSpec {
        leaves: &structural_leaves,
        spec: Spec::exact(&structural_ref),
    });

    let terminator_idx = specs.len();
    stage_names.push("terminator mask");
    specs.push(MultiSpec {
        leaves: &terminator_leaves,
        spec: Spec::exact(&terminator_ref),
    });

    if let (Some(open_leaves), Some(close_leaves)) = (open_leaves.as_ref(), close_leaves.as_ref()) {
        opens_idx = Some(specs.len());
        stage_names.push("live open brackets");
        specs.push(MultiSpec {
            leaves: open_leaves,
            spec: Spec::exact(&opens_ref),
        });

        closes_idx = Some(specs.len());
        stage_names.push("live close brackets");
        specs.push(MultiSpec {
            leaves: close_leaves,
            spec: Spec::exact(&closes_ref),
        });
    }

    let multi = match synthesize_multi(&corpus, &specs, &budget) {
        MultiOutcome::Found(multi) => multi,
        MultiOutcome::NotFound { failed_spec, stats } => {
            return Err(SynthFormatError::NotFound {
                stage: stage_names[failed_spec],
                stats,
            });
        }
    };

    let mut graph = multi.graph.clone();
    graph.set_output(multi.outputs[structural_idx]);
    let terminators = multi.outputs[terminator_idx];
    let nest = opens_idx
        .zip(closes_idx)
        .map(|(open, close)| (multi.outputs[open], multi.outputs[close]));

    Ok(DelimitedParts {
        graph,
        terminators,
        nest,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SynthKey {
    structural: Vec<u8>,
    quote: Option<u8>,
    escape: EscapeKey,
    comment: Option<u8>,
    nesting: Vec<(u8, u8)>,
    profile: SynthProfile,
}

impl SynthKey {
    fn new(dialect: &Dialect, profile: SynthProfile) -> Self {
        Self {
            structural: dialect.structural.clone(),
            quote: dialect.quote,
            escape: EscapeKey::from(dialect.escape),
            comment: dialect.comment,
            nesting: dialect.nesting.clone(),
            profile,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EscapeKey {
    None,
    Backslash(u8),
}

impl From<Escape> for EscapeKey {
    fn from(escape: Escape) -> Self {
        match escape {
            Escape::None => Self::None,
            Escape::Backslash(byte) => Self::Backslash(byte),
        }
    }
}

fn cache() -> &'static Mutex<HashMap<SynthKey, DelimitedParts>> {
    static CACHE: OnceLock<Mutex<HashMap<SynthKey, DelimitedParts>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn budget(profile: SynthProfile) -> Budget {
    match profile {
        SynthProfile::Fast => Budget {
            max_level: 28,
            max_candidates: 20_000_000,
            max_bank: 2_000_000,
            settle_levels: 0,
            cost: CostModel::avx2(),
            order: Order::Cost,
            progress: false,
        },
        SynthProfile::Weighted => Budget {
            max_level: 28,
            max_candidates: 60_000_000,
            max_bank: 2_000_000,
            settle_levels: 2,
            cost: CostModel::avx2(),
            order: Order::Cost,
            progress: false,
        },
    }
}

fn corpus_for(dialect: &Dialect) -> Vec<Vec<u8>> {
    let mut alphabet = dialect.structural.clone();
    if let Some(quote) = dialect.quote {
        alphabet.extend([quote, quote]);
    }
    if let Escape::Backslash(escape) = dialect.escape {
        alphabet.extend([escape, escape, escape]);
    }
    alphabet.extend_from_slice(b"xy \t\r");
    alphabet.sort_unstable();
    alphabet.dedup();

    let mut rng = Rng(0x51A7_EC0D_EC0D_0001);
    vec![
        pad_to_block(Vec::new(), &alphabet),
        pad_to_block(
            alphabet.iter().copied().cycle().take(128).collect(),
            &alphabet,
        ),
        uniform(&alphabet, 2, &mut rng),
        runs(&alphabet, 2, &mut rng),
        quote_boundary_case(dialect, &alphabet),
    ]
}

fn pad_to_block(mut data: Vec<u8>, alphabet: &[u8]) -> Vec<u8> {
    let pad = alphabet
        .iter()
        .copied()
        .find(|&byte| byte != b'"' && byte != b'\\')
        .unwrap_or(b'x');
    let len = data.len().next_multiple_of(64).max(64);
    data.resize(len, pad);
    data
}

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

fn quote_boundary_case(dialect: &Dialect, alphabet: &[u8]) -> Vec<u8> {
    let structural_len =
        21usize.saturating_add(dialect.structural.len().saturating_sub(1).saturating_mul(7));
    let len = structural_len.next_multiple_of(64).max(192);
    let mut data = vec![alphabet[0]; len];
    if let Some(quote) = dialect.quote {
        for pos in [0usize, 63, 64, 65, 127, 128] {
            data[pos] = quote;
        }
    }
    if let Escape::Backslash(escape) = dialect.escape {
        for pos in [10usize, 11, 62, 63, 64, 126, 127, 128] {
            data[pos] = escape;
        }
    }
    for (i, byte) in dialect.structural.iter().copied().enumerate() {
        data[20 + i * 7] = byte;
    }
    data
}

fn quote_escape_conflict(dialect: &Dialect) -> bool {
    matches!(dialect.escape, Escape::Backslash(escape) if dialect.quote == Some(escape))
}

fn validate_dialect(dialect: &Dialect) -> Result<(), SynthFormatError> {
    if dialect.comment.is_some() {
        return Err(SynthFormatError::Unsupported(
            "comment regions currently use the sequential Regions op",
        ));
    }
    if quote_escape_conflict(dialect) {
        return Err(SynthFormatError::Unsupported(
            "quote/escape conflict: quote and escape byte must differ",
        ));
    }

    let mut seen = std::collections::HashSet::new();
    for &(open, close) in &dialect.nesting {
        if open == close {
            return Err(SynthFormatError::Unsupported(
                "nesting pair has identical open and close bytes",
            ));
        }
        for byte in [open, close] {
            if !dialect.structural.contains(&byte) {
                return Err(SynthFormatError::Unsupported(
                    "nesting byte is not in the structural set",
                ));
            }
            if Some(byte) == dialect.quote || Some(byte) == dialect.comment {
                return Err(SynthFormatError::Unsupported(
                    "nesting byte conflicts with the quote or comment byte",
                ));
            }
            if byte == b'\n' {
                return Err(SynthFormatError::Unsupported(
                    "newline cannot be a nesting byte",
                ));
            }
            if !seen.insert(byte) {
                return Err(SynthFormatError::Unsupported(
                    "nesting byte appears in more than one pair",
                ));
            }
        }
    }

    Ok(())
}

fn mask_ref(data: &[u8], mut f: impl FnMut(u8) -> bool) -> Vec<u64> {
    let mut masks = vec![0u64; data.len().div_ceil(64)];
    for (i, &byte) in data.iter().enumerate() {
        if f(byte) {
            masks[i / 64] |= 1 << (i % 64);
        }
    }
    masks
}

fn escaped_reference(data: &[u8], escape: u8) -> Vec<u64> {
    let mut run_odd = false;
    mask_ref(data, |byte| {
        if byte == escape {
            run_odd = !run_odd;
            false
        } else {
            let out = run_odd;
            run_odd = false;
            out
        }
    })
}

fn real_quotes_reference(data: &[u8], quote: Option<u8>, escape: Option<u8>) -> Vec<u64> {
    let Some(quote) = quote else {
        return vec![0; data.len().div_ceil(64)];
    };
    let mut run_odd = false;
    mask_ref(data, |byte| {
        let escaped = match escape {
            Some(escape) if byte == escape => {
                run_odd = !run_odd;
                return false;
            }
            Some(_) => {
                let escaped = run_odd;
                run_odd = false;
                escaped
            }
            None => false,
        };
        byte == quote && !escaped
    })
}

fn in_string_reference(data: &[u8], quote: Option<u8>, escape: Option<u8>) -> Vec<u64> {
    let Some(quote) = quote else {
        return vec![0; data.len().div_ceil(64)];
    };
    let mut run_odd = false;
    let mut in_string = false;
    mask_ref(data, |byte| {
        let escaped = match escape {
            Some(escape) if byte == escape => {
                run_odd = !run_odd;
                return in_string;
            }
            Some(_) => {
                let escaped = run_odd;
                run_odd = false;
                escaped
            }
            None => false,
        };
        if byte == quote && !escaped {
            in_string = !in_string;
        }
        in_string
    })
}

fn live_class_reference(
    data: &[u8],
    class: &[u8],
    quote: Option<u8>,
    escape: Option<u8>,
) -> Vec<u64> {
    if quote.is_none() {
        return mask_ref(data, |byte| class.contains(&byte));
    }
    let inside = in_string_reference(data, quote, escape);
    let raw = mask_ref(data, |byte| class.contains(&byte));
    raw.into_iter()
        .zip(inside)
        .map(|(raw, inside)| raw & !inside)
        .collect()
}
