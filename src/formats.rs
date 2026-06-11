//! Format definitions compiled into bitstream graphs.
//!
//! This module is the seed of the eventual generator: a format description
//! goes in, an IR graph comes out. The M3 front-end parses declarative specs
//! into [`Dialect`] values; the graphs built here are what both the
//! interpreter and the code generator consume.
//!
//! One builder covers the whole current family: a *delimited* format is
//! described by the set of structural bytes (field separators and record
//! terminators alike), an optional quote byte that suppresses structure, and
//! how quotes are escaped inside quoted regions.

use crate::ir::{CharClass, Graph, NodeId};

/// How a quote character can be escaped within a quoted region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Escape {
    /// No escaping (or RFC 4180 doubled quotes: `""` toggles the quote
    /// parity twice, so it needs no dedicated handling — both cases compile
    /// to plain prefix-XOR).
    None,
    /// A designated escape byte (almost always `\`) escapes the following
    /// quote; a doubled escape byte escapes itself. JSON-style.
    Backslash(u8),
}

/// A delimited format: structural bytes outside quoted regions mark field
/// and record boundaries.
#[derive(Clone, Debug)]
pub struct Dialect {
    /// The full structural byte set, record terminators included (CSV is
    /// `[b',', b'\n']`).
    pub structural: Vec<u8>,
    /// Byte that opens/closes a region where structural bytes are inert.
    pub quote: Option<u8>,
    /// How quotes are escaped inside quoted regions.
    pub escape: Escape,
    /// Comment byte: at line start (outside quotes) it makes the line
    /// inert through its terminating newline. The newline stays a record
    /// boundary; record walkers skip records that begin with this byte.
    pub comment: Option<u8>,
    /// Bracket pairs `(open, close)` that nest. Non-empty makes the
    /// generated parser emit a nested tape (`parse_nested`) on top of the
    /// structural index: brackets outside quoted regions are matched into
    /// a navigable tree. Nesting bytes must be members of `structural` —
    /// the indexer only reports bytes it classifies.
    pub nesting: Vec<(u8, u8)>,
}

/// RFC 4180 CSV: comma/newline structure, double-quote regions, `""` escapes.
pub fn csv_dialect() -> Dialect {
    Dialect {
        structural: vec![b',', b'\n'],
        quote: Some(b'"'),
        escape: Escape::None,
        comment: None,
        nesting: vec![],
    }
}

/// Unquoted tab-separated values.
pub fn tsv_dialect() -> Dialect {
    Dialect {
        structural: vec![b'\t', b'\n'],
        quote: None,
        escape: Escape::None,
        comment: None,
        nesting: vec![],
    }
}

/// logfmt-style `key=value key2="quoted val"` lines with backslash escapes.
pub fn logfmt_dialect() -> Dialect {
    Dialect {
        structural: vec![b' ', b'=', b'\n'],
        quote: Some(b'"'),
        escape: Escape::Backslash(b'\\'),
        comment: None,
        nesting: vec![],
    }
}

/// A deliberately separator-rich dialect (10 structural bytes, 9 of them
/// separators): exercises the shuffle-based nibble classifier, which
/// kicks in for classes too large for compare-based classification.
pub fn multi_dialect() -> Dialect {
    Dialect {
        structural: vec![
            b',', b';', b'|', b'\t', b':', b' ', b'/', b'=', b'&', b'\n',
        ],
        quote: Some(b'"'),
        escape: Escape::None,
        comment: None,
        nesting: vec![],
    }
}

/// CSV with `#` comment lines (the data-science CSV convention): a `#` at
/// line start, outside quotes, makes the line inert through its newline.
pub fn csv_hash_dialect() -> Dialect {
    Dialect {
        structural: vec![b',', b'\n'],
        quote: Some(b'"'),
        escape: Escape::None,
        comment: Some(b'#'),
        nesting: vec![],
    }
}

/// NDJSON framing: newlines outside JSON strings delimit records. (Framing
/// only — splitting a stream into documents; not a JSON value parser.)
pub fn ndjson_dialect() -> Dialect {
    Dialect {
        structural: vec![b'\n'],
        quote: Some(b'"'),
        escape: Escape::Backslash(b'\\'),
        comment: None,
        nesting: vec![],
    }
}

/// JSON structural parsing: brace/bracket pairs nest, commas and colons
/// separate, strings are backslash-escaped quote regions. The structural
/// index is exactly simdjson's stage 1 (minus pseudo-structural scalar
/// starts); the nested tape built on top of it matches the brackets.
pub fn json_dialect() -> Dialect {
    Dialect {
        structural: vec![b'{', b'}', b'[', b']', b',', b':'],
        quote: Some(b'"'),
        escape: Escape::Backslash(b'\\'),
        comment: None,
        nesting: vec![(b'{', b'}'), (b'[', b']')],
    }
}

/// A dialect graph plus the auxiliary node the code generator needs to emit
/// record-aware tapes: which structural bytes are record terminators.
pub struct DelimitedParts {
    pub graph: Graph,
    /// Raw `\n` class node (pre quote-masking); ANDed with the output stream
    /// it marks record ends.
    pub terminators: NodeId,
    /// Live open-bracket and close-bracket streams (bracket classes ANDed
    /// with the output, so quote/comment masking is inherited) when the
    /// dialect declares nesting pairs. Splitting brackets out of the
    /// structural stream is what lets the nested-tape matcher treat the
    /// majority class — separators — branchlessly.
    pub nest: Option<(NodeId, NodeId)>,
}

/// Build the structural-indexing graph for a dialect: the output stream
/// marks every structural byte outside quoted regions. Separators and the
/// `\n` terminator are classified separately so kernels can emit record
/// boundaries as their own stream.
pub fn delimited_parts(dialect: &Dialect) -> DelimitedParts {
    let mut g = Graph::new();
    let separators: Vec<u8> = dialect
        .structural
        .iter()
        .copied()
        .filter(|&b| b != b'\n')
        .collect();
    let terminators = g.class_byte(b'\n');
    let candidates = if separators.is_empty() {
        terminators
    } else {
        let seps = g.class(CharClass::from_bytes(&separators));
        if dialect.structural.contains(&b'\n') {
            g.or(seps, terminators)
        } else {
            seps
        }
    };

    // Quote toggles with escapes resolved (None when the dialect has no
    // quote convention; comment handling still needs a stream to feed the
    // region resolver, so it gets a constant 0 there).
    let real_quotes = dialect.quote.map(|quote| {
        let quotes = g.class_byte(quote);
        match dialect.escape {
            Escape::None => quotes,
            Escape::Backslash(escape_byte) => {
                let escaped = escaped_positions(&mut g, escape_byte);
                let not_escaped = g.not(escaped);
                g.and(quotes, not_escaped)
            }
        }
    });

    let output = match (real_quotes, dialect.comment) {
        (None, None) => candidates,
        (Some(quotes), None) => {
            // Pure parity: doubled-quote escapes self-cancel, bit-parallel.
            let inside = g.prefix_xor(quotes);
            let outside = g.not(inside);
            g.and(candidates, outside)
        }
        (quotes, Some(comment)) => {
            // Quotes and comments interleave (each makes the other inert),
            // which parity cannot express: the sequential Regions op
            // resolves both region kinds at once. A comment opens only at
            // line start — position 0 counts via the seeded shift.
            let quotes = quotes.unwrap_or_else(|| g.constant(0));
            let line_start = g.shift_left1_seeded(terminators);
            let comment_class = g.class_byte(comment);
            let comment_starts = g.and(comment_class, line_start);
            let inert = g.regions(quotes, comment_starts, terminators);
            let keep = g.not(inert);
            g.and(candidates, keep)
        }
    };
    g.set_output(output);

    let nest = if dialect.nesting.is_empty() {
        None
    } else {
        let opens: Vec<u8> = dialect.nesting.iter().map(|&(open, _)| open).collect();
        let closes: Vec<u8> = dialect.nesting.iter().map(|&(_, close)| close).collect();
        let open_class = g.class(CharClass::from_bytes(&opens));
        let close_class = g.class(CharClass::from_bytes(&closes));
        let live_opens = g.and(open_class, output);
        let live_closes = g.and(close_class, output);
        Some((live_opens, live_closes))
    };
    DelimitedParts { graph: g, terminators, nest }
}

/// The structural-indexing graph alone (see [`delimited_parts`]).
pub fn delimited(dialect: &Dialect) -> Graph {
    delimited_parts(dialect).graph
}

/// The CSV graph used by tests and benchmarks.
pub fn csv() -> Graph {
    delimited(&csv_dialect())
}

/// Stream marking every byte preceded by an odd-length run of the escape
/// byte — i.e. the positions an escape actually applies to.
///
/// Same semantics as the classic simdjson odd-backslash-run algorithm, in
/// a 9-node form DISCOVERED by `crate::synth`'s cost-weighted automatic
/// abstraction search (`examples/synth_demo.rs`, "from scratch" rung),
/// differentially verified against the hand derivation AND proven
/// equivalent to the serial escape machine for all inputs by product-
/// automaton reachability (`synth::prove`). It needs two carried states
/// where the hand derivation needed three, and no run-start computation
/// at all.
///
/// How it works: `marked` covers, within each escape run, the start bit
/// when the run starts even plus every in-run odd position — so for an
/// even-started run, `EVEN + marked` collides at the start bit and the
/// carry rides the run's contiguous coverage to the landing byte, while
/// an odd-started run interleaves with EVEN and launches no carry. At a
/// landing position `p` the sum is therefore `EVEN[p]`, flipped iff the
/// run started even — which is exactly "the run length was odd", since
/// landing parity = start parity XOR length parity. Positions that are
/// not landings read garbage from the sum (including a stray carry one
/// past an even-length run) and are masked off by `follows`, the landing
/// set. `ShiftLeft1` and `Add` carries make runs spanning 64-byte blocks
/// work unchanged.
fn escaped_positions(g: &mut Graph, escape_byte: u8) -> NodeId {
    const EVEN: u64 = 0x5555_5555_5555_5555;

    let escapes = g.class_byte(escape_byte);
    let not_escapes = g.not(escapes);
    let shifted = g.shift_left1(escapes);
    let follows = g.and(not_escapes, shifted);

    let even_positions = g.constant(EVEN);
    let phase = g.xor(even_positions, shifted);
    let marked = g.and(escapes, phase);
    let sums = g.add(even_positions, marked);
    g.and(follows, sums)
}
