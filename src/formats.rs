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
}

/// RFC 4180 CSV: comma/newline structure, double-quote regions, `""` escapes.
pub fn csv_dialect() -> Dialect {
    Dialect {
        structural: vec![b',', b'\n'],
        quote: Some(b'"'),
        escape: Escape::None,
    }
}

/// Unquoted tab-separated values.
pub fn tsv_dialect() -> Dialect {
    Dialect {
        structural: vec![b'\t', b'\n'],
        quote: None,
        escape: Escape::None,
    }
}

/// logfmt-style `key=value key2="quoted val"` lines with backslash escapes.
pub fn logfmt_dialect() -> Dialect {
    Dialect {
        structural: vec![b' ', b'=', b'\n'],
        quote: Some(b'"'),
        escape: Escape::Backslash(b'\\'),
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
    }
}

/// NDJSON framing: newlines outside JSON strings delimit records. (Framing
/// only — splitting a stream into documents; not a JSON value parser.)
pub fn ndjson_dialect() -> Dialect {
    Dialect {
        structural: vec![b'\n'],
        quote: Some(b'"'),
        escape: Escape::Backslash(b'\\'),
    }
}

/// A dialect graph plus the auxiliary node the code generator needs to emit
/// record-aware tapes: which structural bytes are record terminators.
pub struct DelimitedParts {
    pub graph: Graph,
    /// Raw `\n` class node (pre quote-masking); ANDed with the output stream
    /// it marks record ends.
    pub terminators: NodeId,
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

    let output = match dialect.quote {
        None => candidates,
        Some(quote) => {
            let quotes = g.class_byte(quote);
            let real_quotes = match dialect.escape {
                Escape::None => quotes,
                Escape::Backslash(escape_byte) => {
                    let escaped = escaped_positions(&mut g, escape_byte);
                    let not_escaped = g.not(escaped);
                    g.and(quotes, not_escaped)
                }
            };
            let inside = g.prefix_xor(real_quotes);
            let outside = g.not(inside);
            g.and(candidates, outside)
        }
    };
    g.set_output(output);
    DelimitedParts { graph: g, terminators }
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
/// This is the simdjson odd-backslash-run algorithm. Run starts are split by
/// position parity (even/odd bit constants); adding the run mask to its
/// starts makes a carry ripple to the bit just past each run, and whether
/// that landing bit's parity *changed* tells whether the run length was odd.
/// `ShiftLeft1` and `Add` carries make runs spanning 64-byte blocks work
/// unchanged.
fn escaped_positions(g: &mut Graph, escape_byte: u8) -> NodeId {
    const EVEN: u64 = 0x5555_5555_5555_5555;

    let backslashes = g.class_byte(escape_byte);
    let shifted = g.shift_left1(backslashes);
    let not_shifted = g.not(shifted);
    let starts = g.and(backslashes, not_shifted);

    let even_positions = g.constant(EVEN);
    let odd_positions = g.constant(!EVEN);
    let not_backslashes = g.not(backslashes);

    // Runs starting on even positions: the bit just past the run (the first
    // non-backslash) lands on an odd position iff the run length was odd.
    let even_starts = g.and(starts, even_positions);
    let even_carries = g.add(backslashes, even_starts);
    let even_run_ends = g.and(even_carries, not_backslashes);
    let odd_len_from_even = g.and(even_run_ends, odd_positions);

    // Runs starting on odd positions: symmetric.
    let odd_starts = g.and(starts, odd_positions);
    let odd_carries = g.add(backslashes, odd_starts);
    let odd_run_ends = g.and(odd_carries, not_backslashes);
    let odd_len_from_odd = g.and(odd_run_ends, even_positions);

    g.or(odd_len_from_even, odd_len_from_odd)
}
