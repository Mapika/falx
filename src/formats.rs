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
    pub record_terminator: u8,
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
    /// Lines per logical record. When > 1, the generated record API groups
    /// every N newline-terminated lines into one record and exposes the N
    /// constituent lines as its fields (FASTQ = 4). Only valid for a
    /// newline-only line format; 1 means one record per line.
    pub lines_per_record: u32,
}

/// RFC 4180 CSV: comma/newline structure, double-quote regions, `""` escapes.
pub fn csv_dialect() -> Dialect {
    Dialect {
        structural: vec![b',', b'\n'],
        record_terminator: b'\n',
        quote: Some(b'"'),
        escape: Escape::None,
        comment: None,
        nesting: vec![],
        lines_per_record: 1,
    }
}

/// Unquoted tab-separated values.
pub fn tsv_dialect() -> Dialect {
    Dialect {
        structural: vec![b'\t', b'\n'],
        record_terminator: b'\n',
        quote: None,
        escape: Escape::None,
        comment: None,
        nesting: vec![],
        lines_per_record: 1,
    }
}

/// logfmt-style `key=value key2="quoted val"` lines with backslash escapes.
pub fn logfmt_dialect() -> Dialect {
    Dialect {
        structural: vec![b' ', b'=', b'\n'],
        record_terminator: b'\n',
        quote: Some(b'"'),
        escape: Escape::Backslash(b'\\'),
        comment: None,
        nesting: vec![],
        lines_per_record: 1,
    }
}

/// A deliberately separator-rich dialect (10 structural bytes, 9 of them
/// separators): exercises the shuffle-based nibble classifier, which
/// kicks in for classes too large for compare-based classification.
pub fn multi_dialect() -> Dialect {
    Dialect {
        structural: vec![b',', b';', b'|', b'\t', b':', b' ', b'/', b'=', b'&', b'\n'],
        quote: Some(b'"'),
        escape: Escape::None,
        comment: None,
        nesting: vec![],
        lines_per_record: 1,
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
        lines_per_record: 1,
    }
}

/// VCF / BED / GFF / SAM-style genomics records: tab-delimited, with `#`
/// header/meta lines, and no CSV-style quoting. Comment without quote, so the
/// parallel path uses line-ownership chunking.
pub fn vcf_dialect() -> Dialect {
    Dialect {
        structural: vec![b'\t', b'\n'],
        quote: None,
        escape: Escape::None,
        comment: Some(b'#'),
        nesting: vec![],
        lines_per_record: 1,
    }
}

/// Pure newline framing: the structural index is every line boundary. The
/// base for fixed-line-count record formats — FASTQ (sequencing reads) groups
/// these by 4 (header / sequence / `+` / quality); the quality line may
/// contain `@`/`+`, so framing must count lines, not split on a sigil.
pub fn lines_dialect() -> Dialect {
    Dialect {
        structural: vec![b'\n'],
        quote: None,
        escape: Escape::None,
        comment: None,
        nesting: vec![],
        lines_per_record: 1,
    }
}

/// NDJSON framing: newlines outside JSON strings delimit records. (Framing
/// only — splitting a stream into documents; not a JSON value parser.)
pub fn ndjson_dialect() -> Dialect {
    Dialect {
        structural: vec![b'\n'],
        record_terminator: b'\n',
        quote: Some(b'"'),
        escape: Escape::Backslash(b'\\'),
        comment: None,
        nesting: vec![],
        lines_per_record: 1,
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
        lines_per_record: 1,
    }
}

/// FASTQ-style fixed-line records: newline framing grouped four lines per
/// record (`@header` / sequence / `+` / quality). The declarative
/// generalization of FASTQ's by-4 line grouping — the generated record API
/// yields one record per read with the four lines as its fields.
pub fn fastq_dialect() -> Dialect {
    Dialect {
        lines_per_record: 4,
        ..lines_dialect()
    }
}

/// A dialect graph plus the auxiliary node the code generator needs to emit
/// record-aware tapes: which structural bytes are record terminators.
#[derive(Clone)]
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
        .filter(|&b| b != dialect.record_terminator)
        .collect();
    let terminators = g.class_byte(dialect.record_terminator);
    let candidates = if separators.is_empty() {
        terminators
    } else {
        let seps = g.class(CharClass::from_bytes(&separators));
        if dialect.structural.contains(&dialect.record_terminator) {
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
                // The escaped stream is only meaningful at non-escape
                // bytes; ANDing with the quote class is what discharges
                // that contract, so the two bytes must differ.
                assert_ne!(
                    quote, escape_byte,
                    "quote and escape byte must differ for backslash escaping"
                );
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
        (None, Some(_comment)) => {
            // Comment without quote: a comment line is inert as a *whole
            // record*, and the record/stream iterators already skip any record
            // whose first byte is the comment byte. So the kernel needs no
            // comment machinery — index every structural byte normally and let
            // the skip handle comments. This keeps these dialects (VCF, BED,
            // GFF, SAM, Matrix Market …) on the fast bit-parallel plain path
            // instead of the sequential Regions op.
            candidates
        }
        (Some(quotes), Some(comment)) => {
            // Quotes and comments interleave (each makes the other inert),
            // which parity cannot express: the sequential Regions op resolves
            // both region kinds at once. A comment opens only at line start —
            // position 0 counts via the seeded shift.
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
    DelimitedParts {
        graph: g,
        terminators,
        nest,
    }
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
/// byte — i.e. the positions an escape actually applies to — VALID ONLY
/// AT NON-ESCAPE POSITIONS. At escape bytes themselves the stream is
/// unspecified; the sole consumer ANDs it (negated) with the quote class,
/// and a quote is never an escape byte (asserted by the caller).
///
/// This is a 6-node don't-care form DISCOVERED by `crate::synth` when the
/// spec constrains only the positions the consumer reads
/// (`examples/synth_demo.rs`, don't-care rung), and PROVEN exact at every
/// non-escape position by product-automaton reachability — the test
/// `dont_care_escape_form_is_exact_on_non_escape_bytes` proves
/// `Not(escapes) & form` equal to the serial escape machine for all
/// inputs, which licenses any non-escape-byte consumer. The previous
/// synthesized form needed 9 nodes (and the original hand derivation 16)
/// because exact equality everywhere forces a landing-set gate; with the
/// don't-care the gate disappears entirely.
///
/// How it works: `Xor(escapes, EVEN)` clears even-position bits inside
/// escape runs and sets them at even non-escape positions; adding EVEN
/// back makes each escape run either collide-and-carry or interleave
/// depending on its start parity, so the borrow ripple delivers run-length
/// parity to the byte just past every run, where the final XOR against
/// EVEN (via the complement) extracts it. Bits at escape positions are
/// arithmetic debris — don't-cares by contract. The `Add` carry makes runs
/// spanning 64-byte blocks work unchanged.
fn escaped_positions(g: &mut Graph, escape_byte: u8) -> NodeId {
    const EVEN: u64 = 0x5555_5555_5555_5555;

    let escapes = g.class_byte(escape_byte);
    let even_positions = g.constant(EVEN);
    let phase = g.xor(escapes, even_positions);
    let sums = g.add(even_positions, phase);
    let inverted = g.not(sums);
    g.xor(even_positions, inverted)
}
