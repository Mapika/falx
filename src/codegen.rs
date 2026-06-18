//! IR-to-Rust code generation.
//!
//! [`emit`] turns a bitstream [`Graph`] into a self-contained Rust source
//! file with no dependency on this crate: a public `index_structurals`
//! entry point that runtime-dispatches between x86 SIMD kernels. Generated
//! parser artifacts are native-code artifacts; portable scalar semantics
//! stay in the in-tree interpreter and scalar reference tests rather than
//! being emitted into every generated parser.
//!
//! Emission is a single pass over the graph: nodes are already in
//! topological order, every node becomes one `let` binding, and stateful
//! nodes (`ShiftLeft1`, `PrefixXor`, `Add`) get a slot in a carry array that
//! lives across blocks. The two kernels share these per-node lines verbatim
//! except for `Class` nodes, where the byte-comparison primitive differs.

use crate::ir::{Graph, Op};
use std::fmt::Write;

/// Codegen rejected the graph (currently only: a character class too large
/// for compare-based classification).
#[derive(Debug)]
pub struct CodegenError(pub String);

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CodegenError {}

/// Largest class emitted as an OR of byte compares; bigger classes go
/// through the shuffle-based nibble classifier instead.
const MAX_CLASS_BYTES: usize = 8;

/// Decompose a character class into PSHUFB nibble tables: byte `b` is a
/// member iff `lo_table[b & 15] & hi_table[b >> 4] != 0`.
///
/// Each of the 16 hi-nibble rows of the 16x16 membership grid is a set of
/// lo nibbles; rows sharing the same set share one table bit, so the
/// decomposition is exact whenever the class has at most 8 *distinct*
/// non-empty row patterns (any class built from ASCII separators easily
/// qualifies). Returns None when it does not.
fn nibble_tables(class: &crate::ir::CharClass) -> Option<([u8; 16], [u8; 16])> {
    let mut rows = [0u16; 16];
    for byte in class.members() {
        rows[(byte >> 4) as usize] |= 1 << (byte & 15);
    }
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    let mut patterns: Vec<u16> = Vec::new();
    for (h, &row) in rows.iter().enumerate() {
        if row == 0 {
            continue;
        }
        let bit = match patterns.iter().position(|&p| p == row) {
            Some(i) => i,
            None => {
                patterns.push(row);
                patterns.len() - 1
            }
        };
        if bit >= 8 {
            return None;
        }
        hi[h] |= 1 << bit;
        for (l, lo_entry) in lo.iter_mut().enumerate() {
            if row & (1 << l) != 0 {
                *lo_entry |= 1 << bit;
            }
        }
    }
    Some((lo, hi))
}

/// The cell type of a projected column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    /// Parsed exactly like `str::parse::<i64>`; SWAR fast path.
    I64,
    /// Parsed exactly like `str::parse::<f64>`; Clinger fast path with a
    /// `str::parse` fallback for the hard cases.
    F64,
    /// Zero-copy `(start, end)` byte spans into the input, quotes and
    /// escapes intact.
    Bytes,
    /// Cleaned cell bytes (quotes stripped, escapes resolved) materialized
    /// into an Arrow varbinary-layout pair: `<name>_offsets: Vec<i32>`
    /// (rows + 1 entries) plus a contiguous `<name>_data: Vec<u8>`.
    /// A missing field is invalid; an empty cell is a valid empty string.
    Str,
}

/// One requested typed column: project field `index` of every record.
#[derive(Clone, Debug)]
pub struct Column {
    /// Zero-based field index within a record.
    pub index: usize,
    /// Generated struct field name; defaults to `c{index}`.
    pub name: Option<String>,
    pub ty: ColumnType,
}

impl Column {
    fn field_name(&self) -> String {
        match &self.name {
            Some(name) => name.clone(),
            None => format!("c{}", self.index),
        }
    }

    /// Element type of the values Vec; `Str` columns have no single values
    /// Vec (offsets + data buffers instead) and never ask for one.
    fn rust_type(&self) -> &'static str {
        match self.ty {
            ColumnType::I64 => "i64",
            ColumnType::F64 => "f64",
            ColumnType::Bytes => "(u32, u32)",
            // String columns use the Arrow varbinary layout (offsets + data buffers),
            // so they never go through the scalar value-type/zero-placeholder path.
            ColumnType::Str => unreachable!("string columns emit offsets + data"),
        }
    }

    fn zero(&self) -> &'static str {
        match self.ty {
            ColumnType::I64 => "0",
            ColumnType::F64 => "0.0",
            ColumnType::Bytes => "(0, 0)",
            // String columns use the Arrow varbinary layout (offsets + data buffers),
            // so they never go through the scalar value-type/zero-placeholder path.
            ColumnType::Str => unreachable!("string columns emit offsets + data"),
        }
    }
}

/// Source used to build the parser graph before native backend emission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphSource {
    /// Use the handwritten graph builder in `formats`.
    Manual,
    /// Use weighted synthesis for supported dialects and the handwritten
    /// graph builder for dialects the synthesizer cannot express yet.
    AutoWeighted(crate::synth_formats::SynthProfile),
    /// Use weighted synthesis to build the graph, then emit the same native
    /// SIMD backend as manual graph generation.
    SynthWeighted(crate::synth_formats::SynthProfile),
}

/// Options for parser code generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodegenOptions {
    pub graph_source: GraphSource,
    pub graph_optimizer: GraphOptimizer,
}

/// Graph optimization pass applied after graph-source selection and before
/// backend emission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphOptimizer {
    /// Emit the selected graph as-is. Used for baseline comparisons.
    Disabled,
    /// Run the deterministic cost-weighted graph simplifier using the AVX2
    /// cost model.
    CostWeightedAvx2,
    /// Run the equality-saturation optimizer ([`crate::egraph`]) using the
    /// AVX2 cost model: a superset of the `CostWeightedAvx2` rewrites that
    /// extracts the globally cheapest graph rather than the cheaper of two
    /// whole-graph candidates.
    EqSat,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            graph_source: GraphSource::AutoWeighted(crate::synth_formats::SynthProfile::Weighted),
            graph_optimizer: GraphOptimizer::EqSat,
        }
    }
}

/// Field names the generated `Columns` struct reserves for itself.
const RESERVED_FIELDS: &[&str] = &["data", "rows"];

fn validate_columns(columns: &[Column]) -> Result<(), CodegenError> {
    // The sink's found-mask is a u32, one bit per declared column.
    if columns.len() > 32 {
        return Err(CodegenError(format!(
            "{} columns declared; the projection sink supports at most 32",
            columns.len()
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for column in columns {
        let name = column.field_name();
        let mut chars = name.chars();
        let head_ok = chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        if !head_ok || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(CodegenError(format!(
                "column name '{name}' is not a valid identifier"
            )));
        }
        if RESERVED_FIELDS.contains(&name.as_str()) {
            return Err(CodegenError(format!(
                "column name '{name}' is reserved by the generated struct"
            )));
        }
        // Each column claims its whole derived-field namespace (values,
        // validity, and the string-layout buffers) regardless of type, so
        // no pair of declarations can make the struct uncompilable.
        let claims = [
            name.clone(),
            format!("{name}_valid"),
            format!("{name}_offsets"),
            format!("{name}_data"),
        ];
        if claims.into_iter().any(|claim| !seen.insert(claim)) {
            return Err(CodegenError(format!(
                "column name '{name}' collides with another column"
            )));
        }
    }
    Ok(())
}

fn validate_nesting(dialect: &crate::formats::Dialect) -> Result<(), CodegenError> {
    let mut seen = std::collections::HashSet::new();
    for &(open, close) in &dialect.nesting {
        if open == close {
            return Err(CodegenError(format!(
                "nesting pair has identical open and close byte 0x{open:02x}"
            )));
        }
        for byte in [open, close] {
            if !dialect.structural.contains(&byte) {
                return Err(CodegenError(format!(
                    "nesting byte 0x{byte:02x} is not in the structural set; \
                     the indexer only reports bytes it classifies"
                )));
            }
            if Some(byte) == dialect.quote || Some(byte) == dialect.comment {
                return Err(CodegenError(format!(
                    "nesting byte 0x{byte:02x} conflicts with the quote or \
                     comment byte"
                )));
            }
            if byte == b'\n' {
                return Err(CodegenError(
                    "'\\n' cannot nest; it is the record-terminator class".into(),
                ));
            }
            if !seen.insert(byte) {
                return Err(CodegenError(format!(
                    "nesting byte 0x{byte:02x} appears in more than one pair"
                )));
            }
        }
    }
    Ok(())
}

/// Which kernel a step body is being emitted for.
#[derive(Clone, Copy, PartialEq)]
enum Flavor {
    Avx512,
    Avx2,
    /// ARM NEON (aarch64). A 64-byte block is four 128-bit `uint8x16_t`
    /// vectors `b0..b3`; classification reduces them to the same `u64` mask
    /// as the x86 flavors, so all mask-level logic downstream is shared.
    Neon,
}

/// The `#[target_feature]` body attribute every NEON kernel fn carries. NEON
/// (Advanced SIMD) is baseline on aarch64; `aes` gates `vmull_p64`, the PMULL
/// carryless multiply that stands in for x86 PCLMULQDQ in `prefix_xor`. As on
/// x86 (where every fn carries `pclmulqdq`), we require it uniformly so all
/// intra-module calls are target-feature supersets.
const NEON_ATTR: &str = "    #[target_feature(enable = \"neon\", enable = \"aes\")]\n";

/// The AVX2 driver attribute and its NEON counterpart. Driver bodies (the
/// tape/seeded/cells/… kernels) only ever touch `u64` masks via `step`, so
/// they are byte-identical across flavors apart from this attribute — a NEON
/// driver is its AVX2 sibling with the attribute swapped (see [`neon_driver`]).
const AVX2_ATTR_INNER: &str = "enable = \"avx2\", enable = \"pclmulqdq\"";
const NEON_ATTR_INNER: &str = "enable = \"neon\", enable = \"aes\"";

/// Build a NEON driver kernel from its already-emitted AVX2 twin by swapping
/// the target-feature attribute. Returns an empty string unchanged (drivers
/// that a dialect does not emit), so it composes with the `if … {} else {
/// String::new() }` driver bindings.
fn neon_driver(avx2: &str) -> String {
    avx2.replace(AVX2_ATTR_INNER, NEON_ATTR_INNER)
}

/// Inject an aarch64/NEON sibling after every x86 AVX2 runtime-dispatch block
/// in the generated `code`. Each dispatch fn tries AVX-512, then AVX2, then
/// `unsupported_cpu()`; the AVX2 block is uniform across all of them, so we
/// clone it, retarget it to aarch64 (`neon`+`aes` detection, `neon::` calls),
/// and splice it in just before the panic. The block always closes at the
/// first 4-space-indented `}` after its header — the call line and any inner
/// `unsafe {…}` are indented deeper — which makes the boundary unambiguous.
fn add_neon_dispatch(code: &str) -> String {
    const MARKER: &str = "    #[cfg(target_arch = \"x86_64\")]\n    if std::arch::is_x86_feature_detected!(\"avx2\")\n        && std::arch::is_x86_feature_detected!(\"pclmulqdq\")\n    {\n";
    const CLOSE: &str = "\n    }";
    let mut out = String::with_capacity(code.len() + code.len() / 16);
    let mut rest = code;
    while let Some(start) = rest.find(MARKER) {
        let after = start + MARKER.len();
        let close = rest[after..]
            .find(CLOSE)
            .expect("AVX2 dispatch block must close at a 4-space `}`");
        let end = after + close + CLOSE.len();
        let neon_block = rest[start..end]
            .replace("target_arch = \"x86_64\"", "target_arch = \"aarch64\"")
            .replace(
                "is_x86_feature_detected!(\"avx2\")",
                "is_aarch64_feature_detected!(\"neon\")",
            )
            .replace(
                "is_x86_feature_detected!(\"pclmulqdq\")",
                "is_aarch64_feature_detected!(\"aes\")",
            )
            .replace("avx2::", "neon::");
        out.push_str(&rest[..end]);
        out.push('\n');
        out.push_str(&neon_block);
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// Emit a generated source file exposing only the structural indexer.
pub fn emit(graph: &Graph, format_name: &str) -> Result<String, CodegenError> {
    emit_with(graph, format_name, None, &[])
}

/// Emit a full generated parser for a delimited dialect: the structural
/// indexer, a record-aware tape indexer (separator stream + record-end
/// stream), and a records/fields span API with quote stripping and
/// escape-aware field cleaning.
pub fn emit_parser(
    dialect: &crate::formats::Dialect,
    format_name: &str,
) -> Result<String, CodegenError> {
    emit_parser_with_columns(dialect, format_name, &[])
}

/// Like [`emit_parser`], additionally emitting a typed columnar projection
/// API (`parse_columns`, and `parse_columns_par` where the dialect supports
/// parallel indexing) for the declared `columns`.
pub fn emit_parser_with_columns(
    dialect: &crate::formats::Dialect,
    format_name: &str,
    columns: &[Column],
) -> Result<String, CodegenError> {
    emit_parser_with_columns_options(dialect, format_name, columns, CodegenOptions::default())
}

pub fn emit_parser_with_columns_options(
    dialect: &crate::formats::Dialect,
    format_name: &str,
    columns: &[Column],
    options: CodegenOptions,
) -> Result<String, CodegenError> {
    validate_columns(columns)?;
    validate_nesting(dialect)?;
    let parts = match options.graph_source {
        GraphSource::Manual => crate::formats::delimited_parts(dialect),
        GraphSource::AutoWeighted(profile) => {
            if crate::synth_formats::supports_weighted(dialect) {
                crate::synth_formats::synthesize_delimited_parts_with_profile(dialect, profile)
                    .map_err(|err| CodegenError(format!("synth-weighted {format_name}: {err}")))?
            } else {
                crate::formats::delimited_parts(dialect)
            }
        }
        GraphSource::SynthWeighted(profile) => {
            crate::synth_formats::synthesize_delimited_parts_with_profile(dialect, profile)
                .map_err(|err| CodegenError(format!("synth-weighted {format_name}: {err}")))?
        }
    };
    let parts = match options.graph_optimizer {
        GraphOptimizer::Disabled => parts,
        GraphOptimizer::CostWeightedAvx2 => {
            crate::graph_opt::optimize_parts(parts, crate::synth::CostModel::avx2()).parts
        }
        GraphOptimizer::EqSat => {
            crate::egraph::optimize_parts(parts, crate::synth::CostModel::avx2()).parts
        }
    };
    emit_with(
        &parts.graph,
        format_name,
        Some((dialect, parts.terminators, parts.nest)),
        columns,
    )
}

/// Parser-mode emission inputs: the dialect, its record-terminator node,
/// and — when the dialect declares nesting — the live open/close bracket
/// stream nodes.
type ParserParts<'d> = (
    &'d crate::formats::Dialect,
    crate::ir::NodeId,
    Option<(crate::ir::NodeId, crate::ir::NodeId)>,
);

fn emit_with(
    graph: &Graph,
    format_name: &str,
    parser: Option<ParserParts<'_>>,
    columns: &[Column],
) -> Result<String, CodegenError> {
    let dialect = parser.map(|(d, _, _)| d);
    let nest = parser.and_then(|(_, _, n)| n);
    let output = graph.output();

    // Assign carry slots to stateful nodes, recording each slot's initial
    // value (seeded shifts start at 1, everything else at 0).
    let mut carry_slot = vec![usize::MAX; graph.nodes().len()];
    let mut carry_init: Vec<u64> = Vec::new();
    for (i, op) in graph.nodes().iter().enumerate() {
        let init = match op {
            Op::ShiftLeft1(_) | Op::PrefixXor(_) | Op::Add(_, _) | Op::Regions(..) => Some(0),
            Op::ShiftLeft1Seeded(_) => Some(1),
            _ => None,
        };
        if let Some(init) = init {
            carry_slot[i] = carry_init.len();
            carry_init.push(init);
        }
    }
    let carry_count = carry_init.len();

    let uses_eq_class = graph
        .nodes()
        .iter()
        .any(|op| matches!(op, Op::Class(c) if c.members().count() <= MAX_CLASS_BYTES));
    let uses_table_class = graph
        .nodes()
        .iter()
        .any(|op| matches!(op, Op::Class(c) if c.members().count() > MAX_CLASS_BYTES));
    let uses_prefix_xor = graph
        .nodes()
        .iter()
        .any(|op| matches!(op, Op::PrefixXor(_)));
    let uses_regions = graph.nodes().iter().any(|op| matches!(op, Op::Regions(..)));

    for op in graph.nodes() {
        if let Op::Class(class) = op {
            let n = class.members().count();
            if n > MAX_CLASS_BYTES && nibble_tables(class).is_none() {
                return Err(CodegenError(format!(
                    "character class with {n} bytes has more than 8 distinct \
                     hi-nibble row patterns and cannot be decomposed into \
                     PSHUFB tables"
                )));
            }
        }
    }

    // Pieces that differ depending on whether the graph carries state.
    // CARRY_INIT (emitted at the file root) holds each slot's stream-start
    // value: 0 for most carries, 1 for seeded shifts.
    let carry_init_const = if carry_count > 0 {
        let values: Vec<String> = carry_init.iter().map(|v| v.to_string()).collect();
        format!(
            "/// Stream-start carry values; kernels and the stream parser all\n\
             /// begin from this state.\n\
             const CARRY_INIT: [u64; {carry_count}] = [{}];\n\n",
            values.join(", ")
        )
    } else {
        String::new()
    };
    let carry_decl = if carry_count > 0 {
        "        let mut carries = super::CARRY_INIT;\n".to_string()
    } else {
        String::new()
    };
    let carry_param = if carry_count > 0 {
        format!(", carries: &mut [u64; {carry_count}]")
    } else {
        String::new()
    };
    let carry_arg = if carry_count > 0 {
        ", &mut carries"
    } else {
        ""
    };

    // In parser mode the step function also returns the record-terminator
    // subset of the structural mask, so tape indexing gets record boundaries
    // for free; the plain indexer selects the first element.
    let step_ret_ty = if parser.is_some() {
        "(u64, u64)"
    } else {
        "u64"
    };
    let sel = if parser.is_some() { ".0" } else { "" };
    let step_ret = match parser {
        // When every structural byte is a record terminator (e.g. a
        // newline-only framing dialect), the terminator node is the output
        // node itself — emit it directly rather than a redundant `v & v`.
        Some((_, term, _)) if term == output => format!("(v{out}, v{out})", out = output.0),
        Some((_, term, _)) => format!("(v{out}, v{out} & v{term})", out = output.0, term = term.0),
        None => format!("v{}", output.0),
    };
    // Per-variant return roots: each emitted step variant prunes the graph
    // to the nodes its own return tuple actually needs.
    let step_roots: Vec<crate::ir::NodeId> = match parser {
        Some((_, term, _)) => vec![output, term],
        None => vec![output],
    };
    let nested_roots: Vec<crate::ir::NodeId> = match nest {
        Some((opens, closes)) => vec![output, opens, closes],
        None => Vec::new(),
    };

    // Parallel indexing (emitted for doubled-quote/no-escape dialects):
    // a chunk's entry state is one bit — the parity of quote bytes before
    // it — so a counting prepass makes chunks independent. Comment-without-
    // quote dialects use line ownership instead.
    let par_mode = matches!(
        dialect,
        Some(d) if d.escape == crate::formats::Escape::None
            && !(d.comment.is_some() && d.quote.is_some())
    );
    // Comment+quote dialects (csv_hash): region state crosses chunk boundaries
    // and is not XOR-linear (a quote can hide a comment-start and vice versa),
    // so neither the quote-parity nor the line-ownership scheme applies. They
    // get a dedicated `parse_par` that indexes each chunk speculatively in the
    // NORMAL region state, then reconciles the true entry state serially across
    // chunks and re-indexes the rare chunk that began mid-quote/comment.
    let region_par = matches!(
        dialect,
        Some(d) if d.escape == crate::formats::Escape::None
            && d.comment.is_some() && d.quote.is_some()
    );
    // Structural/tape and column parallelism both use line ownership for
    // comment dialects: each nonzero worker starts emitting only after the
    // first terminator it owns, and comment records are discarded by the
    // sink during flush.
    let col_par = par_mode;
    let seed_init = if carry_count == 1 {
        "        let mut carries = [seed];\n".to_string()
    } else if carry_count == 0 {
        "        let _ = seed;\n".to_string()
    } else {
        // Comment-without-quote dialects carry [line-start, region-state]. The
        // parallel driver starts every worker on a fresh line, so the standard
        // CARRY_INIT entry is always correct and no seed is propagated.
        "        let _ = seed;\n        let mut carries = super::CARRY_INIT;\n".to_string()
    };
    // The final carry is each chunk's quote parity (0 outside / all-ones
    // inside a quoted region), which the parallel driver recovers for free
    // instead of running a separate counting prepass.
    let carry_ret = if carry_count == 1 { "carries[0]" } else { "0" };
    let seeded_kernel = |loads: &str, tail_loads: &str, attr: &str| {
        format!(
            r#"
    /// Like `index_structurals` but seeded with the entry quote-parity carry
    /// and an absolute base offset; returns the *final* carry so the parallel
    /// driver can recover each chunk's quote parity without a counting prepass.
{attr}    pub fn index_structurals_seeded(data: &[u8], seed: u64, base: u32, out: &mut Vec<u32>) -> u64 {{
{seed_init}        let mut offset = 0usize;
        while offset + 64 <= data.len() {{
            {loads}
            push_indexes(mask, base + offset as u32, out);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            {tail_loads}
            push_indexes(mask, base + offset as u32, out);
        }}
        {carry_ret}
    }}
"#
        )
    };
    let avx512_seeded = if par_mode {
        seeded_kernel(
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let mask = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let mask = unsafe {{ step(block.as_ptr(){carry_arg}) }}{sel} & ((1u64 << rem) - 1);"
            ),
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
        )
    } else {
        String::new()
    };
    let avx2_seeded = if par_mode {
        seeded_kernel(
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let mask = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let mask = unsafe {{ step(block.as_ptr(){carry_arg}) }}{sel} & ((1u64 << rem) - 1);"
            ),
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
        )
    } else {
        String::new()
    };
    let avx512_tape_seeded = if par_mode {
        format!(
            r#"
    /// Seeded-carry, based variant of `index_tape` for parallel parsing.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "pclmulqdq")]
    pub fn index_tape_seeded(data: &[u8], seed: u64, base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) -> u64 {{
{seed_init}        let mut offset = 0usize;
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};
            push_tape(mask, term, base + offset as u32, seps, ends);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};
            push_tape(mask & live, term & live, base + offset as u32, seps, ends);
        }}
        {carry_ret}
    }}
"#
        )
    } else {
        String::new()
    };
    let avx2_tape_seeded = if par_mode {
        format!(
            r#"
    /// Seeded-carry, based variant of `index_tape` for parallel parsing.
    #[target_feature(enable = "avx2", enable = "pclmulqdq")]
    pub fn index_tape_seeded(data: &[u8], seed: u64, base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) -> u64 {{
{seed_init}        let mut offset = 0usize;
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};
            push_tape(mask, term, base + offset as u32, seps, ends);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};
            push_tape(mask & live, term & live, base + offset as u32, seps, ends);
        }}
        {carry_ret}
    }}
"#
        )
    } else {
        String::new()
    };
    let quote_parity_mode = par_mode
        && dialect.is_some_and(|d| d.quote.is_some())
        && carry_count > 0
        && (!columns.is_empty() || format_name == "csv");
    let quote_parity_dispatch = if quote_parity_mode {
        r#"fn quote_parity_dispatch(data: &[u8], seed: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        return unsafe { avx512::quote_parity(data, seed) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        return unsafe { avx2::quote_parity(data, seed) };
    }
    unsupported_cpu();
}

"#
        .to_string()
    } else {
        String::new()
    };
    let quote_parity_tpl = r#"
    /// Scan a chunk only far enough to recover the generated quote-region
    /// carry. Used by parallel fused stats paths to seed workers without a
    /// scalar quote-count prepass.
@ATTR@    pub fn quote_parity(data: &[u8], seed: u64) -> u64 {
        let mut carries = super::CARRY_INIT;
        carries[0] = seed;
        let mut offset = 0usize;
        while offset + 64 <= data.len() {
@LOAD@            offset += 64;
        }
        let rem = data.len() - offset;
        if rem > 0 {
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
@LOAD2@        }
        carries[0]
    }
"#;
    let avx512_quote_parity = if quote_parity_mode {
        quote_parity_tpl
            .replace("@ATTR@", "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n")
            .replace("@LOAD@", &format!("            // SAFETY: offset + 64 <= data.len().\n            let _ = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};\n"))
            .replace("@LOAD2@", &format!("            // SAFETY: block is a readable 64-byte buffer.\n            let _ = unsafe {{ step(block.as_ptr(){carry_arg}) }};\n"))
    } else {
        String::new()
    };
    let avx2_quote_parity = if quote_parity_mode {
        quote_parity_tpl
            .replace("@ATTR@", "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n")
            .replace("@LOAD@", &format!("            // SAFETY: offset + 64 <= data.len().\n            let _ = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};\n"))
            .replace("@LOAD2@", &format!("            // SAFETY: block is a readable 64-byte buffer.\n            let _ = unsafe {{ step(block.as_ptr(){carry_arg}) }};\n"))
    } else {
        String::new()
    };
    // Fused projection drivers (only when columns are declared): identical
    // loop shape to index_tape, but masks feed the column sink directly —
    // nothing is materialized in between. Parallel-capable dialects get a
    // seeded variant that can start mid-stream and stops once the sink has
    // finished its assigned record range.
    let has_columns = parser.is_some() && !columns.is_empty();
    let cells_tpl = r#"
    /// Fused projection driver: structural masks go straight into the
    /// column sink; no tape is materialized.@DOC@
@ATTR@    pub(crate) fn index_cells(@PARAMS@sink: &mut super::ColumnSink) {
@INIT@@START@        while offset + 64 <= data.len() {
            @LOAD@
            sink.drive(mask, term, offset as u32);
            if sink.done {
                return;
            }
            offset += 64;
        }
        let rem = data.len() - offset;
        if rem > 0 {
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            @LOAD2@
            sink.drive(mask & live, term & live, offset as u32);
        }
    }
"#;
    let cells_fill = |attr: &str, load: &str, load2: &str| -> String {
        let (doc, params, init, start) = if col_par {
            (
                " Scans 64-byte blocks from\n    /// `start` (block-aligned) onward, until end of data or until the\n    /// sink completes its record range.",
                "data: &[u8], seed: u64, start: usize, ",
                seed_init.as_str(),
                "        let mut offset = start;\n",
            )
        } else {
            (
                "",
                "data: &[u8], ",
                carry_decl.as_str(),
                "        let mut offset = 0usize;\n",
            )
        };
        cells_tpl
            .replace("@DOC@", doc)
            .replace("@ATTR@", attr)
            .replace("@PARAMS@", params)
            .replace("@INIT@", init)
            .replace("@START@", start)
            .replace("@LOAD@", load)
            .replace("@LOAD2@", load2)
    };
    let avx512_cells = if has_columns {
        cells_fill(
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
        )
    } else {
        String::new()
    };
    let avx2_cells = if has_columns {
        cells_fill(
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
        )
    } else {
        String::new()
    };
    let vcf_stats_mode = format_name == "vcf_typed";
    let vcf_stats_cells = |code: String| {
        code.replace("index_cells", "index_vcf_stats")
            .replace("ColumnSink", "VcfStatsSink")
            .replace("column sink", "VCF stats sink")
    };
    let avx512_vcf_stats = if vcf_stats_mode {
        vcf_stats_cells(cells_fill(
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
        ))
    } else {
        String::new()
    };
    let avx2_vcf_stats = if vcf_stats_mode {
        vcf_stats_cells(cells_fill(
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
        ))
    } else {
        String::new()
    };
    let csv_geo_stats_mode = format_name == "csv_geo" || format_name == "csv_geo_text";
    let csv_geo_stats_index = if format_name == "csv_geo_text" {
        "index_csv_geo_text_stats"
    } else {
        "index_csv_geo_stats"
    };
    let csv_geo_stats_sink = if format_name == "csv_geo_text" {
        "CsvGeoTextStatsSink"
    } else {
        "CsvGeoStatsSink"
    };
    let csv_geo_stats_doc = if format_name == "csv_geo_text" {
        "CSV geo text stats sink"
    } else {
        "CSV geo stats sink"
    };
    let csv_geo_stats_cells = |code: String| {
        code.replace("index_cells", csv_geo_stats_index)
            .replace("ColumnSink", csv_geo_stats_sink)
            .replace("column sink", csv_geo_stats_doc)
    };
    let avx512_csv_geo_stats = if csv_geo_stats_mode {
        csv_geo_stats_cells(cells_fill(
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
        ))
    } else {
        String::new()
    };
    let avx2_csv_geo_stats = if csv_geo_stats_mode {
        csv_geo_stats_cells(cells_fill(
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
        ))
    } else {
        String::new()
    };
    let field_byte_stats_mode = format_name == "csv" || format_name == "tsv";
    let field_byte_stats_cells = |code: String| {
        code.replace("index_cells", "index_field_bytes")
            .replace("ColumnSink", "FieldByteSink")
            .replace("column sink", "field-byte stats sink")
    };
    let avx512_field_byte_stats = if field_byte_stats_mode {
        field_byte_stats_cells(cells_fill(
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
        ))
    } else {
        String::new()
    };
    let avx2_field_byte_stats = if field_byte_stats_mode {
        field_byte_stats_cells(cells_fill(
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
            &format!(
                "// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
        ))
    } else {
        String::new()
    };
    let carry_fwd = if carry_count > 0 { ", carries" } else { "" };
    // Streaming building block: index only the full blocks of a slice with
    // caller-owned carries; the sub-block remainder stays unconsumed.
    let partial_tpl = r#"
    /// Index the full 64-byte blocks of `data` (carries persist across
    /// calls); returns the number of bytes consumed. Streaming primitive.
@ATTR@    pub fn index_tape_partial(data: &[u8], carries: &mut [u64; @K@], base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {
        let _ = &carries;
        let mut offset = 0usize;
        while offset + 64 <= data.len() {
@LOAD@            push_tape(mask, term, base + offset as u32, seps, ends);
            offset += 64;
        }
    }

    /// Index one final zero-padded block (end-of-stream only); `live`
    /// masks off the padding bits.
@ATTR@    pub fn index_tape_block(block: &[u8; 64], live: u64, carries: &mut [u64; @K@], base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {
        let _ = &carries;
@LOAD2@        push_tape(mask & live, term & live, base, seps, ends);
    }
"#;
    // The streaming primitives (index_tape_partial/_block + their dispatchers)
    // exist only to feed the per-line StreamParser. A grouped (lines_per_record
    // > 1) dialect emits no StreamParser, so they would be dead — gate them off.
    let stream_enabled = !matches!(dialect, Some(d) if d.lines_per_record > 1);
    let avx512_partial = if parser.is_some() && stream_enabled {
        partial_tpl
            .replace("@ATTR@", "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n")
            .replace("@K@", &carry_count.to_string())
            .replace("@LOAD@", &format!("            // SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_fwd}) }};\n"))
            .replace("@LOAD2@", &format!("        // SAFETY: block is a readable 64-byte buffer.\n        let (mask, term) = unsafe {{ step(block.as_ptr(){carry_fwd}) }};\n"))
    } else {
        String::new()
    };
    let avx2_partial = if parser.is_some() && stream_enabled {
        partial_tpl
            .replace("@ATTR@", "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n")
            .replace("@K@", &carry_count.to_string())
            .replace("@LOAD@", &format!("            // SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_fwd}) }};\n"))
            .replace("@LOAD2@", &format!("        // SAFETY: block is a readable 64-byte buffer.\n        let (mask, term) = unsafe {{ step(block.as_ptr(){carry_fwd}) }};\n"))
    } else {
        String::new()
    };
    // Comment+quote dialects also emit a no-tape region-state scan, used by the
    // transfer-function parallel parse to learn each chunk's exit region state
    // for all three entry states without writing the tape.
    let (avx512_partial, avx2_partial, neon_partial) = if region_par {
        // `region_scan3`: a single no-tape pass that computes the region inputs
        // (q, s, n) once per block and advances all three candidate region
        // states (the transfer function's three entry states), instead of three
        // separate full scans. Its per-block body is the q/s/n subgraph —
        // emitted by `emit_step_body` rooted at the `Regions` operands, so only
        // their upstream cone (not the Regions node or the structural output) is
        // produced — followed by three `resolve_regions` state advances.
        let (rq, rs, rn) = graph
            .nodes()
            .iter()
            .find_map(|op| match op {
                Op::Regions(q, s, n) => Some((*q, *s, *n)),
                _ => None,
            })
            .expect("region_par implies a Regions node");
        let emit_scan3 = |flavor: Flavor, attr: &str| -> String {
            // Per-block body: define v{rq}, v{rs}, v{rn}, then advance 3 states.
            let mut block = String::new();
            emit_step_body(&mut block, graph, &carry_slot, flavor, &[rq, rs, rn]);
            let _ = write!(
                block,
                "        let _ = resolve_regions(v{q}, v{s}, v{n}, &mut states[0]);\n\
                 \x20       let _ = resolve_regions(v{q}, v{s}, v{n}, &mut states[1]);\n\
                 \x20       let _ = resolve_regions(v{q}, v{s}, v{n}, &mut states[2]);\n",
                q = rq.0,
                s = rs.0,
                n = rn.0,
            );
            // Block loads bind the per-flavor step inputs (`lo, hi` on x86;
            // `b0..b3` on NEON); the classification body in `block` consumes
            // whichever names this flavor produced.
            let (full_load, tail_load) = match flavor {
                Flavor::Avx512 | Flavor::Avx2 => (
                    "            let (lo, hi) = unsafe { (_mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i), _mm256_loadu_si256(data.as_ptr().add(offset + 32) as *const __m256i)) };\n",
                    "            let (lo, hi) = unsafe { (_mm256_loadu_si256(blk.as_ptr() as *const __m256i), _mm256_loadu_si256(blk.as_ptr().add(32) as *const __m256i)) };\n",
                ),
                Flavor::Neon => (
                    "            let (b0, b1, b2, b3) = unsafe { (vld1q_u8(data.as_ptr().add(offset)), vld1q_u8(data.as_ptr().add(offset + 16)), vld1q_u8(data.as_ptr().add(offset + 32)), vld1q_u8(data.as_ptr().add(offset + 48))) };\n",
                    "            let (b0, b1, b2, b3) = unsafe { (vld1q_u8(blk.as_ptr()), vld1q_u8(blk.as_ptr().add(16)), vld1q_u8(blk.as_ptr().add(32)), vld1q_u8(blk.as_ptr().add(48))) };\n",
                ),
            };
            let mut f = String::new();
            f.push_str(attr);
            f.push_str("    pub fn region_scan3(data: &[u8], carries: &mut [u64; 2], states: &mut [u64; 3]) {\n");
            f.push_str("        let mut offset = 0usize;\n");
            f.push_str("        while offset + 64 <= data.len() {\n");
            f.push_str("            // SAFETY: offset + 64 <= data.len().\n");
            f.push_str(full_load);
            f.push_str(&block);
            f.push_str("            offset += 64;\n");
            f.push_str("        }\n");
            f.push_str("        let rem = data.len() - offset;\n");
            f.push_str("        if rem > 0 {\n");
            f.push_str("            let mut blk = [0u8; 64];\n");
            f.push_str("            blk[..rem].copy_from_slice(&data[offset..]);\n");
            f.push_str("            // SAFETY: blk is a readable 64-byte buffer (zero padded).\n");
            f.push_str(tail_load);
            f.push_str(&block);
            f.push_str("        }\n");
            f.push_str("    }\n");
            f
        };
        (
            format!(
                "{avx512_partial}\n{}",
                emit_scan3(
                    Flavor::Avx512,
                    "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n"
                )
            ),
            format!(
                "{avx2_partial}\n{}",
                emit_scan3(
                    Flavor::Avx2,
                    "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n"
                )
            ),
            // NEON: the streaming partial body is attribute-swapped from AVX2,
            // but region_scan3 inlines its own per-flavor block loads, so it is
            // emitted fresh rather than attribute-swapped.
            format!(
                "{}\n{}",
                neon_driver(&avx2_partial),
                emit_scan3(Flavor::Neon, NEON_ATTR)
            ),
        )
    } else {
        let neon_partial = neon_driver(&avx2_partial);
        (avx512_partial, avx2_partial, neon_partial)
    };

    let par_block = if par_mode {
        // One per-chunk tape part; aliased to keep the parallel driver's
        // collection types within clippy's complexity budget.
        let tape_part = "/// One chunk's tape: separator positions and end entries.\ntype TapePart = (Vec<u32>, Vec<u64>);";
        // Comment-without-quote dialects (VCF/BED/GFF/SAM/MTX …) parallelize by
        // line ownership; quote/plain dialects by speculative entry-parity.
        let has_comment = dialect.is_some_and(|d| d.comment.is_some());
        let index_par_body = if has_comment {
            COMMENT_INDEX_PAR
        } else {
            SPEC_INDEX_PAR
        };
        let parse_par_body = if has_comment {
            COMMENT_PARSE_PAR
        } else {
            SPEC_PARSE_PAR
        };
        format!(
            r#"
{tape_part}

{index_par_body}

/// Append per-chunk index parts to `out`, copying each into its own disjoint
/// slot concurrently. The previous single-threaded concat serialized an
/// O(positions) copy and was the parallel scaling ceiling.
fn scatter_u32(out: &mut Vec<u32>, parts: &[Vec<u32>]) {{
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let start = out.len();
    out.reserve(total);
    {{
        let mut rest = &mut out.spare_capacity_mut()[..total];
        let mut slots: Vec<&mut [std::mem::MaybeUninit<u32>]> = Vec::with_capacity(parts.len());
        for p in parts {{
            let (head, tail) = rest.split_at_mut(p.len());
            slots.push(head);
            rest = tail;
        }}
        std::thread::scope(|s| {{
            for (slot, part) in slots.into_iter().zip(parts.iter()) {{
                s.spawn(move || {{
                    // SAFETY: slot.len() == part.len(); the copy initializes
                    // exactly this disjoint slice of `out`'s spare capacity.
                    unsafe {{
                        std::ptr::copy_nonoverlapping(
                            part.as_ptr(),
                            slot.as_mut_ptr().cast::<u32>(),
                            part.len(),
                        );
                    }}
                }});
            }}
        }});
    }}
    // SAFETY: the scatter initialized every element of spare[..total].
    unsafe {{ out.set_len(start + total); }}
}}

fn index_structurals_seeded_dispatch(data: &[u8], seed: u64, base: u32, out: &mut Vec<u32>) -> u64 {{
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        return unsafe {{ avx512::index_structurals_seeded(data, seed, base, out) }};
    }}
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        return unsafe {{ avx2::index_structurals_seeded(data, seed, base, out) }};
    }}
    unsupported_cpu()
}}

{parse_par_body}

/// Concatenate per-chunk tape parts into the master tape, copying each into
/// its own disjoint slot concurrently. Separator positions and end byte
/// offsets are already absolute; only each end entry's cumulative-separator
/// high word is rebased by the separators in the preceding chunks.
fn scatter_tape(parts: &[TapePart]) -> (Vec<u32>, Vec<u64>) {{
    let sep_total: usize = parts.iter().map(|p| p.0.len()).sum();
    let end_total: usize = parts.iter().map(|p| p.1.len()).sum();
    let mut seps: Vec<u32> = Vec::with_capacity(sep_total);
    let mut ends: Vec<u64> = Vec::with_capacity(end_total);
    // Cumulative separator count before each chunk (the ends rebase amount).
    let mut sep_prefix: Vec<u64> = Vec::with_capacity(parts.len());
    {{
        let mut acc = 0u64;
        for p in parts {{
            sep_prefix.push(acc);
            acc += p.0.len() as u64;
        }}
    }}
    {{
        let mut srest = &mut seps.spare_capacity_mut()[..sep_total];
        let mut erest = &mut ends.spare_capacity_mut()[..end_total];
        let mut sslots: Vec<&mut [std::mem::MaybeUninit<u32>]> = Vec::with_capacity(parts.len());
        let mut eslots: Vec<&mut [std::mem::MaybeUninit<u64>]> = Vec::with_capacity(parts.len());
        for p in parts {{
            let (sh, st) = srest.split_at_mut(p.0.len());
            sslots.push(sh);
            srest = st;
            let (eh, et) = erest.split_at_mut(p.1.len());
            eslots.push(eh);
            erest = et;
        }}
        std::thread::scope(|s| {{
            for (((sslot, eslot), part), &prefix) in
                sslots.into_iter().zip(eslots).zip(parts.iter()).zip(sep_prefix.iter())
            {{
                s.spawn(move || {{
                    // SAFETY: each slot's length equals its part's; the writes
                    // initialize exactly these disjoint slices.
                    unsafe {{
                        std::ptr::copy_nonoverlapping(
                            part.0.as_ptr(),
                            sslot.as_mut_ptr().cast::<u32>(),
                            part.0.len(),
                        );
                    }}
                    let rebase = prefix << 32;
                    let dst = eslot.as_mut_ptr().cast::<u64>();
                    for (i, &e) in part.1.iter().enumerate() {{
                        // SAFETY: i < part.1.len() == eslot.len().
                        unsafe {{ *dst.add(i) = e + rebase; }}
                    }}
                }});
            }}
        }});
    }}
    // SAFETY: the scatter initialized every element of both spare regions.
    unsafe {{
        seps.set_len(sep_total);
        ends.set_len(end_total);
    }}
    (seps, ends)
}}

fn index_tape_seeded_dispatch(data: &[u8], seed: u64, base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) -> u64 {{
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        return unsafe {{ avx512::index_tape_seeded(data, seed, base, seps, ends) }};
    }}
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        return unsafe {{ avx2::index_tape_seeded(data, seed, base, seps, ends) }};
    }}
    unsupported_cpu()
}}
"#
        )
    } else if region_par {
        REGION_PARSE_PAR.to_string()
    } else {
        String::new()
    };

    // Record-aware tape indexers, emitted only in parser mode. Identical in
    // both SIMD kernels except for target features and class primitives.
    let avx512_tape = if parser.is_some() {
        format!(
            r#"
    /// Record-aware indexing; structural and terminator masks encode the tape.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "pclmulqdq")]
    pub fn index_tape(data: &[u8], seps: &mut Vec<u32>, ends: &mut Vec<u64>) {{
{carry_decl}        let mut offset = 0usize;
        // Two blocks per iteration (see index_structurals).
        while offset + 128 <= data.len() {{
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let (m0, t0) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};
            push_tape(m0, t0, offset as u32, seps, ends);
            let (m1, t1) = unsafe {{ step(data.as_ptr().add(offset + 64){carry_arg}) }};
            push_tape(m1, t1, (offset + 64) as u32, seps, ends);
            offset += 128;
        }}
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};
            push_tape(mask, term, offset as u32, seps, ends);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};
            push_tape(mask & live, term & live, offset as u32, seps, ends);
        }}
    }}

    fn push_tape(structural: u64, term: u64, base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {{
        let sep_mask = structural & !term;
        let base_count = seps.len() as u64;
        push_indexes(sep_mask, base, seps);
        let mut t = term;
        while t != 0 {{
            let idx = t.trailing_zeros();
            let below = (sep_mask & ((1u64 << idx) - 1)).count_ones() as u64;
            ends.push(((base_count + below) << 32) | (base + idx) as u64);
            t &= t - 1;
        }}
    }}
"#
        )
    } else {
        String::new()
    };
    let avx2_tape = if parser.is_some() {
        format!(
            r#"
    /// Record-aware indexing; structural and terminator masks encode the tape.
    #[target_feature(enable = "avx2", enable = "pclmulqdq")]
    pub fn index_tape(data: &[u8], seps: &mut Vec<u32>, ends: &mut Vec<u64>) {{
{carry_decl}        let mut offset = 0usize;
        // Two blocks per iteration (see index_structurals).
        while offset + 128 <= data.len() {{
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let (m0, t0) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};
            push_tape(m0, t0, offset as u32, seps, ends);
            let (m1, t1) = unsafe {{ step(data.as_ptr().add(offset + 64){carry_arg}) }};
            push_tape(m1, t1, (offset + 64) as u32, seps, ends);
            offset += 128;
        }}
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};
            push_tape(mask, term, offset as u32, seps, ends);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};
            push_tape(mask & live, term & live, offset as u32, seps, ends);
        }}
    }}

    fn push_tape(structural: u64, term: u64, base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {{
        let sep_mask = structural & !term;
        let base_count = seps.len() as u64;
        push_indexes(sep_mask, base, seps);
        let mut t = term;
        while t != 0 {{
            let idx = t.trailing_zeros();
            let below = (sep_mask & ((1u64 << idx) - 1)).count_ones() as u64;
            ends.push(((base_count + below) << 32) | (base + idx) as u64);
            t &= t - 1;
        }}
    }}
"#
        )
    } else {
        String::new()
    };

    let fastq_mode = format_name == "fastq";
    let fastq_tpl = r#"
    /// Fused FASTQ stats driver: newline masks come from the generated
    /// line-kernel step and feed the FASTQ validation/stat sink directly.
@BLOCK_ATTR@    pub fn fastq_blocks(data: &[u8], sink: &mut super::FastqSink) -> Result<(), super::FastqError> {
@CARRY_DECL@@ACC_INIT@
        let mut offset = 0usize;
        while offset + 128 <= data.len() {
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let (m0, _) = unsafe { step(data.as_ptr().add(offset)@CARRY_ARG@) };
            let (m1, _) = unsafe { step(data.as_ptr().add(offset + 64)@CARRY_ARG@) };
            let error = unsafe { sink.@DRIVE_PAIR@(data, m0, m1, offset, &mut checksum_acc) };
            if error != 0 {
                return Err(sink.take_error(error));
            }
            offset += 128;
        }
        while offset + 64 <= data.len() {
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (mask, _) = unsafe { step(data.as_ptr().add(offset)@CARRY_ARG@) };
            let error = unsafe { sink.@DRIVE@(data, mask, offset, &mut checksum_acc) };
            if error != 0 {
                return Err(sink.take_error(error));
            }
            offset += 64;
        }
        let rem = data.len() - offset;
        if rem > 0 {
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (mask, _) = unsafe { step(block.as_ptr()@CARRY_ARG@) };
            let error = unsafe { sink.@DRIVE@(data, mask & live, offset, &mut checksum_acc) };
            if error != 0 {
                return Err(sink.take_error(error));
            }
        }
@ACC_STORE@        Ok(())
    }

"#;
    let fastq_fill =
        |block_attr: &str, acc_init: &str, acc_store: &str, drive_pair: &str, drive: &str| {
            fastq_tpl
                .replace("@BLOCK_ATTR@", block_attr)
                .replace("@CARRY_DECL@", &carry_decl)
                .replace("@CARRY_ARG@", carry_arg)
                .replace("@ACC_INIT@", acc_init)
                .replace("@ACC_STORE@", acc_store)
                .replace("@DRIVE_PAIR@", drive_pair)
                .replace("@DRIVE@", drive)
        };
    let avx512_fastq = if fastq_mode {
        fastq_fill(
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
            "        // SAFETY: __m512i and [u64; 8] are both 64-byte vector storage here.\n        let mut checksum_acc: __m512i = unsafe {\n            std::mem::transmute::<[u64; 8], __m512i>(sink.checksum_lanes)\n        };\n",
            "        // SAFETY: __m512i and [u64; 8] are both 64-byte vector storage here.\n        sink.checksum_lanes = unsafe { std::mem::transmute::<__m512i, [u64; 8]>(checksum_acc) };\n",
            "drive_pair_avx512",
            "drive_avx512",
        )
    } else {
        String::new()
    };
    let avx2_fastq = if fastq_mode {
        fastq_fill(
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
            "        // SAFETY: __m256i and [u64; 4] are both 32-byte vector storage here.\n        let mut checksum_acc: __m256i = unsafe {\n            std::mem::transmute::<[u64; 4], __m256i>([\n                sink.checksum_lanes[0],\n                sink.checksum_lanes[1],\n                sink.checksum_lanes[2],\n                sink.checksum_lanes[3],\n            ])\n        };\n",
            "        // SAFETY: __m256i and [u64; 4] are both 32-byte vector storage here.\n        let checksum_lanes: [u64; 4] = unsafe {\n            std::mem::transmute::<__m256i, [u64; 4]>(checksum_acc)\n        };\n        sink.checksum_lanes[..4].copy_from_slice(&checksum_lanes);\n",
            "drive_pair_avx2",
            "drive_avx2",
        )
    } else {
        String::new()
    };
    // NEON's checksum accumulator is a 128-bit `uint64x2_t` (two lanes), so it
    // mirrors the AVX2 path but with a `[u64; 2]` view of `checksum_lanes`.
    let neon_fastq = if fastq_mode {
        fastq_fill(
            NEON_ATTR,
            "        // SAFETY: uint64x2_t and [u64; 2] are both 16-byte vector storage here.\n        let mut checksum_acc: uint64x2_t = unsafe {\n            std::mem::transmute::<[u64; 2], uint64x2_t>([\n                sink.checksum_lanes[0],\n                sink.checksum_lanes[1],\n            ])\n        };\n",
            "        // SAFETY: uint64x2_t and [u64; 2] are both 16-byte vector storage here.\n        let checksum_lanes: [u64; 2] = unsafe {\n            std::mem::transmute::<uint64x2_t, [u64; 2]>(checksum_acc)\n        };\n        sink.checksum_lanes[..2].copy_from_slice(&checksum_lanes);\n",
            "drive_pair_neon",
            "drive_neon",
        )
    } else {
        String::new()
    };

    let ndjson_lines_mode = format_name == "ndjson";
    let ndjson_lines_tpl = r#"
    /// Count NDJSON line terminators with the generated quote/escape-aware
    /// newline step.
@ATTR@    pub fn index_ndjson_lines(data: &[u8]) -> u64 {
@CARRY_DECL@        let mut offset = 0usize;
        let mut count = 0u64;
        while offset + 128 <= data.len() {
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let (_, t0) = unsafe { step(data.as_ptr().add(offset)@CARRY_ARG@) };
            let (_, t1) = unsafe { step(data.as_ptr().add(offset + 64)@CARRY_ARG@) };
            count += t0.count_ones() as u64 + t1.count_ones() as u64;
            offset += 128;
        }
        while offset + 64 <= data.len() {
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (_, term) = unsafe { step(data.as_ptr().add(offset)@CARRY_ARG@) };
            count += term.count_ones() as u64;
            offset += 64;
        }
        let rem = data.len() - offset;
        if rem > 0 {
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (_, term) = unsafe { step(block.as_ptr()@CARRY_ARG@) };
            count += (term & live).count_ones() as u64;
        }
        count
    }
"#;
    let ndjson_lines_fill = |attr: &str| {
        ndjson_lines_tpl
            .replace("@ATTR@", attr)
            .replace("@CARRY_DECL@", &carry_decl)
            .replace("@CARRY_ARG@", carry_arg)
    };
    let avx512_ndjson_lines = if ndjson_lines_mode {
        ndjson_lines_fill(
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
        )
    } else {
        String::new()
    };
    let avx2_ndjson_lines = if ndjson_lines_mode {
        ndjson_lines_fill("    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n")
    } else {
        String::new()
    };

    let logfmt_mode = format_name == "logfmt";
    let logfmt_tpl = r#"
    /// Fused logfmt pair-stat driver: structural masks feed the pair sink
    /// directly, avoiding the generic separator/end tape for this domain API.
@ATTR@    pub fn logfmt_blocks(data: &[u8], sink: &mut super::LogfmtSink) {
@CARRY_DECL@        let mut offset = 0usize;
        while offset + 128 <= data.len() {
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let (m0, t0) = unsafe { step(data.as_ptr().add(offset)@CARRY_ARG@) };
            sink.drive(data, m0, t0, offset);
            let (m1, t1) = unsafe { step(data.as_ptr().add(offset + 64)@CARRY_ARG@) };
            sink.drive(data, m1, t1, offset + 64);
            offset += 128;
        }
        while offset + 64 <= data.len() {
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (mask, term) = unsafe { step(data.as_ptr().add(offset)@CARRY_ARG@) };
            sink.drive(data, mask, term, offset);
            offset += 64;
        }
        let rem = data.len() - offset;
        if rem > 0 {
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (mask, term) = unsafe { step(block.as_ptr()@CARRY_ARG@) };
            sink.drive(data, mask & live, term & live, offset);
        }
    }
"#;
    let logfmt_fill = |attr: &str| {
        logfmt_tpl
            .replace("@ATTR@", attr)
            .replace("@CARRY_DECL@", &carry_decl)
            .replace("@CARRY_ARG@", carry_arg)
    };
    let avx512_logfmt = if logfmt_mode {
        logfmt_fill(
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
        )
    } else {
        String::new()
    };
    let avx2_logfmt = if logfmt_mode {
        logfmt_fill("    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n")
    } else {
        String::new()
    };

    // Fused nested-tape drivers (emitted when the dialect declares bracket
    // pairs): blocks stream straight from step() into the bracket matcher,
    // no intermediate position vector. The matcher itself (push_nested) is
    // emitted once at the file root by push_nested_api.
    let nested_mode = matches!(dialect, Some(d) if !d.nesting.is_empty());
    let entry_ty = format!("[u64; {carry_count}]");
    let nested_seed_decl = if carry_count > 0 {
        "        let mut carries = seed;\n"
    } else {
        "        let _ = seed;\n"
    };
    // The seeded driver and prepass are shared by both flavors modulo the
    // step-call shape, like the other seeded kernels.
    let entry_push = if carry_count > 0 {
        "entries.push(carries);"
    } else {
        "entries.push([0u64; 0]);"
    };
    let nested_par_kernels = |step_call: &str,
                              tail_step_call: &str,
                              prepass_step: &str,
                              prepass_tail: &str,
                              attr: &str| {
        format!(
            r#"
    /// Serial prepass for parallel nested parsing: replays the kernel over
    /// the input, snapshotting the carries entering each interior chunk
    /// boundary and counting each chunk's structural events — which is
    /// exactly its master-tape slot count. Exact for any dialect, so the
    /// parallel pass can write into disjoint master ranges directly.
{attr}    pub fn nested_prepass(data: &[u8], bounds: &[usize], entries: &mut Vec<{entry_ty}>, counts: &mut Vec<usize>) {{
{carry_decl}        let mut offset = 0usize;
        let chunks = bounds.len() - 1;
        for t in 0..chunks {{
            let bound = bounds[t + 1];
            let mut count = 0usize;
            while offset + 64 <= bound {{
                {prepass_step}
                count += mask.count_ones() as usize;
                offset += 64;
            }}
            // Only the final bound can be unaligned (== data.len()).
            if offset < bound {{
                let rem = bound - offset;
                let mut block = [0u8; 64];
                block[..rem].copy_from_slice(&data[offset..bound]);
                {prepass_tail}
                count += (mask & ((1u64 << rem) - 1)).count_ones() as usize;
                offset = bound;
            }}
            counts.push(count);
            if t + 1 < chunks {{
                {entry_push}
            }}
        }}
    }}

    /// Seeded variant of `nested_tape` for parallel parsing: entry carries
    /// come from the prepass, tape entries go directly into the master
    /// buffer at this chunk's slot range with globally-indexed partners
    /// (so no rebase or concat pass exists), and a close with no local
    /// open is recorded into `pending` — its open lives in an earlier
    /// chunk. Returns the first definite error and the entries written.
    ///
    /// # Safety
    /// `master.add(tape_base + i)` must be valid for `i` up to this
    /// chunk's prepass count, and that slot range must be owned
    /// exclusively by this call.
{attr}    pub unsafe fn nested_tape_seeded(data: &[u8], seed: {entry_ty}, pos_base: u32, master: *mut u64, tape_base: usize, stack: &mut Vec<u64>, pending: &mut Vec<u64>) -> (Option<super::NestError>, usize) {{
{nested_seed_decl}        let mut offset = 0usize;
        let mut tlen = 0usize;
        while offset + 64 <= data.len() {{
            {step_call}
            // SAFETY: forwarded from the caller's contract.
            let err = unsafe {{ super::push_nested_par(data, mask, opens, closes, offset as u32, pos_base, master, tape_base, &mut tlen, stack, pending) }};
            if err.is_some() {{
                return (err, tlen);
            }}
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            {tail_step_call}
            let live = (1u64 << rem) - 1;
            // SAFETY: forwarded from the caller's contract.
            let err = unsafe {{ super::push_nested_par(data, mask & live, opens & live, closes & live, offset as u32, pos_base, master, tape_base, &mut tlen, stack, pending) }};
            if err.is_some() {{
                return (err, tlen);
            }}
        }}
        (None, tlen)
    }}
"#
        )
    };
    let avx512_nested = if nested_mode {
        format!(
            r#"
    /// Fused nested-tape driver.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "pclmulqdq")]
    pub fn nested_tape(data: &[u8], tape: &mut Vec<u64>, stack: &mut Vec<u64>) -> Option<super::NestError> {{
{carry_decl}        let mut offset = 0usize;
        // Two blocks per iteration (see index_structurals).
        while offset + 128 <= data.len() {{
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let (m0, o0, c0) = unsafe {{ step_nested(data.as_ptr().add(offset){carry_arg}) }};
            if let Some(err) = super::push_nested(data, m0, o0, c0, offset as u32, tape, stack) {{
                return Some(err);
            }}
            let (m1, o1, c1) = unsafe {{ step_nested(data.as_ptr().add(offset + 64){carry_arg}) }};
            if let Some(err) = super::push_nested(data, m1, o1, c1, (offset + 64) as u32, tape, stack) {{
                return Some(err);
            }}
            offset += 128;
        }}
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (mask, opens, closes) = unsafe {{ step_nested(data.as_ptr().add(offset){carry_arg}) }};
            if let Some(err) = super::push_nested(data, mask, opens, closes, offset as u32, tape, stack) {{
                return Some(err);
            }}
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (mask, opens, closes) = unsafe {{ step_nested(block.as_ptr(){carry_arg}) }};
            let live = (1u64 << rem) - 1;
            if let Some(err) =
                super::push_nested(data, mask & live, opens & live, closes & live, offset as u32, tape, stack)
            {{
                return Some(err);
            }}
        }}
        None
    }}
"#
        ) + &nested_par_kernels(
            &format!(
                "// SAFETY: the while guard keeps offset + 64 within data (interior\n                // bounds are 64-aligned and at most data.len()).\n                let (mask, opens, closes) = unsafe {{ step_nested(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n                let (mask, opens, closes) = unsafe {{ step_nested(block.as_ptr(){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: the while guard keeps offset + 64 within data (interior\n                // bounds are 64-aligned and at most data.len()).\n                let (mask, _) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n                let (mask, _) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
            "    #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n",
        )
    } else {
        String::new()
    };
    let avx2_nested = if nested_mode {
        format!(
            r#"
    /// Fused nested-tape driver.
    #[target_feature(enable = "avx2", enable = "pclmulqdq")]
    pub fn nested_tape(data: &[u8], tape: &mut Vec<u64>, stack: &mut Vec<u64>) -> Option<super::NestError> {{
{carry_decl}        let mut offset = 0usize;
        // Two blocks per iteration (see index_structurals).
        while offset + 128 <= data.len() {{
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let (m0, o0, c0) = unsafe {{ step_nested(data.as_ptr().add(offset){carry_arg}) }};
            if let Some(err) = super::push_nested(data, m0, o0, c0, offset as u32, tape, stack) {{
                return Some(err);
            }}
            let (m1, o1, c1) = unsafe {{ step_nested(data.as_ptr().add(offset + 64){carry_arg}) }};
            if let Some(err) = super::push_nested(data, m1, o1, c1, (offset + 64) as u32, tape, stack) {{
                return Some(err);
            }}
            offset += 128;
        }}
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let (mask, opens, closes) = unsafe {{ step_nested(data.as_ptr().add(offset){carry_arg}) }};
            if let Some(err) = super::push_nested(data, mask, opens, closes, offset as u32, tape, stack) {{
                return Some(err);
            }}
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let (mask, opens, closes) = unsafe {{ step_nested(block.as_ptr(){carry_arg}) }};
            let live = (1u64 << rem) - 1;
            if let Some(err) =
                super::push_nested(data, mask & live, opens & live, closes & live, offset as u32, tape, stack)
            {{
                return Some(err);
            }}
        }}
        None
    }}
"#
        ) + &nested_par_kernels(
            &format!(
                "// SAFETY: the while guard keeps offset + 64 within data (interior\n                // bounds are 64-aligned and at most data.len()).\n                let (mask, opens, closes) = unsafe {{ step_nested(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n                let (mask, opens, closes) = unsafe {{ step_nested(block.as_ptr(){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: the while guard keeps offset + 64 within data (interior\n                // bounds are 64-aligned and at most data.len()).\n                let (mask, _) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"
            ),
            &format!(
                "// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n                let (mask, _) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"
            ),
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
        )
    } else {
        String::new()
    };

    let parser_doc = if dialect.is_some() {
        "//\n// Also exposes a span API: `parse(data)` -> `records()` -> `field(i)`,\n\
         // with dialect-aware quote stripping and escape resolution.\n"
    } else {
        ""
    };

    let mut code = String::new();

    let _ = write!(
        code,
        r#"// Generated by falx from format `{format_name}`. Do not edit by hand;
// regenerate with `cargo run --example generate` (in-tree) or the falx CLI.
//
// Structural indexer: appends to `out` the byte offset of every set bit in
// the format's output bitstream (its structural positions). Self-contained:
// depends only on std.
{parser_doc}
#[rustfmt::skip]
// The SIMD bodies are gated to `x86_64` (AVX-512/AVX2) and `aarch64` (NEON),
// so on any other architecture the kernel functions are dispatch stubs whose
// parameters and helpers go unused; allow the resulting (arch-conditional)
// lints there only — on the SIMD targets every lint stays active and catches
// real issues.
#[cfg_attr(not(any(target_arch = "x86_64", target_arch = "aarch64")), allow(unused_variables, dead_code, clippy::ptr_arg))]
mod generated {{
{carry_init_const}/// Index the structural positions of `data` into `out`.
pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {{
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        unsafe {{ avx512::index_structurals(data, out) }};
        return;
    }}
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        unsafe {{ avx2::index_structurals(data, out) }};
        return;
    }}
    unsupported_cpu();
}}

fn unsupported_cpu() -> ! {{
    panic!("falx generated kernels require x86_64 (AVX2+PCLMULQDQ or AVX-512F/BW/VL+PCLMULQDQ) or aarch64 (NEON+AES)");
}}

{quote_parity_dispatch}
"#
    );

    code.push_str(&par_block);

    if let Some(dialect) = dialect {
        push_span_api(&mut code, dialect, carry_count);
        if format_name == "logfmt" {
            push_logfmt_api(&mut code);
        }
        if format_name == "ndjson" {
            push_ndjson_lines_api(&mut code);
        }
        if format_name == "csv" || format_name == "tsv" {
            push_field_byte_stats_api(&mut code, dialect);
        }
        if !columns.is_empty() {
            push_columns_api(&mut code, dialect, columns, par_mode);
            if format_name == "csv_geo" {
                push_csv_geo_stats_api(&mut code, false);
            }
            if format_name == "csv_geo_text" {
                push_csv_geo_stats_api(&mut code, true);
            }
            if format_name == "vcf_typed" {
                push_vcf_stats_api(&mut code);
            }
        }
        if !dialect.nesting.is_empty() {
            push_nested_api(&mut code, dialect, carry_count);
        }
        if fastq_mode {
            push_fastq_api(&mut code);
        }
    }

    // NEON driver kernels. Every driver body operates only on `u64` masks via
    // `step`, so it is byte-identical to its AVX2 twin apart from the target-
    // feature attribute (see `neon_driver`). The exceptions — `neon_partial`
    // (folds in region_scan3's inlined loads) and `neon_fastq` (NEON checksum
    // accumulator) — are built explicitly above.
    let neon_seeded = neon_driver(&avx2_seeded);
    let neon_tape_seeded = neon_driver(&avx2_tape_seeded);
    let neon_quote_parity = neon_driver(&avx2_quote_parity);
    let neon_cells = neon_driver(&avx2_cells);
    let neon_vcf_stats = neon_driver(&avx2_vcf_stats);
    let neon_csv_geo_stats = neon_driver(&avx2_csv_geo_stats);
    let neon_field_byte_stats = neon_driver(&avx2_field_byte_stats);
    let neon_tape = neon_driver(&avx2_tape);
    let neon_ndjson_lines = neon_driver(&avx2_ndjson_lines);
    let neon_logfmt = neon_driver(&avx2_logfmt);
    let neon_nested = neon_driver(&avx2_nested);

    let _ = write!(
        code,
        r#"#[cfg(target_arch = "x86_64")]
mod avx512 {{
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "pclmulqdq")]
    pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {{
{carry_decl}        let mut offset = 0usize;
        // Two blocks per iteration: amortizes loop control and lets the
        // second block's classification overlap the first block's extract.
        while offset + 128 <= data.len() {{
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let m0 = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};
            push_indexes(m0, offset as u32, out);
            let m1 = unsafe {{ step(data.as_ptr().add(offset + 64){carry_arg}) }}{sel};
            push_indexes(m1, (offset + 64) as u32, out);
            offset += 128;
        }}
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let mask = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};
            push_indexes(mask, offset as u32, out);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let mask = unsafe {{ step(block.as_ptr(){carry_arg}) }}{sel} & ((1u64 << rem) - 1);
            push_indexes(mask, offset as u32, out);
        }}
    }}

{avx512_tape}{avx512_nested}{avx512_seeded}{avx512_tape_seeded}{avx512_quote_parity}{avx512_partial}{avx512_cells}{avx512_vcf_stats}{avx512_csv_geo_stats}{avx512_field_byte_stats}{avx512_fastq}{avx512_ndjson_lines}{avx512_logfmt}
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "pclmulqdq")]
    unsafe fn step(ptr: *const u8{carry_param}) -> {step_ret_ty} {{
        // SAFETY: caller guarantees 64 readable bytes at `ptr`.
        let (lo, hi) = unsafe {{
            (
                _mm256_loadu_si256(ptr as *const __m256i),
                _mm256_loadu_si256(ptr.add(32) as *const __m256i),
            )
        }};
"#
    );
    emit_step_body(&mut code, graph, &carry_slot, Flavor::Avx512, &step_roots);
    let _ = write!(code, "        {step_ret}\n    }}\n");
    if let Some((opens, closes)) = nest {
        let _ = write!(
            code,
            "\n    /// `step` twin for the fused nested driver.\n    \
             #[target_feature(enable = \"avx512f\", enable = \"avx512bw\", enable = \"avx512vl\", enable = \"pclmulqdq\")]\n    \
             unsafe fn step_nested(ptr: *const u8{carry_param}) -> (u64, u64, u64) {{\n        \
             // SAFETY: caller guarantees 64 readable bytes at `ptr`.\n        \
             let (lo, hi) = unsafe {{\n            (\n                \
             _mm256_loadu_si256(ptr as *const __m256i),\n                \
             _mm256_loadu_si256(ptr.add(32) as *const __m256i),\n            )\n        }};\n"
        );
        emit_step_body(&mut code, graph, &carry_slot, Flavor::Avx512, &nested_roots);
        let _ = write!(
            code,
            "        (v{}, v{}, v{})\n    }}\n",
            output.0, opens.0, closes.0
        );
    }

    if uses_regions {
        code.push_str(REGIONS_HELPER);
    }
    if uses_eq_class || uses_table_class {
        code.push_str(
            r#"
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
    fn eq_mask(lo: __m256i, hi: __m256i, byte: u8) -> u64 {
        let needle = _mm256_set1_epi8(byte as i8);
        let lo_bits = _mm256_cmpeq_epi8_mask(lo, needle) as u64;
        let hi_bits = _mm256_cmpeq_epi8_mask(hi, needle) as u64;
        lo_bits | (hi_bits << 32)
    }
"#,
        );
    }
    if uses_prefix_xor || uses_regions {
        code.push_str(
            r#"
    #[target_feature(enable = "pclmulqdq")]
    fn prefix_xor(mask: u64) -> u64 {
        let ones = _mm_set1_epi8(-1);
        let value = _mm_set_epi64x(0, mask as i64);
        let product = _mm_clmulepi64_si128::<0>(value, ones);
        _mm_cvtsi128_si64(product) as u64
    }
"#,
        );
    }
    code.push_str(
        r#"
    /// Branchless bit decoding (simdjson flatten_bits): write indexes in
    /// unconditional chunks of 8, then expose only the popcount-many real
    /// entries.
    #[inline]
    fn push_indexes(mut mask: u64, base: u32, out: &mut Vec<u32>) {
        let count = mask.count_ones() as usize;
        if count == 0 {
            return;
        }
        let len = out.len();
        out.reserve(count + 8);
        // SAFETY: capacity covers len + count + 8; chunked writes overshoot
        // by at most 7 entries and set_len exposes only the real ones.
        unsafe {
            let mut ptr = out.as_mut_ptr().add(len);
            let mut remaining = count as isize;
            while remaining > 0 {
                let mut j = 0;
                while j < 8 {
                    *ptr.add(j) = base + mask.trailing_zeros();
                    mask &= mask.wrapping_sub(1);
                    j += 1;
                }
                ptr = ptr.add(8);
                remaining -= 8;
            }
            out.set_len(len + count);
        }
    }
}

"#,
    );

    let _ = write!(
        code,
        r#"#[cfg(target_arch = "x86_64")]
mod avx2 {{
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2", enable = "pclmulqdq")]
    pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {{
{carry_decl}        let mut offset = 0usize;
        // Two blocks per iteration: amortizes loop control and lets the
        // second block's classification overlap the first block's extract.
        while offset + 128 <= data.len() {{
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let m0 = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};
            push_indexes(m0, offset as u32, out);
            let m1 = unsafe {{ step(data.as_ptr().add(offset + 64){carry_arg}) }}{sel};
            push_indexes(m1, (offset + 64) as u32, out);
            offset += 128;
        }}
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let mask = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};
            push_indexes(mask, offset as u32, out);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let mask = unsafe {{ step(block.as_ptr(){carry_arg}) }}{sel} & ((1u64 << rem) - 1);
            push_indexes(mask, offset as u32, out);
        }}
    }}

{avx2_tape}{avx2_nested}{avx2_seeded}{avx2_tape_seeded}{avx2_quote_parity}{avx2_partial}{avx2_cells}{avx2_vcf_stats}{avx2_csv_geo_stats}{avx2_field_byte_stats}{avx2_fastq}{avx2_ndjson_lines}{avx2_logfmt}
    #[target_feature(enable = "avx2", enable = "pclmulqdq")]
    unsafe fn step(ptr: *const u8{carry_param}) -> {step_ret_ty} {{
        // SAFETY: caller guarantees 64 readable bytes at `ptr`.
        let (lo, hi) = unsafe {{
            (
                _mm256_loadu_si256(ptr as *const __m256i),
                _mm256_loadu_si256(ptr.add(32) as *const __m256i),
            )
        }};
"#
    );
    emit_step_body(&mut code, graph, &carry_slot, Flavor::Avx2, &step_roots);
    let _ = write!(code, "        {step_ret}\n    }}\n");
    if let Some((opens, closes)) = nest {
        let _ = write!(
            code,
            "\n    /// `step` twin for the fused nested driver.\n    \
             #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n    \
             unsafe fn step_nested(ptr: *const u8{carry_param}) -> (u64, u64, u64) {{\n        \
             // SAFETY: caller guarantees 64 readable bytes at `ptr`.\n        \
             let (lo, hi) = unsafe {{\n            (\n                \
             _mm256_loadu_si256(ptr as *const __m256i),\n                \
             _mm256_loadu_si256(ptr.add(32) as *const __m256i),\n            )\n        }};\n"
        );
        emit_step_body(&mut code, graph, &carry_slot, Flavor::Avx2, &nested_roots);
        let _ = write!(
            code,
            "        (v{}, v{}, v{})\n    }}\n",
            output.0, opens.0, closes.0
        );
    }

    if uses_regions {
        code.push_str(REGIONS_HELPER);
    }
    if uses_eq_class {
        code.push_str(
            r#"
    #[target_feature(enable = "avx2")]
    fn eq_mask(lo: __m256i, hi: __m256i, byte: u8) -> u64 {
        let needle = _mm256_set1_epi8(byte as i8);
        let lo_bits = _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, needle)) as u32 as u64;
        let hi_bits = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, needle)) as u32 as u64;
        lo_bits | (hi_bits << 32)
    }
"#,
        );
    }
    if uses_table_class {
        code.push_str(
            r#"
    /// Shuffle-based classification (simdjson): byte b is a member iff
    /// lo_tbl[b & 15] & hi_tbl[b >> 4] != 0. Two PSHUFBs and an AND per
    /// 32-byte half, regardless of class size.
    #[target_feature(enable = "avx2")]
    fn table_mask(lo: __m256i, hi: __m256i, lo_tbl: __m256i, hi_tbl: __m256i) -> u64 {
        let nib = _mm256_set1_epi8(0x0F);
        let lo_lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(lo, nib));
        let lo_hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(lo), nib));
        let hi_lo = _mm256_shuffle_epi8(lo_tbl, _mm256_and_si256(hi, nib));
        let hi_hi = _mm256_shuffle_epi8(hi_tbl, _mm256_and_si256(_mm256_srli_epi16::<4>(hi), nib));
        let zero = _mm256_setzero_si256();
        let lo_z = _mm256_cmpeq_epi8(_mm256_and_si256(lo_lo, lo_hi), zero);
        let hi_z = _mm256_cmpeq_epi8(_mm256_and_si256(hi_lo, hi_hi), zero);
        let lo_bits = !(_mm256_movemask_epi8(lo_z) as u32) as u64;
        let hi_bits = !(_mm256_movemask_epi8(hi_z) as u32) as u64;
        lo_bits | (hi_bits << 32)
    }
"#,
        );
    }
    if uses_prefix_xor || uses_regions {
        code.push_str(
            r#"
    #[target_feature(enable = "pclmulqdq")]
    fn prefix_xor(mask: u64) -> u64 {
        let ones = _mm_set1_epi8(-1);
        let value = _mm_set_epi64x(0, mask as i64);
        let product = _mm_clmulepi64_si128::<0>(value, ones);
        _mm_cvtsi128_si64(product) as u64
    }
"#,
        );
    }
    code.push_str(
        r#"
    /// Branchless bit decoding (simdjson flatten_bits): write indexes in
    /// unconditional chunks of 8, then expose only the popcount-many real
    /// entries.
    #[inline]
    fn push_indexes(mut mask: u64, base: u32, out: &mut Vec<u32>) {
        let count = mask.count_ones() as usize;
        if count == 0 {
            return;
        }
        let len = out.len();
        out.reserve(count + 8);
        // SAFETY: capacity covers len + count + 8; chunked writes overshoot
        // by at most 7 entries and set_len exposes only the real ones.
        unsafe {
            let mut ptr = out.as_mut_ptr().add(len);
            let mut remaining = count as isize;
            while remaining > 0 {
                let mut j = 0;
                while j < 8 {
                    *ptr.add(j) = base + mask.trailing_zeros();
                    mask &= mask.wrapping_sub(1);
                    j += 1;
                }
                ptr = ptr.add(8);
                remaining -= 8;
            }
            out.set_len(len + count);
        }
    }
}
"#,
    );

    // ── ARM NEON (aarch64) ──────────────────────────────────────────────
    // A 64-byte block is four 128-bit vectors `b0..b3`; classification folds
    // them to the same `u64` mask as the x86 flavors (see `movemask`), so the
    // driver kernels and all mask-level logic are shared verbatim.
    let _ = write!(
        code,
        r#"#[cfg(target_arch = "aarch64")]
mod neon {{
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon", enable = "aes")]
    pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {{
{carry_decl}        let mut offset = 0usize;
        // Two blocks per iteration: amortizes loop control and lets the
        // second block's classification overlap the first block's extract.
        while offset + 128 <= data.len() {{
            // SAFETY: offset + 128 <= data.len(), so both blocks are readable.
            let m0 = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};
            push_indexes(m0, offset as u32, out);
            let m1 = unsafe {{ step(data.as_ptr().add(offset + 64){carry_arg}) }}{sel};
            push_indexes(m1, (offset + 64) as u32, out);
            offset += 128;
        }}
        while offset + 64 <= data.len() {{
            // SAFETY: offset + 64 <= data.len(), so 64 bytes are readable.
            let mask = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};
            push_indexes(mask, offset as u32, out);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            // SAFETY: block is a readable 64-byte buffer. Pad bits masked.
            let mask = unsafe {{ step(block.as_ptr(){carry_arg}) }}{sel} & ((1u64 << rem) - 1);
            push_indexes(mask, offset as u32, out);
        }}
    }}

{neon_tape}{neon_nested}{neon_seeded}{neon_tape_seeded}{neon_quote_parity}{neon_partial}{neon_cells}{neon_vcf_stats}{neon_csv_geo_stats}{neon_field_byte_stats}{neon_fastq}{neon_ndjson_lines}{neon_logfmt}
    #[target_feature(enable = "neon", enable = "aes")]
    unsafe fn step(ptr: *const u8{carry_param}) -> {step_ret_ty} {{
        // SAFETY: caller guarantees 64 readable bytes at `ptr`.
        let (b0, b1, b2, b3) = unsafe {{
            (
                vld1q_u8(ptr),
                vld1q_u8(ptr.add(16)),
                vld1q_u8(ptr.add(32)),
                vld1q_u8(ptr.add(48)),
            )
        }};
"#
    );
    emit_step_body(&mut code, graph, &carry_slot, Flavor::Neon, &step_roots);
    let _ = write!(code, "        {step_ret}\n    }}\n");
    if let Some((opens, closes)) = nest {
        let _ = write!(
            code,
            "\n    /// `step` twin for the fused nested driver.\n    \
             #[target_feature(enable = \"neon\", enable = \"aes\")]\n    \
             unsafe fn step_nested(ptr: *const u8{carry_param}) -> (u64, u64, u64) {{\n        \
             // SAFETY: caller guarantees 64 readable bytes at `ptr`.\n        \
             let (b0, b1, b2, b3) = unsafe {{\n            (\n                \
             vld1q_u8(ptr),\n                \
             vld1q_u8(ptr.add(16)),\n                \
             vld1q_u8(ptr.add(32)),\n                \
             vld1q_u8(ptr.add(48)),\n            )\n        }};\n"
        );
        emit_step_body(&mut code, graph, &carry_slot, Flavor::Neon, &nested_roots);
        let _ = write!(
            code,
            "        (v{}, v{}, v{})\n    }}\n",
            output.0, opens.0, closes.0
        );
    }

    if uses_regions {
        // `resolve_regions` calls `prefix_xor`, so it must carry the NEON
        // target features (PMULL = `aes`) in place of x86 `pclmulqdq`.
        code.push_str(&REGIONS_HELPER.replace("enable = \"pclmulqdq\"", NEON_ATTR_INNER));
    }
    if uses_eq_class || uses_table_class {
        code.push_str(
            r#"
    /// NEON has no movemask: AND each lane with its bit value, then fold the
    /// four 16-byte compare results to one dense `u64` (bit j ⇔ byte j) with
    /// three pairwise adds — bit-identical to x86 `movemask`/`cmpeq_epi8_mask`.
    #[target_feature(enable = "neon")]
    fn movemask(c0: uint8x16_t, c1: uint8x16_t, c2: uint8x16_t, c3: uint8x16_t) -> u64 {
        let bits = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
        let s0 = vpaddq_u8(vandq_u8(c0, bits), vandq_u8(c1, bits));
        let s1 = vpaddq_u8(vandq_u8(c2, bits), vandq_u8(c3, bits));
        let s2 = vpaddq_u8(s0, s1);
        let s3 = vpaddq_u8(s2, s2);
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(s3))
    }
"#,
        );
    }
    if uses_eq_class {
        code.push_str(
            r#"
    #[target_feature(enable = "neon")]
    fn eq_mask(b0: uint8x16_t, b1: uint8x16_t, b2: uint8x16_t, b3: uint8x16_t, byte: u8) -> u64 {
        let needle = vdupq_n_u8(byte);
        movemask(vceqq_u8(b0, needle), vceqq_u8(b1, needle), vceqq_u8(b2, needle), vceqq_u8(b3, needle))
    }
"#,
        );
    }
    if uses_table_class {
        code.push_str(
            r#"
    /// Shuffle-based classification (simdjson): byte b is a member iff
    /// lo_tbl[b & 15] & hi_tbl[b >> 4] != 0. One TBL per nibble per 16-byte
    /// vector, regardless of class size.
    #[target_feature(enable = "neon")]
    fn table_lane(v: uint8x16_t, lo_tbl: uint8x16_t, hi_tbl: uint8x16_t) -> uint8x16_t {
        let lo = vqtbl1q_u8(lo_tbl, vandq_u8(v, vdupq_n_u8(0x0F)));
        let hi = vqtbl1q_u8(hi_tbl, vshrq_n_u8::<4>(v));
        vtstq_u8(lo, hi)
    }

    #[target_feature(enable = "neon")]
    fn table_mask(b0: uint8x16_t, b1: uint8x16_t, b2: uint8x16_t, b3: uint8x16_t, lo_tbl: uint8x16_t, hi_tbl: uint8x16_t) -> u64 {
        movemask(table_lane(b0, lo_tbl, hi_tbl), table_lane(b1, lo_tbl, hi_tbl), table_lane(b2, lo_tbl, hi_tbl), table_lane(b3, lo_tbl, hi_tbl))
    }
"#,
        );
    }
    if uses_prefix_xor || uses_regions {
        code.push_str(
            r#"
    /// Carryless multiply by all-ones = inclusive prefix XOR. PMULL
    /// (`vmull_p64`) stands in for x86 PCLMULQDQ; the low 64 bits are the
    /// result.
    #[target_feature(enable = "neon", enable = "aes")]
    fn prefix_xor(mask: u64) -> u64 {
        vmull_p64(mask, !0u64) as u64
    }
"#,
        );
    }
    code.push_str(
        r#"
    /// Branchless bit decoding (simdjson flatten_bits): write indexes in
    /// unconditional chunks of 8, then expose only the popcount-many real
    /// entries.
    #[inline]
    fn push_indexes(mut mask: u64, base: u32, out: &mut Vec<u32>) {
        let count = mask.count_ones() as usize;
        if count == 0 {
            return;
        }
        let len = out.len();
        out.reserve(count + 8);
        // SAFETY: capacity covers len + count + 8; chunked writes overshoot
        // by at most 7 entries and set_len exposes only the real ones.
        unsafe {
            let mut ptr = out.as_mut_ptr().add(len);
            let mut remaining = count as isize;
            while remaining > 0 {
                let mut j = 0;
                while j < 8 {
                    *ptr.add(j) = base + mask.trailing_zeros();
                    mask &= mask.wrapping_sub(1);
                    j += 1;
                }
                ptr = ptr.add(8);
                remaining -= 8;
            }
            out.set_len(len + count);
        }
    }
}
"#,
    );

    code.push_str("\n}\n\npub use self::generated::*;\n");

    // Give every x86 runtime-dispatch block an aarch64/NEON sibling so the
    // generated kernels select a real implementation on ARM instead of
    // panicking in `unsupported_cpu`.
    let code = add_neon_dispatch(&code);

    Ok(code)
}

/// The quote-parity counting prepass shared by every parallel entry
/// point; the emitted code leaves `entry[t]` as the carry seed for chunk t.
/// Fragments that resolve each chunk's quote-parity seed *inside* the
/// indexing scope. Quote dialects otherwise need a separate prepass scope —
/// a whole second round of thread creation, which the breakdown showed costs
/// as much as the indexing itself. Folding it in behind one barrier spawns
/// the workers once. Returns (declarations before the scope, reference
/// captures in the map closure, per-thread seed computation in the spawn).
/// Parallel driver bodies (quote/plain dialects): index each chunk
/// speculatively as outside-quote, recover parity from the returned carry,
/// re-index the rare chunk that began mid-quote. Inserted verbatim into the
/// generated kernel; uses single braces (it is not re-processed by `format!`).
/// Parallel driver for comment+quote dialects (csv_hash). Region state
/// (NORMAL/QUOTE/COMMENT) crosses chunk boundaries and is not XOR-linear, so
/// the quote-parity reconstruction does not apply and a `\n` inside a quoted
/// field defeats line-ownership chunking. Instead: index every chunk
/// speculatively in the NORMAL state (its line-start carry is region
/// independent — the newline status of the preceding byte), then thread the
/// true region state across chunks serially and re-index the rare chunk that
/// truly began inside a quoted field or comment. Inserted verbatim (single
/// braces). parse_par only — index_structurals_par stays serial here.
const REGION_PARSE_PAR: &str = r#"/// One chunk's tape: separator positions and end entries.
type TapePart = (Vec<u32>, Vec<u64>);

/// Index one chunk with the given entry carries `[line_start, region_state]`,
/// returning the exit region state. Full 64-byte blocks go through
/// `index_tape_partial_dispatch`; a trailing partial block (only ever the final
/// chunk) is zero-padded exactly as the serial tail.
fn index_tape_seeded(slice: &[u8], mut carries: [u64; 2], base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) -> u64 {
    let full = slice.len() & !63;
    index_tape_partial_dispatch(&slice[..full], &mut carries, base, seps, ends);
    let rem = slice.len() - full;
    if rem > 0 {
        let mut block = [0u8; 64];
        block[..rem].copy_from_slice(&slice[full..]);
        let live = (1u64 << rem) - 1;
        index_tape_block_dispatch(&block, live, &mut carries, base + full as u32, seps, ends);
    }
    carries[1]
}

/// Runtime-dispatched no-tape region-state scan that advances all three
/// candidate region states in one pass: `carries[0]` is the line-start carry,
/// `states` is seeded `[0,1,2]` (the three entry states N/Q/C) and exits as the
/// chunk's transfer function `[f(N), f(Q), f(C)]`. The region inputs (q,s,n) are
/// computed once per block and shared across the three advances.
fn region_scan3_dispatch(data: &[u8], carries: &mut [u64; 2], states: &mut [u64; 3]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::region_scan3(data, carries, states) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::region_scan3(data, carries, states) };
        return;
    }
    unsupported_cpu();
}

/// Parallel [`parse`]: byte-identical tape to serial [`parse`]. The region
/// machine (NORMAL/QUOTE/COMMENT) carries state across chunks and its
/// transitions are not XOR-linear, so the entry state cannot be recovered with
/// a parity prefix as in the quote-only dialects. Instead each chunk's region
/// *transfer function* `f: state -> state` is computed in parallel (phase 1, a
/// no-tape scan for each of the three possible entry states); every chunk's
/// true entry state then follows from an O(threads) serial composition
/// (phase 2); and every chunk is indexed exactly once, in parallel, with its
/// known entry state (phase 3) — no serial re-indexing regardless of how many
/// boundaries land mid-quote/comment. The line-start carry is region
/// independent (the newline status of the preceding byte), read directly.
pub fn parse_par(data: &[u8], threads: usize) -> Parsed<'_> {
    match parse_par_parts(data, threads) {
        Some(parts) => {
            let (seps, ends) = scatter_tape(&parts);
            Parsed { data, seps, ends }
        }
        None => parse(data),
    }
}

/// Like [`parse_par`], reusing `recycle`'s tape allocations for the master tape
/// (the big one). At GiB/s the soft page faults of a fresh master tape are a
/// measurable share of a parse, so batch callers — one parse per request/file —
/// hand the previous parse back to skip them; the per-chunk tapes are still
/// fresh.
pub fn parse_par_into<'a>(data: &'a [u8], threads: usize, recycle: Parsed<'_>) -> Parsed<'a> {
    match parse_par_parts(data, threads) {
        Some(parts) => {
            let (seps, ends) = scatter_tape_into(&parts, recycle.seps, recycle.ends);
            Parsed { data, seps, ends }
        }
        None => parse_into(data, recycle),
    }
}

/// Build the per-chunk tape parts in parallel via the transfer-function scheme
/// (phases below); `None` means fall back to the serial path.
fn parse_par_parts(data: &[u8], threads: usize) -> Option<Vec<TapePart>> {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        return None;
    }
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| if t == threads { data.len() } else { (t * chunk).min(data.len()) })
        .collect();
    // Line-start carry entering a chunk: region-independent, = whether the byte
    // before it was a newline (CARRY_INIT at the file start).
    let line_start = |b: usize| -> u64 {
        if b == 0 { CARRY_INIT[0] } else { (data[b - 1] == 10u8) as u64 }
    };
    // Phase 1 (parallel): each chunk's region transfer function — the exit
    // region state for each of the three possible entry region states.
    let transfer: Vec<[u64; 3]> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let slice = &data[bounds[t]..bounds[t + 1]];
                let ls = line_start(bounds[t]);
                s.spawn(move || {
                    // One pass advances all three entry states (N/Q/C) at once.
                    let mut states = [0u64, 1, 2];
                    let mut c = [ls, 0];
                    region_scan3_dispatch(slice, &mut c, &mut states);
                    states
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("scan thread ok")).collect()
    });
    // Phase 2 (serial, O(threads)): compose transfer functions to get every
    // chunk's true entry region state (file start = NORMAL = CARRY_INIT[1]).
    let mut entry = vec![0u64; threads];
    let mut region = 0u64;
    for t in 0..threads {
        entry[t] = region;
        region = transfer[t][region as usize];
    }
    // Phase 3 (parallel): index every chunk exactly once with its true carries.
    let parts: Vec<TapePart> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let slice = &data[bounds[t]..bounds[t + 1]];
                let base = bounds[t] as u32;
                let carries = [line_start(bounds[t]), entry[t]];
                s.spawn(move || {
                    let mut seps = Vec::with_capacity(slice.len() / 16 + 8);
                    let mut ends = Vec::with_capacity(slice.len() / 32 + 8);
                    let _ = index_tape_seeded(slice, carries, base, &mut seps, &mut ends);
                    (seps, ends)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("index thread ok")).collect()
    });
    Some(parts)
}

/// Concatenate per-chunk tape parts into the master tape, copying each into its
/// own disjoint slot concurrently. Separator positions and end byte offsets are
/// already absolute; only each end entry's cumulative-separator high word is
/// rebased by the separators in the preceding chunks.
fn scatter_tape(parts: &[TapePart]) -> (Vec<u32>, Vec<u64>) {
    scatter_tape_into(parts, Vec::new(), Vec::new())
}

/// [`scatter_tape`] reusing the provided buffers' allocations for the master
/// tape (cleared first). `reserve` is a no-op when the recycled capacity already
/// covers the totals, so the steady-state batch caller takes no page faults.
fn scatter_tape_into(parts: &[TapePart], mut seps: Vec<u32>, mut ends: Vec<u64>) -> (Vec<u32>, Vec<u64>) {
    let sep_total: usize = parts.iter().map(|p| p.0.len()).sum();
    let end_total: usize = parts.iter().map(|p| p.1.len()).sum();
    seps.clear();
    ends.clear();
    seps.reserve(sep_total);
    ends.reserve(end_total);
    let mut sep_prefix: Vec<u64> = Vec::with_capacity(parts.len());
    {
        let mut acc = 0u64;
        for p in parts {
            sep_prefix.push(acc);
            acc += p.0.len() as u64;
        }
    }
    {
        let mut srest = &mut seps.spare_capacity_mut()[..sep_total];
        let mut erest = &mut ends.spare_capacity_mut()[..end_total];
        let mut sslots: Vec<&mut [std::mem::MaybeUninit<u32>]> = Vec::with_capacity(parts.len());
        let mut eslots: Vec<&mut [std::mem::MaybeUninit<u64>]> = Vec::with_capacity(parts.len());
        for p in parts {
            let (sh, st) = srest.split_at_mut(p.0.len());
            sslots.push(sh);
            srest = st;
            let (eh, et) = erest.split_at_mut(p.1.len());
            eslots.push(eh);
            erest = et;
        }
        std::thread::scope(|s| {
            for (((sslot, eslot), part), &prefix) in
                sslots.into_iter().zip(eslots).zip(parts.iter()).zip(sep_prefix.iter())
            {
                s.spawn(move || {
                    // SAFETY: each slot's length equals its part's; the writes
                    // initialize exactly these disjoint slices.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            part.0.as_ptr(),
                            sslot.as_mut_ptr().cast::<u32>(),
                            part.0.len(),
                        );
                    }
                    let rebase = prefix << 32;
                    let dst = eslot.as_mut_ptr().cast::<u64>();
                    for (i, &e) in part.1.iter().enumerate() {
                        // SAFETY: i < part.1.len() == eslot.len().
                        unsafe { *dst.add(i) = e + rebase; }
                    }
                });
            }
        });
    }
    // SAFETY: the scatter initialized every element of both spare regions.
    unsafe {
        seps.set_len(sep_total);
        ends.set_len(end_total);
    }
    (seps, ends)
}
"#;

const SPEC_INDEX_PAR: &str = r#"/// Parallel structural indexing: byte-identical to [`index_structurals`],
/// split across `threads` chunks. Each chunk is indexed speculatively as if
/// it started outside a quoted region; its quote parity falls out of the
/// kernel's final carry, so there is no separate counting prepass over the
/// data. The rare chunk that truly began inside a quoted field is re-indexed.
pub fn index_structurals_par(data: &[u8], threads: usize, out: &mut Vec<u32>) {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        index_structurals(data, out);
        return;
    }
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| if t == threads { data.len() } else { (t * chunk).min(data.len()) })
        .collect();
    let mut results: Vec<(Vec<u32>, u64)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let slice = &data[bounds[t]..bounds[t + 1]];
                let base = bounds[t] as u32;
                s.spawn(move || {
                    let mut part = Vec::with_capacity(slice.len() / 16 + 8);
                    let carry = index_structurals_seeded_dispatch(slice, 0, base, &mut part);
                    (part, carry)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("index thread ok")).collect()
    });
    let mut entry = 0u64;
    let mut redo: Vec<usize> = Vec::new();
    for (t, r) in results.iter().enumerate() {
        if entry != 0 {
            redo.push(t);
        }
        entry ^= r.1;
    }
    if !redo.is_empty() {
        let redone: Vec<(usize, Vec<u32>)> = std::thread::scope(|s| {
            let handles: Vec<_> = redo
                .iter()
                .map(|&t| {
                    let slice = &data[bounds[t]..bounds[t + 1]];
                    let base = bounds[t] as u32;
                    s.spawn(move || {
                        let mut part = Vec::with_capacity(slice.len() / 16 + 8);
                        index_structurals_seeded_dispatch(slice, u64::MAX, base, &mut part);
                        (t, part)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("redo thread ok")).collect()
        });
        for (t, part) in redone {
            results[t].0 = part;
        }
    }
    let parts: Vec<Vec<u32>> = results.into_iter().map(|r| r.0).collect();
    scatter_u32(out, &parts);
}"#;

const SPEC_PARSE_PAR: &str = r#"/// Parallel [`parse`]: identical tape, built across `threads` chunks.
pub fn parse_par(data: &[u8], threads: usize) -> Parsed<'_> {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        return parse(data);
    }
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| if t == threads { data.len() } else { (t * chunk).min(data.len()) })
        .collect();
    let mut results: Vec<(TapePart, u64)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let slice = &data[bounds[t]..bounds[t + 1]];
                let base = bounds[t] as u32;
                s.spawn(move || {
                    let mut seps = Vec::with_capacity(slice.len() / 16 + 8);
                    let mut ends = Vec::with_capacity(slice.len() / 32 + 8);
                    let carry = index_tape_seeded_dispatch(slice, 0, base, &mut seps, &mut ends);
                    ((seps, ends), carry)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("parse thread ok")).collect()
    });
    let mut entry = 0u64;
    let mut redo: Vec<usize> = Vec::new();
    for (t, r) in results.iter().enumerate() {
        if entry != 0 {
            redo.push(t);
        }
        entry ^= r.1;
    }
    if !redo.is_empty() {
        let redone: Vec<(usize, TapePart)> = std::thread::scope(|s| {
            let handles: Vec<_> = redo
                .iter()
                .map(|&t| {
                    let slice = &data[bounds[t]..bounds[t + 1]];
                    let base = bounds[t] as u32;
                    s.spawn(move || {
                        let mut seps = Vec::with_capacity(slice.len() / 16 + 8);
                        let mut ends = Vec::with_capacity(slice.len() / 32 + 8);
                        index_tape_seeded_dispatch(slice, u64::MAX, base, &mut seps, &mut ends);
                        (t, (seps, ends))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("redo thread ok")).collect()
        });
        for (t, part) in redone {
            results[t].0 = part;
        }
    }
    let parts: Vec<TapePart> = results.into_iter().map(|r| r.0).collect();
    let (seps, ends) = scatter_tape(&parts);
    Parsed { data, seps, ends }
}"#;

/// Parallel driver bodies (comment-without-quote dialects). Comments end at
/// every newline and there are no newline-spanning quoted regions, so any
/// line boundary is a clean NORMAL region state: each worker just starts on a
/// fresh line and indexes whole lines with the standard entry carry — no
/// region state crosses a chunk, no seed propagation, no re-indexing.
const COMMENT_INDEX_PAR: &str = r#"/// Chunk bounds snapped to just past a record terminator (newline) so every
/// worker processes only whole lines: the byte before `bounds[t]` (t>0) is a
/// terminator, so the worker begins in the NORMAL region state.
fn line_aligned_bounds(data: &[u8], threads: usize) -> Vec<usize> {
    let approx = data.len() / threads;
    let mut bounds = vec![0usize; threads + 1];
    bounds[threads] = data.len();
    for (i, slot) in bounds[1..threads].iter_mut().enumerate() {
        let from = ((i + 1) * approx).min(data.len());
        *slot = match data[from..].iter().position(|&b| b == 10u8) {
            Some(p) => from + p + 1,
            None => data.len(),
        };
    }
    bounds
}

/// Parallel structural indexing: byte-identical to [`index_structurals`],
/// split across `threads` line-aligned chunks (comment state never crosses a
/// chunk boundary, so no carry is propagated).
pub fn index_structurals_par(data: &[u8], threads: usize, out: &mut Vec<u32>) {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    if threads == 1 {
        index_structurals(data, out);
        return;
    }
    let bounds = line_aligned_bounds(data, threads);
    let parts: Vec<Vec<u32>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let slice = &data[bounds[t]..bounds[t + 1]];
                let base = bounds[t] as u32;
                s.spawn(move || {
                    let mut part = Vec::with_capacity(slice.len() / 16 + 8);
                    let _ = index_structurals_seeded_dispatch(slice, 0, base, &mut part);
                    part
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("index thread ok")).collect()
    });
    scatter_u32(out, &parts);
}"#;

const COMMENT_PARSE_PAR: &str = r#"/// Parallel [`parse`] for a comment dialect; line-aligned chunking (see
/// [`index_structurals_par`]). Chunk tapes concatenate; each end entry's
/// cumulative separator count is rebased during the merge.
pub fn parse_par(data: &[u8], threads: usize) -> Parsed<'_> {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    if threads == 1 {
        return parse(data);
    }
    let bounds = line_aligned_bounds(data, threads);
    let parts: Vec<TapePart> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let slice = &data[bounds[t]..bounds[t + 1]];
                let base = bounds[t] as u32;
                s.spawn(move || {
                    let mut seps = Vec::with_capacity(slice.len() / 16 + 8);
                    let mut ends = Vec::with_capacity(slice.len() / 32 + 8);
                    let _ = index_tape_seeded_dispatch(slice, 0, base, &mut seps, &mut ends);
                    (seps, ends)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("parse thread ok")).collect()
    });
    let (seps, ends) = scatter_tape(&parts);
    Parsed { data, seps, ends }
}"#;

fn par_seed_parts(quote: Option<u8>, chunk: &str) -> (String, String, String) {
    match quote {
        Some(_) => (
            "    // Quote seeds are resolved inside the indexing scope: each thread\n\
             \x20   // contributes its chunk's quote-byte parity, then reads the parity\n\
             \x20   // of all preceding chunks after one barrier (doubled-quote escapes\n\
             \x20   // self-cancel, so raw parity is exact).\n\
             \x20   let parity: Vec<std::sync::atomic::AtomicU64> =\n\
             \x20       (0..threads).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();\n\
             \x20   let barrier = std::sync::Barrier::new(threads);\n"
                .to_string(),
            "                let parity = &parity;\n\
             \x20               let barrier = &barrier;\n"
                .to_string(),
            format!(
                "                    use std::sync::atomic::Ordering::Relaxed;\n\
             \x20                   parity[t].store(quote_parity_dispatch(&{chunk}, 0) & 1, Relaxed);\n\
             \x20                   barrier.wait();\n\
             \x20                   // Seed = quote parity of every preceding chunk.\n\
             \x20                   let seed = parity[..t]\n\
             \x20                       .iter()\n\
             \x20                       .fold(0u64, |acc, p| acc ^ p.load(Relaxed))\n\
             \x20                       .wrapping_neg();\n"
            ),
        ),
        None => (
            String::new(),
            String::new(),
            "                    let seed = 0u64;\n".to_string(),
        ),
    }
}

/// Emit the records/fields span API: lazy walking of the structural index
/// into record and field spans, with dialect-specific field cleaning.
fn push_span_api(code: &mut String, dialect: &crate::formats::Dialect, carry_count: usize) {
    // Fixed-line-count records (FASTQ-style): group N newline-terminated lines
    // per record. A distinct, self-contained record API; the streaming parser
    // (which delivers one record per line) is not emitted in this mode.
    if dialect.lines_per_record > 1 {
        push_grouped_span_api(code, dialect.lines_per_record);
        code.push_str(&clean_fn(dialect));
        code.push('\n');
        return;
    }
    // Comment dialects keep comment lines on the tape (their newline is a
    // record boundary) and skip them lazily during iteration.
    let next_body = match dialect.comment {
        Some(c) => format!(
            "        // Comment records stay on the tape as boundaries;\n\
             \x20       // iteration skips them lazily here.\n\
             \x20       loop {{\n\
             \x20           let record = self.next_raw()?;\n\
             \x20           if record.end > record.start && self.data[record.start] == {c}u8 {{\n\
             \x20               continue;\n\
             \x20           }}\n\
             \x20           return Some(record);\n\
             \x20       }}"
        ),
        None => "        self.next_raw()".to_string(),
    };
    let span_tpl = r#"/// Record-aware tape indexing used by [`parse`].
fn index_tape(data: &[u8], seps: &mut Vec<u32>, ends: &mut Vec<u64>) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::index_tape(data, seps, ends) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_tape(data, seps, ends) };
        return;
    }
    unsupported_cpu();
}

/// Index `data` and return a lazy record/field view over it.
pub fn parse(data: &[u8]) -> Parsed<'_> {
    let mut seps = Vec::with_capacity(data.len() / 16 + 8);
    let mut ends = Vec::with_capacity(data.len() / 32 + 8);
    index_tape(data, &mut seps, &mut ends);
    Parsed { data, seps, ends }
}

/// Like [`parse`], recycling the tape allocations of a previous parse (its
/// contents are discarded). At GiB/s the soft page faults of fresh tape
/// buffers are a measurable share of a parse; steady-state callers — one
/// parse per batch, file, or request — avoid them entirely.
pub fn parse_into<'a>(data: &'a [u8], recycle: Parsed<'_>) -> Parsed<'a> {
    let mut seps = recycle.seps;
    let mut ends = recycle.ends;
    seps.clear();
    ends.clear();
    index_tape(data, &mut seps, &mut ends);
    Parsed { data, seps, ends }
}

/// A structural tape over borrowed input: separator positions plus record
/// ends carrying cumulative separator counts, so record iteration is O(1)
/// per record and never rescans the input.
pub struct Parsed<'a> {
    data: &'a [u8],
    seps: Vec<u32>,
    ends: Vec<u64>,
}

impl<'a> Parsed<'a> {
    /// Iterate records (lines outside quoted regions). A record's trailing
    /// `\r` is trimmed; an empty line yields a record with one empty field.
    pub fn records(&self) -> Records<'_> {
        self.records_range(0..self.ends.len())
    }

    /// Number of newline-terminated records (a trailing unterminated record
    /// is not counted but is still yielded by iteration).
    pub fn terminated_record_count(&self) -> usize {
        self.ends.len()
    }

    /// Iterate a sub-range of terminated records. Disjoint ranges cover
    /// disjoint input and can be walked from different threads — the
    /// building block for parallel record processing. The range ending at
    /// `terminated_record_count()` also yields any trailing unterminated
    /// record.
    pub fn records_range(&self, range: std::ops::Range<usize>) -> Records<'_> {
        let (byte_pos, sep_pos) = if range.start == 0 {
            (0, 0)
        } else {
            let prev = self.ends[range.start - 1];
            ((prev & 0xFFFF_FFFF) as usize + 1, (prev >> 32) as usize)
        };
        let data_end = if range.start >= range.end && !self.ends.is_empty() {
            byte_pos // empty sub-range: fence immediately, yield nothing
        } else if range.end == self.ends.len() {
            self.data.len()
        } else {
            // Coverage fence: the byte after this chunk's last terminator.
            (self.ends[range.end - 1] & 0xFFFF_FFFF) as usize + 1
        };
        Records {
            data: self.data,
            seps: &self.seps,
            ends: &self.ends[..range.end],
            next_end: range.start,
            byte_pos,
            sep_pos,
            data_end,
        }
    }
}

pub struct Records<'p> {
    data: &'p [u8],
    seps: &'p [u32],
    ends: &'p [u64],
    next_end: usize,
    byte_pos: usize,
    sep_pos: usize,
    /// Trailing-record fence: iteration past the last tape entry stops here.
    data_end: usize,
}

impl<'p> Records<'p> {
    /// Produce the next record in tape order, comment lines included.
    fn next_raw(&mut self) -> Option<Record<'p>> {
        let start = self.byte_pos;
        let (end, seps) = if self.next_end < self.ends.len() {
            let entry = self.ends[self.next_end];
            self.next_end += 1;
            let end = (entry & 0xFFFF_FFFF) as usize;
            let cum = (entry >> 32) as usize;
            let seps = &self.seps[self.sep_pos..cum];
            self.sep_pos = cum;
            self.byte_pos = end + 1;
            (end, seps)
        } else {
            // Trailing record without a newline (only at the true data end).
            if start >= self.data_end {
                return None;
            }
            let seps = &self.seps[self.sep_pos..];
            self.sep_pos = self.seps.len();
            self.byte_pos = self.data_end;
            (self.data_end, seps)
        };
        Some(Record { data: self.data, start, end, seps })
    }
}

impl<'p> Iterator for Records<'p> {
    type Item = Record<'p>;

    fn next(&mut self) -> Option<Record<'p>> {
@NEXT_BODY@
    }
}

/// One record: a span of input plus the separator positions inside it.
/// The `\r` of a `\r\n` terminator is trimmed lazily, where bytes are
/// actually read, so tape-only walks never touch the input buffer.
#[derive(Clone, Copy)]
pub struct Record<'p> {
    data: &'p [u8],
    start: usize,
    end: usize,
    seps: &'p [u32],
}

impl<'p> Record<'p> {
    /// Record end with a trailing `\r` (of `\r\n`) trimmed.
    #[inline]
    fn trimmed_end(&self) -> usize {
        if self.end > self.start && self.data[self.end - 1] == b'\r' {
            self.end - 1
        } else {
            self.end
        }
    }

    /// The whole record span, terminator excluded.
    pub fn as_bytes(&self) -> &'p [u8] {
        &self.data[self.start..self.trimmed_end()]
    }

    pub fn field_count(&self) -> usize {
        self.seps.len() + 1
    }

    /// Byte-offset span `(from, to)` of field `i`, quotes and escapes
    /// intact; offsets index the buffer the tape was built over.
    pub fn field_span(&self, i: usize) -> Option<(u32, u32)> {
        if i > self.seps.len() {
            return None;
        }
        let from = if i == 0 {
            self.start
        } else {
            self.seps[i - 1] as usize + 1
        };
        let to = if i == self.seps.len() {
            self.trimmed_end()
        } else {
            self.seps[i] as usize
        };
        Some((from as u32, to as u32))
    }

    /// Raw span of field `i`: quotes and escapes intact.
    pub fn field_raw(&self, i: usize) -> Option<&'p [u8]> {
        self.field_span(i)
            .map(|(from, to)| &self.data[from as usize..to as usize])
    }

    /// Field `i` with surrounding quotes stripped and escapes resolved;
    /// borrows unless an escape sequence forces a copy.
    pub fn field(&self, i: usize) -> Option<std::borrow::Cow<'p, [u8]>> {
        self.field_raw(i).map(clean)
    }

    pub fn fields(&self) -> Fields<'p> {
        Fields {
            data: self.data,
            seps: self.seps,
            next_sep: 0,
            from: self.start,
            end: self.trimmed_end(),
            done: false,
        }
    }

    /// Zero-copy field iterator: raw `&[u8]` spans with quotes and escapes
    /// intact — no `Cow`, no cleaning. The fastest field walk, for callers
    /// that handle (or don't need) quote stripping themselves.
    pub fn fields_raw(&self) -> FieldsRaw<'p> {
        FieldsRaw {
            data: self.data,
            seps: self.seps,
            next_sep: 0,
            from: self.start,
            end: self.trimmed_end(),
            done: false,
        }
    }
}

/// Field iterator: one running offset, one separator-slice cursor — no
/// per-field bounds re-derivation.
pub struct Fields<'p> {
    data: &'p [u8],
    seps: &'p [u32],
    next_sep: usize,
    from: usize,
    end: usize,
    done: bool,
}

impl<'p> Iterator for Fields<'p> {
    type Item = std::borrow::Cow<'p, [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.next_sep < self.seps.len() {
            // SAFETY: next_sep was just bounds-checked, and tape invariants
            // make the derived span valid: separator positions are strictly
            // increasing offsets into `data`, and `from` is either the
            // record start or one past the previous separator, so
            // from <= to < data.len() always holds. Fields walking is the
            // per-field hot path; the redundant checks measurably dominate
            // it for short fields.
            let span = unsafe {
                let to = *self.seps.get_unchecked(self.next_sep) as usize;
                self.next_sep += 1;
                let span = self.data.get_unchecked(self.from..to);
                self.from = to + 1;
                span
            };
            Some(clean(span))
        } else if !self.done {
            self.done = true;
            Some(clean(&self.data[self.from..self.end]))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.seps.len() - self.next_sep + (!self.done) as usize;
        (n, Some(n))
    }
}

/// Zero-copy variant of [`Fields`]: yields raw `&[u8]` spans, quotes and
/// escapes intact, with no `Cow` and no cleaning.
pub struct FieldsRaw<'p> {
    data: &'p [u8],
    seps: &'p [u32],
    next_sep: usize,
    from: usize,
    end: usize,
    done: bool,
}

impl<'p> Iterator for FieldsRaw<'p> {
    type Item = &'p [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.next_sep < self.seps.len() {
            // SAFETY: identical tape invariants to `Fields::next`.
            let span = unsafe {
                let to = *self.seps.get_unchecked(self.next_sep) as usize;
                self.next_sep += 1;
                let span = self.data.get_unchecked(self.from..to);
                self.from = to + 1;
                span
            };
            Some(span)
        } else if !self.done {
            self.done = true;
            Some(&self.data[self.from..self.end])
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.seps.len() - self.next_sep + (!self.done) as usize;
        (n, Some(n))
    }
}

"#;
    code.push_str(&span_tpl.replace("@NEXT_BODY@", &next_body));
    let mut stream = STREAM_TPL.replace("@K@", &carry_count.to_string()).replace(
        "@CINIT@",
        if carry_count > 0 {
            "CARRY_INIT"
        } else {
            "[0u64; 0]"
        },
    );
    if let Some(c) = dialect.comment {
        // emit_ready: comment records advance the cursors without being
        // delivered. The terminator byte at `end` is always in-bounds, so
        // indexing buf[record_start] is too.
        stream = stream.replace(
            "            on_record(Record {
                data: &self.buf,
                start: self.record_start,
                end,
                seps: &self.seps[self.emitted_seps..cum],
            });",
            &format!(
                "            // Comment lines remain boundaries but are not records.
            if end == self.record_start || self.buf[self.record_start] != {c}u8 {{
                on_record(Record {{
                    data: &self.buf,
                    start: self.record_start,
                    end,
                    seps: &self.seps[self.emitted_seps..cum],
                }});
            }}"
            ),
        );
        stream = stream.replace(
            "        if self.record_start < self.buf.len() {",
            &format!(
                "        if self.record_start < self.buf.len() && self.buf[self.record_start] != {c}u8 {{"
            ),
        );
    }
    code.push_str(&stream);
    code.push_str(&clean_fn(dialect));
    code.push('\n');
}

/// Record API for a fixed-line-count line format (`lines_per_record = N > 1`):
/// `records()` yields one record per N newline-terminated lines, exposing the
/// N constituent lines as its fields. Self-contained; `@N@` is the line count.
fn push_grouped_span_api(code: &mut String, n: u32) {
    let tpl = r#"/// Record-aware tape indexing used by [`parse`].
fn index_tape(data: &[u8], seps: &mut Vec<u32>, ends: &mut Vec<u64>) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::index_tape(data, seps, ends) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_tape(data, seps, ends) };
        return;
    }
    unsupported_cpu();
}

/// Index `data` into a lazy record view: one record per @N@ newline-terminated
/// lines (a trailing partial group of fewer than @N@ lines is dropped).
pub fn parse(data: &[u8]) -> Parsed<'_> {
    let mut seps = Vec::with_capacity(data.len() / 16 + 8);
    let mut ends = Vec::with_capacity(data.len() / 32 + 8);
    index_tape(data, &mut seps, &mut ends);
    Parsed { data, seps, ends }
}

/// Like [`parse`], recycling the tape allocations of a previous parse.
pub fn parse_into<'a>(data: &'a [u8], recycle: Parsed<'_>) -> Parsed<'a> {
    let mut seps = recycle.seps;
    let mut ends = recycle.ends;
    seps.clear();
    ends.clear();
    index_tape(data, &mut seps, &mut ends);
    Parsed { data, seps, ends }
}

/// A structural tape over borrowed input. Records group @N@ newline-terminated
/// lines; `seps` is retained for the parallel tape merge (`parse_par`).
pub struct Parsed<'a> {
    data: &'a [u8],
    seps: Vec<u32>,
    ends: Vec<u64>,
}

impl<'a> Parsed<'a> {
    /// Iterate records, each grouping @N@ newline-terminated lines.
    pub fn records(&self) -> Records<'_> {
        self.records_range(0..self.terminated_record_count())
    }

    /// Number of complete @N@-line records (a trailing partial group is not
    /// counted).
    pub fn terminated_record_count(&self) -> usize {
        self.ends.len() / @N@
    }

    /// Iterate a sub-range of records by record index. Disjoint ranges cover
    /// disjoint input and can be walked from different threads.
    pub fn records_range(&self, range: std::ops::Range<usize>) -> Records<'_> {
        let groups = self.terminated_record_count();
        let end = range.end.min(groups);
        let start = range.start.min(end);
        Records {
            data: self.data,
            ends: &self.ends[..end * @N@],
            group: start,
        }
    }
}

pub struct Records<'p> {
    data: &'p [u8],
    ends: &'p [u64],
    group: usize,
}

impl<'p> Records<'p> {
    #[inline]
    fn next_raw(&mut self) -> Option<Record<'p>> {
        let base = self.group * @N@;
        if base + @N@ > self.ends.len() {
            return None;
        }
        // Record start: just past the previous group's last terminator.
        let start = if base == 0 {
            0
        } else {
            (self.ends[base - 1] & 0xFFFF_FFFF) as usize + 1
        };
        let lines = &self.ends[base..base + @N@];
        self.group += 1;
        Some(Record { data: self.data, start, lines })
    }
}

impl<'p> Iterator for Records<'p> {
    type Item = Record<'p>;

    #[inline]
    fn next(&mut self) -> Option<Record<'p>> {
        self.next_raw()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.ends.len() / @N@ - self.group;
        (n, Some(n))
    }
}

/// One record: @N@ consecutive lines. Field `i` is line `i`, its trailing
/// newline excluded (a `\r` of `\r\n` trimmed). Trimming and byte access are
/// lazy, so tape-only walks never touch the input buffer.
#[derive(Clone, Copy)]
pub struct Record<'p> {
    data: &'p [u8],
    start: usize,
    lines: &'p [u64],
}

impl<'p> Record<'p> {
    #[inline]
    fn line_bounds(&self, i: usize) -> (usize, usize) {
        let from = if i == 0 {
            self.start
        } else {
            (self.lines[i - 1] & 0xFFFF_FFFF) as usize + 1
        };
        let mut to = (self.lines[i] & 0xFFFF_FFFF) as usize;
        if to > from && self.data[to - 1] == b'\r' {
            to -= 1;
        }
        (from, to)
    }

    /// The whole record span (all @N@ lines, final terminator excluded).
    pub fn as_bytes(&self) -> &'p [u8] {
        let to = (self.lines[self.lines.len() - 1] & 0xFFFF_FFFF) as usize;
        let to = if to > self.start && self.data[to - 1] == b'\r' {
            to - 1
        } else {
            to
        };
        &self.data[self.start..to]
    }

    /// Number of fields: always @N@ (the record's lines).
    pub fn field_count(&self) -> usize {
        self.lines.len()
    }

    /// Byte-offset span `(from, to)` of line `i`.
    pub fn field_span(&self, i: usize) -> Option<(u32, u32)> {
        if i >= self.lines.len() {
            return None;
        }
        let (from, to) = self.line_bounds(i);
        Some((from as u32, to as u32))
    }

    /// Raw bytes of line `i`.
    pub fn field_raw(&self, i: usize) -> Option<&'p [u8]> {
        if i >= self.lines.len() {
            return None;
        }
        let (from, to) = self.line_bounds(i);
        Some(&self.data[from..to])
    }

    /// Line `i`. A line format has no quoting, so this borrows the raw bytes.
    pub fn field(&self, i: usize) -> Option<std::borrow::Cow<'p, [u8]>> {
        self.field_raw(i).map(clean)
    }

    /// Iterate the record's @N@ lines.
    pub fn fields(&self) -> Fields<'p> {
        Fields { data: self.data, start: self.start, lines: self.lines, i: 0 }
    }

    /// Zero-copy line iterator: raw `&[u8]` spans.
    pub fn fields_raw(&self) -> FieldsRaw<'p> {
        FieldsRaw { data: self.data, start: self.start, lines: self.lines, i: 0 }
    }
}

/// Cleaned-line iterator over a record's @N@ lines.
pub struct Fields<'p> {
    data: &'p [u8],
    start: usize,
    lines: &'p [u64],
    i: usize,
}

impl<'p> Iterator for Fields<'p> {
    type Item = std::borrow::Cow<'p, [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.lines.len() {
            return None;
        }
        let from = if self.i == 0 {
            self.start
        } else {
            (self.lines[self.i - 1] & 0xFFFF_FFFF) as usize + 1
        };
        let mut to = (self.lines[self.i] & 0xFFFF_FFFF) as usize;
        if to > from && self.data[to - 1] == b'\r' {
            to -= 1;
        }
        self.i += 1;
        Some(clean(&self.data[from..to]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.lines.len() - self.i;
        (n, Some(n))
    }
}

/// Zero-copy variant of [`Fields`]: raw `&[u8]` line spans.
pub struct FieldsRaw<'p> {
    data: &'p [u8],
    start: usize,
    lines: &'p [u64],
    i: usize,
}

impl<'p> Iterator for FieldsRaw<'p> {
    type Item = &'p [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.lines.len() {
            return None;
        }
        let from = if self.i == 0 {
            self.start
        } else {
            (self.lines[self.i - 1] & 0xFFFF_FFFF) as usize + 1
        };
        let mut to = (self.lines[self.i] & 0xFFFF_FFFF) as usize;
        if to > from && self.data[to - 1] == b'\r' {
            to -= 1;
        }
        self.i += 1;
        Some(&self.data[from..to])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.lines.len() - self.i;
        (n, Some(n))
    }
}
"#;
    code.push_str(&tpl.replace("@N@", &n.to_string()));
}

/// Emit the typed columnar projection API: a `Columns` struct with one
/// values Vec plus an Arrow-style validity bitmap per declared column,
/// filled by walking the structural tape so unrequested fields are never
/// read, cleaned, or copied.
fn push_fastq_api(code: &mut String) {
    code.push_str(
        r#"
/// FASTQ parser summary emitted by [`parse_fastq`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FastqStats {
    pub records: u64,
    pub sequence_bytes: u64,
    pub quality_bytes: u64,
    pub checksum: u64,
}

/// FASTQ validation error from the generated fixed-four-line parser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastqError {
    IncompleteRecord,
    BadHeader {
        offset: usize,
    },
    BadSeparator {
        offset: usize,
    },
    LengthMismatch {
        record: u64,
        sequence: usize,
        quality: usize,
    },
    TrailingBytes {
        offset: usize,
    },
}

/// Parse FASTQ records and return validated summary counters.
///
/// This is a generated domain API over the generated newline kernel: the SIMD
/// step emits newline masks, and a fixed-four-line sink validates FASTQ shape
/// while accumulating counters.
///
/// Strictness: a record is exactly four `\n`-terminated lines (`@` header,
/// sequence, `+` separator, quality) and the input must end with a newline —
/// a final record without a trailing `\n` is rejected (`IncompleteRecord` /
/// `TrailingBytes`). Bytes are taken verbatim: `\r` is not stripped, so CRLF
/// input either fails validation or counts the `\r` in line lengths and the
/// checksum. This is intentionally stricter than lenient FASTQ readers;
/// callers that must accept CRLF or an unterminated final record normalize
/// the input first.
pub fn parse_fastq(data: &[u8]) -> Result<FastqStats, FastqError> {
    let mut sink = FastqSink::new(data.len());
    fastq_blocks_dispatch(data, &mut sink)?;
    sink.finish()
}

/// Parallel FASTQ parser using generated mask streaming over record chunks.
///
/// A SIMD newline-count prepass finds true FASTQ record boundaries for worker
/// splits; each worker then streams generated newline masks directly into a
/// chunk-local [`FastqSink`]. This keeps the parallel path allocation-light:
/// it never materializes a full newline-position vector.
pub fn parse_fastq_par(data: &[u8], threads: usize) -> Result<FastqStats, FastqError> {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    if threads == 1 || data.len() < (1 << 20) {
        return parse_fastq(data);
    }

    let Some(bounds) = fastq_record_bounds(data, threads) else {
        return parse_fastq(data);
    };
    let ranges: Vec<(usize, usize)> = bounds
        .windows(2)
        .filter_map(|w| (w[0] < w[1]).then_some((w[0], w[1])))
        .collect();
    if ranges.len() <= 1 {
        return parse_fastq(data);
    }

    let results: Vec<Result<FastqStats, FastqError>> = std::thread::scope(|s| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(start, end)| {
                let slice = &data[start..end];
                s.spawn(move || {
                    let mut sink = FastqSink::new(slice.len());
                    fastq_blocks_dispatch(slice, &mut sink)?;
                    sink.finish()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("fastq parse thread ok"))
            .collect()
    });

    let mut merged = FastqStats::default();
    for result in results {
        match result {
            Ok(stats) => merged.merge(stats),
            Err(_) => return parse_fastq(data),
        }
    }
    Ok(merged)
}

fn fastq_blocks_dispatch(data: &[u8], sink: &mut FastqSink) -> Result<(), FastqError> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        return unsafe { avx512::fastq_blocks(data, sink) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        return unsafe { avx2::fastq_blocks(data, sink) };
    }
    unsupported_cpu()
}

fn fastq_record_bounds(data: &[u8], threads: usize) -> Option<Vec<usize>> {
    if data.is_empty() {
        return Some(vec![0]);
    }
    if data.last().copied() != Some(b'\n') {
        return None;
    }

    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        return Some(vec![0, data.len()]);
    }

    let approx: Vec<usize> = (0..=threads)
        .map(|t| {
            if t == threads {
                data.len()
            } else {
                (t * chunk).min(data.len())
            }
        })
        .collect();

    let mut bounds = Vec::with_capacity(threads + 1);
    bounds.push(0);
    for &target in approx.iter().take(threads).skip(1) {
        let bound = fastq_find_record_boundary(data, target)?;
        if bound > *bounds.last().expect("initial bound") && bound < data.len() {
            bounds.push(bound);
        }
    }
    bounds.push(data.len());
    Some(bounds)
}

/// Find a safe worker-split point at or after `offset`.
///
/// Splits must land on a true record start, so this scans forward to a line
/// beginning with `@` that revalidates as a complete four-line record
/// (`fastq_record_end_at`). A structurally wrong split makes a worker's sink
/// error and `parse_fastq_par` falls back to the serial parser, so a bad guess
/// cannot corrupt the result. The one residual (and vanishingly unlikely) case
/// is adversarial input where a quality line beginning with `@` coincidentally
/// starts a shape-valid four-line record at a chunk target: that can shift
/// framing without erroring, so the parallel record count is not trustworthy
/// on hostile, hand-crafted FASTQ.
fn fastq_find_record_boundary(data: &[u8], offset: usize) -> Option<usize> {
    if offset == 0 {
        return Some(0);
    }
    if offset >= data.len() {
        return Some(data.len());
    }

    let mut pos = offset;
    if data[pos - 1] != b'\n' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        pos = pos.saturating_add(1);
    }

    while pos < data.len() {
        if data[pos] == b'@' && fastq_record_end_at(data, pos).is_some() {
            return Some(pos);
        }
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        pos = pos.saturating_add(1);
    }
    Some(data.len())
}

fn fastq_record_end_at(data: &[u8], start: usize) -> Option<usize> {
    if data.get(start).copied() != Some(b'@') {
        return None;
    }
    let header_end = fastq_line_end(data, start)?;
    let sequence_start = header_end + 1;
    let sequence_end = fastq_line_end(data, sequence_start)?;
    let separator_start = sequence_end + 1;
    if data.get(separator_start).copied() != Some(b'+') {
        return None;
    }
    let separator_end = fastq_line_end(data, separator_start)?;
    let quality_start = separator_end + 1;
    let quality_end = fastq_line_end(data, quality_start)?;
    let sequence_len = sequence_end.checked_sub(sequence_start)?;
    let quality_len = quality_end.checked_sub(quality_start)?;
    (sequence_len == quality_len).then_some(quality_end + 1)
}

fn fastq_line_end(data: &[u8], start: usize) -> Option<usize> {
    data.get(start..)?
        .iter()
        .position(|&byte| byte == b'\n')
        .map(|end| start + end)
}

struct FastqSink {
    data_len: usize,
    line_start: usize,
    line_in_record: u8,
    sequence_start: usize,
    sequence_len: usize,
    error_quality_len: usize,
    checksum_lanes: [u64; 8],
    stats: FastqStats,
}

impl FastqSink {
    fn new(data_len: usize) -> Self {
        Self {
            data_len,
            line_start: 0,
            line_in_record: 0,
            sequence_start: 0,
            sequence_len: 0,
            error_quality_len: 0,
            checksum_lanes: [0; 8],
            stats: FastqStats::default(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn drive_avx512(
        &mut self,
        data: &[u8],
        mut mask: u64,
        base: usize,
        checksum_acc: &mut std::arch::x86_64::__m512i,
    ) -> u8 {
        while mask != 0 {
            let end = base + mask.trailing_zeros() as usize;
            let error = unsafe { self.line_avx512(data, end, checksum_acc) };
            if error != 0 {
                return error;
            }
            self.line_start = end + 1;
            mask &= mask - 1;
        }
        0
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn drive_pair_avx512(
        &mut self,
        data: &[u8],
        m0: u64,
        m1: u64,
        base: usize,
        checksum_acc: &mut std::arch::x86_64::__m512i,
    ) -> u8 {
        let error = unsafe { self.drive_avx512(data, m0, base, checksum_acc) };
        if error != 0 {
            return error;
        }
        unsafe { self.drive_avx512(data, m1, base + 64, checksum_acc) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn line_avx512(
        &mut self,
        data: &[u8],
        end: usize,
        checksum_acc: &mut std::arch::x86_64::__m512i,
    ) -> u8 {
        match self.line_in_record {
            0 => {
                if unsafe { *data.get_unchecked(self.line_start) } != b'@' {
                    return 1;
                }
                self.line_in_record = 1;
            }
            1 => {
                self.sequence_start = self.line_start;
                self.sequence_len = end - self.line_start;
                self.line_in_record = 2;
            }
            2 => {
                if unsafe { *data.get_unchecked(self.line_start) } != b'+' {
                    return 2;
                }
                self.line_in_record = 3;
            }
            _ => {
                return unsafe { self.record_avx512(data, end, checksum_acc) };
            }
        }
        0
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn record_avx512(
        &mut self,
        data: &[u8],
        end: usize,
        checksum_acc: &mut std::arch::x86_64::__m512i,
    ) -> u8 {
        let quality_len = end - self.line_start;
        if quality_len != self.sequence_len {
            self.error_quality_len = quality_len;
            return 3;
        }
        let sequence = unsafe {
            data.get_unchecked(self.sequence_start..self.sequence_start + self.sequence_len)
        };
        let quality = unsafe { data.get_unchecked(self.line_start..end) };
        self.stats.records += 1;
        self.stats.sequence_bytes += sequence.len() as u64;
        self.stats.quality_bytes += quality.len() as u64;
        unsafe { checksum_fastq_record_avx512_acc(checksum_acc, sequence, quality) };
        self.line_in_record = 0;
        0
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn drive_avx2(
        &mut self,
        data: &[u8],
        mut mask: u64,
        base: usize,
        checksum_acc: &mut std::arch::x86_64::__m256i,
    ) -> u8 {
        while mask != 0 {
            let end = base + mask.trailing_zeros() as usize;
            let error = unsafe { self.line_avx2(data, end, checksum_acc) };
            if error != 0 {
                return error;
            }
            self.line_start = end + 1;
            mask &= mask - 1;
        }
        0
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn drive_pair_avx2(
        &mut self,
        data: &[u8],
        m0: u64,
        m1: u64,
        base: usize,
        checksum_acc: &mut std::arch::x86_64::__m256i,
    ) -> u8 {
        let error = unsafe { self.drive_avx2(data, m0, base, checksum_acc) };
        if error != 0 {
            return error;
        }
        unsafe { self.drive_avx2(data, m1, base + 64, checksum_acc) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn line_avx2(
        &mut self,
        data: &[u8],
        end: usize,
        checksum_acc: &mut std::arch::x86_64::__m256i,
    ) -> u8 {
        match self.line_in_record {
            0 => {
                if unsafe { *data.get_unchecked(self.line_start) } != b'@' {
                    return 1;
                }
                self.line_in_record = 1;
            }
            1 => {
                self.sequence_start = self.line_start;
                self.sequence_len = end - self.line_start;
                self.line_in_record = 2;
            }
            2 => {
                if unsafe { *data.get_unchecked(self.line_start) } != b'+' {
                    return 2;
                }
                self.line_in_record = 3;
            }
            _ => {
                return unsafe { self.record_avx2(data, end, checksum_acc) };
            }
        }
        0
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn record_avx2(
        &mut self,
        data: &[u8],
        end: usize,
        checksum_acc: &mut std::arch::x86_64::__m256i,
    ) -> u8 {
        let quality_len = end - self.line_start;
        if quality_len != self.sequence_len {
            self.error_quality_len = quality_len;
            return 3;
        }
        let sequence = unsafe {
            data.get_unchecked(self.sequence_start..self.sequence_start + self.sequence_len)
        };
        let quality = unsafe { data.get_unchecked(self.line_start..end) };
        self.stats.records += 1;
        self.stats.sequence_bytes += sequence.len() as u64;
        self.stats.quality_bytes += quality.len() as u64;
        unsafe { checksum_fastq_record_avx2_acc(checksum_acc, sequence, quality) };
        self.line_in_record = 0;
        0
    }

    #[cold]
    fn take_error(&self, code: u8) -> FastqError {
        match code {
            1 => FastqError::BadHeader {
                offset: self.line_start,
            },
            2 => FastqError::BadSeparator {
                offset: self.line_start,
            },
            3 => FastqError::LengthMismatch {
                record: self.stats.records,
                sequence: self.sequence_len,
                quality: self.error_quality_len,
            },
            _ => unreachable!("fastq sink error code"),
        }
    }

    fn finish(mut self) -> Result<FastqStats, FastqError> {
        if self.line_in_record != 0 {
            return Err(FastqError::IncompleteRecord);
        }
        if self.line_start != self.data_len {
            return Err(FastqError::TrailingBytes {
                offset: self.line_start,
            });
        }
        self.stats.checksum = finish_fastq_checksum(&self.checksum_lanes);
        Ok(self.stats)
    }
}

impl FastqStats {
    fn merge(&mut self, other: Self) {
        self.records += other.records;
        self.sequence_bytes += other.sequence_bytes;
        self.quality_bytes += other.quality_bytes;
        self.checksum = self.checksum.wrapping_add(other.checksum);
    }
}

#[inline]
fn finish_fastq_checksum(checksum_lanes: &[u64; 8]) -> u64 {
    checksum_lanes
        .iter()
        .fold(0u64, |checksum, lane| checksum.wrapping_add(*lane))
}

#[inline]
fn checksum_bytes_scalar(mut checksum: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        let pairs = (word & 0x00ff_00ff_00ff_00ff) + ((word >> 8) & 0x00ff_00ff_00ff_00ff);
        let quads = (pairs & 0x0000_ffff_0000_ffff) + ((pairs >> 16) & 0x0000_ffff_0000_ffff);
        checksum = checksum.wrapping_add((quads & 0x0000_0000_ffff_ffff) + (quads >> 32));
    }
    for &byte in chunks.remainder() {
        checksum = checksum.wrapping_add(byte as u64);
    }
    checksum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn checksum_fastq_record_avx512_acc(
    acc: &mut std::arch::x86_64::__m512i,
    sequence: &[u8],
    quality: &[u8],
) {
    use std::arch::x86_64::{
        __m512i, __mmask64, _mm512_add_epi64, _mm512_loadu_si512, _mm512_maskz_loadu_epi8,
        _mm512_sad_epu8, _mm512_setzero_si512,
    };

    debug_assert_eq!(sequence.len(), quality.len());
    let zero = _mm512_setzero_si512();
    let mut local = *acc;
    let mut offset = 0usize;
    while offset + 64 <= sequence.len() {
        // SAFETY: offset + 64 <= both input lengths, so both unaligned loads are readable.
        let seq = unsafe { _mm512_loadu_si512(sequence.as_ptr().add(offset) as *const __m512i) };
        let qual = unsafe { _mm512_loadu_si512(quality.as_ptr().add(offset) as *const __m512i) };
        local = _mm512_add_epi64(local, _mm512_sad_epu8(seq, zero));
        local = _mm512_add_epi64(local, _mm512_sad_epu8(qual, zero));
        offset += 64;
    }
    let rem = sequence.len() - offset;
    if rem > 0 {
        let mask = ((1u64 << rem) - 1) as __mmask64;
        // SAFETY: the mask limits both loads to bytes within each slice.
        let seq = unsafe { _mm512_maskz_loadu_epi8(mask, sequence.as_ptr().add(offset) as *const i8) };
        let qual = unsafe { _mm512_maskz_loadu_epi8(mask, quality.as_ptr().add(offset) as *const i8) };
        local = _mm512_add_epi64(local, _mm512_sad_epu8(seq, zero));
        local = _mm512_add_epi64(local, _mm512_sad_epu8(qual, zero));
    }
    *acc = local;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn checksum_fastq_record_avx2_acc(
    acc: &mut std::arch::x86_64::__m256i,
    sequence: &[u8],
    quality: &[u8],
) {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_sad_epu8, _mm256_setzero_si256,
    };

    debug_assert_eq!(sequence.len(), quality.len());
    let zero = _mm256_setzero_si256();
    let mut local = *acc;
    let mut offset = 0usize;
    while offset + 32 <= sequence.len() {
        // SAFETY: offset + 32 <= both input lengths, so both unaligned loads are readable.
        let seq = unsafe { _mm256_loadu_si256(sequence.as_ptr().add(offset) as *const __m256i) };
        let qual = unsafe { _mm256_loadu_si256(quality.as_ptr().add(offset) as *const __m256i) };
        local = _mm256_add_epi64(local, _mm256_sad_epu8(seq, zero));
        local = _mm256_add_epi64(local, _mm256_sad_epu8(qual, zero));
        offset += 32;
    }
    *acc = local;
    checksum_lanes_tail(acc, &sequence[offset..], &quality[offset..]);
}

#[cfg(target_arch = "x86_64")]
fn checksum_lanes_tail(
    acc: &mut std::arch::x86_64::__m256i,
    sequence: &[u8],
    quality: &[u8],
) {
    use std::arch::x86_64::__m256i;

    // SAFETY: __m256i and [u64; 4] are both 32-byte vector storage here.
    let mut lanes: [u64; 4] = unsafe { std::mem::transmute::<__m256i, [u64; 4]>(*acc) };
    lanes[0] = checksum_bytes_scalar(lanes[0], sequence);
    lanes[0] = checksum_bytes_scalar(lanes[0], quality);
    // SAFETY: __m256i and [u64; 4] are both 32-byte vector storage here.
    *acc = unsafe { std::mem::transmute::<[u64; 4], __m256i>(lanes) };
}

#[cfg(target_arch = "aarch64")]
impl FastqSink {
    #[target_feature(enable = "neon")]
    unsafe fn drive_neon(
        &mut self,
        data: &[u8],
        mut mask: u64,
        base: usize,
        checksum_acc: &mut std::arch::aarch64::uint64x2_t,
    ) -> u8 {
        while mask != 0 {
            let end = base + mask.trailing_zeros() as usize;
            let error = unsafe { self.line_neon(data, end, checksum_acc) };
            if error != 0 {
                return error;
            }
            self.line_start = end + 1;
            mask &= mask - 1;
        }
        0
    }

    #[target_feature(enable = "neon")]
    unsafe fn drive_pair_neon(
        &mut self,
        data: &[u8],
        m0: u64,
        m1: u64,
        base: usize,
        checksum_acc: &mut std::arch::aarch64::uint64x2_t,
    ) -> u8 {
        let error = unsafe { self.drive_neon(data, m0, base, checksum_acc) };
        if error != 0 {
            return error;
        }
        unsafe { self.drive_neon(data, m1, base + 64, checksum_acc) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn line_neon(
        &mut self,
        data: &[u8],
        end: usize,
        checksum_acc: &mut std::arch::aarch64::uint64x2_t,
    ) -> u8 {
        match self.line_in_record {
            0 => {
                if unsafe { *data.get_unchecked(self.line_start) } != b'@' {
                    return 1;
                }
                self.line_in_record = 1;
            }
            1 => {
                self.sequence_start = self.line_start;
                self.sequence_len = end - self.line_start;
                self.line_in_record = 2;
            }
            2 => {
                if unsafe { *data.get_unchecked(self.line_start) } != b'+' {
                    return 2;
                }
                self.line_in_record = 3;
            }
            _ => {
                return unsafe { self.record_neon(data, end, checksum_acc) };
            }
        }
        0
    }

    #[target_feature(enable = "neon")]
    unsafe fn record_neon(
        &mut self,
        data: &[u8],
        end: usize,
        checksum_acc: &mut std::arch::aarch64::uint64x2_t,
    ) -> u8 {
        let quality_len = end - self.line_start;
        if quality_len != self.sequence_len {
            self.error_quality_len = quality_len;
            return 3;
        }
        let sequence = unsafe {
            data.get_unchecked(self.sequence_start..self.sequence_start + self.sequence_len)
        };
        let quality = unsafe { data.get_unchecked(self.line_start..end) };
        self.stats.records += 1;
        self.stats.sequence_bytes += sequence.len() as u64;
        self.stats.quality_bytes += quality.len() as u64;
        unsafe { checksum_fastq_record_neon_acc(checksum_acc, sequence, quality) };
        self.line_in_record = 0;
        0
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn checksum_fastq_record_neon_acc(
    acc: &mut std::arch::aarch64::uint64x2_t,
    sequence: &[u8],
    quality: &[u8],
) {
    use std::arch::aarch64::{vaddq_u64, vld1q_u8, vpaddlq_u16, vpaddlq_u32, vpaddlq_u8};

    debug_assert_eq!(sequence.len(), quality.len());
    let mut local = *acc;
    let mut offset = 0usize;
    while offset + 16 <= sequence.len() {
        // SAFETY: offset + 16 <= both input lengths, so both unaligned loads are readable.
        let seq = unsafe { vld1q_u8(sequence.as_ptr().add(offset)) };
        let qual = unsafe { vld1q_u8(quality.as_ptr().add(offset)) };
        // Widen-sum each 16-byte vector to two u64 lanes (bytes 0..8, 8..16)
        // and accumulate; the final fold over all lanes is the byte sum.
        local = vaddq_u64(local, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(seq))));
        local = vaddq_u64(local, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(qual))));
        offset += 16;
    }
    *acc = local;
    checksum_lanes_tail_neon(acc, &sequence[offset..], &quality[offset..]);
}

#[cfg(target_arch = "aarch64")]
fn checksum_lanes_tail_neon(
    acc: &mut std::arch::aarch64::uint64x2_t,
    sequence: &[u8],
    quality: &[u8],
) {
    use std::arch::aarch64::uint64x2_t;

    // SAFETY: uint64x2_t and [u64; 2] are both 16-byte vector storage here.
    let mut lanes: [u64; 2] = unsafe { std::mem::transmute::<uint64x2_t, [u64; 2]>(*acc) };
    lanes[0] = checksum_bytes_scalar(lanes[0], sequence);
    lanes[0] = checksum_bytes_scalar(lanes[0], quality);
    // SAFETY: uint64x2_t and [u64; 2] are both 16-byte vector storage here.
    *acc = unsafe { std::mem::transmute::<[u64; 2], uint64x2_t>(lanes) };
}

"#,
    );
}

fn push_logfmt_api(code: &mut String) {
    code.push_str(
        r#"
/// logfmt key/value summary emitted by [`parse_logfmt_pairs`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogfmtStats {
    pub pairs: u64,
    pub key_bytes: u64,
    pub value_bytes: u64,
    pub checksum: u64,
}

/// Parse logfmt pairs and return cleaned key/value counters.
///
/// This is a generated domain API over the generated structural kernel:
/// spaces, `=`, and record terminators outside quote regions are streamed
/// directly into a pair sink, avoiding the generic separator/end tape.
pub fn parse_logfmt_pairs(data: &[u8]) -> LogfmtStats {
    let mut sink = LogfmtSink::new(data.len());
    logfmt_blocks_dispatch(data, &mut sink);
    sink.finish(data)
}

/// Parallel [`parse_logfmt_pairs`] over newline-aligned chunks.
///
/// The comparable logfmt parser is line-oriented, so every worker owns whole
/// lines and starts with the normal generated quote/escape carries.
pub fn parse_logfmt_pairs_par(data: &[u8], threads: usize) -> LogfmtStats {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    if threads == 1 || data.len() < (1 << 20) {
        return parse_logfmt_pairs(data);
    }

    let bounds = logfmt_line_bounds(data, threads);
    let parts: Vec<LogfmtStats> = std::thread::scope(|s| {
        let handles: Vec<_> = bounds
            .windows(2)
            .filter_map(|w| {
                let start = w[0];
                let end = w[1];
                (start < end).then(|| {
                    let slice = &data[start..end];
                    s.spawn(move || parse_logfmt_pairs(slice))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("logfmt parse thread ok"))
            .collect()
    });

    let mut merged = LogfmtStats::default();
    for part in parts {
        merged.merge(part);
    }
    merged
}

fn logfmt_line_bounds(data: &[u8], threads: usize) -> Vec<usize> {
    let approx = data.len() / threads;
    let mut bounds = vec![0usize; threads + 1];
    bounds[threads] = data.len();
    for (i, slot) in bounds[1..threads].iter_mut().enumerate() {
        let from = ((i + 1) * approx).min(data.len());
        *slot = match data[from..].iter().position(|&byte| byte == b'\n') {
            Some(pos) => from + pos + 1,
            None => data.len(),
        };
    }
    bounds
}

fn logfmt_blocks_dispatch(data: &[u8], sink: &mut LogfmtSink) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::logfmt_blocks(data, sink) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::logfmt_blocks(data, sink) };
        return;
    }
    unsupported_cpu()
}

struct LogfmtSink {
    data_len: usize,
    record_start: usize,
    field_start: usize,
    key_start: usize,
    key_end: usize,
    has_key: bool,
    stats: LogfmtStats,
}

impl LogfmtSink {
    fn new(data_len: usize) -> Self {
        Self {
            data_len,
            record_start: 0,
            field_start: 0,
            key_start: 0,
            key_end: 0,
            has_key: false,
            stats: LogfmtStats::default(),
        }
    }

    #[inline(always)]
    fn drive(&mut self, data: &[u8], mut structural: u64, term: u64, base: usize) {
        while structural != 0 {
            let bit = structural & structural.wrapping_neg();
            let idx = bit.trailing_zeros() as usize;
            let end = base + idx;
            let is_term = (term & bit) != 0;
            let field_end =
                if is_term && end > self.record_start && data[end.saturating_sub(1)] == b'\r' {
                    end - 1
                } else {
                    end
                };
            self.field(data, field_end);
            self.field_start = end + 1;
            if is_term {
                self.has_key = false;
                self.record_start = end + 1;
            }
            structural &= structural - 1;
        }
    }

    #[inline(always)]
    fn field(&mut self, data: &[u8], end: usize) {
        if self.has_key {
            // SAFETY: all offsets come from structural positions in `data`;
            // `finish` passes at most `data_len`.
            let key = unsafe { data.get_unchecked(self.key_start..self.key_end) };
            // SAFETY: `field_start` and `end` are in the same bounded record.
            let value = unsafe { data.get_unchecked(self.field_start..end) };
            add_logfmt_pair(
                &mut self.stats,
                key,
                value,
            );
            self.has_key = false;
        } else {
            self.key_start = self.field_start;
            self.key_end = end;
            self.has_key = true;
        }
    }

    fn finish(mut self, data: &[u8]) -> LogfmtStats {
        if self.field_start < self.data_len
            || (self.data_len > 0 && data[self.data_len - 1] != b'\n')
        {
            let end = if self.data_len > self.record_start && data[self.data_len - 1] == b'\r' {
                self.data_len - 1
            } else {
                self.data_len
            };
            self.field(data, end);
        }
        self.stats
    }
}

impl LogfmtStats {
    fn merge(&mut self, other: Self) {
        self.pairs += other.pairs;
        self.key_bytes += other.key_bytes;
        self.value_bytes += other.value_bytes;
        self.checksum = self.checksum.wrapping_add(other.checksum);
    }
}

#[inline(always)]
fn add_logfmt_pair(stats: &mut LogfmtStats, key: &[u8], value: &[u8]) {
    stats.pairs += 1;
    stats.key_bytes += key.len() as u64;
    stats.checksum = logfmt_checksum_bytes(stats.checksum, key);
    add_logfmt_value(stats, value);
}

#[inline]
fn add_logfmt_value(stats: &mut LogfmtStats, value: &[u8]) {
    const Q: u8 = b'"';
    const E: u8 = b'\\';
    if value.first() != Some(&Q) {
        stats.value_bytes += value.len() as u64;
        stats.checksum = logfmt_checksum_bytes(stats.checksum, value);
        return;
    }

    let body = if value.len() >= 2 && value[value.len() - 1] == Q {
        &value[1..value.len() - 1]
    } else {
        value
    };
    if !contains_byte(body, E) {
        stats.value_bytes += body.len() as u64;
        stats.checksum = logfmt_checksum_bytes(stats.checksum, body);
        return;
    }
    
    let mut i = 0;
    while i < body.len() {
        if body[i] == E && i + 1 < body.len() {
            stats.value_bytes += 1;
            stats.checksum = stats.checksum.wrapping_add(body[i + 1] as u64);
            i += 2;
        } else {
            stats.value_bytes += 1;
            stats.checksum = stats.checksum.wrapping_add(body[i] as u64);
            i += 1;
        }
    }
}

#[inline]
fn logfmt_checksum_bytes(mut checksum: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        let pairs = (word & 0x00ff_00ff_00ff_00ff) + ((word >> 8) & 0x00ff_00ff_00ff_00ff);
        let quads = (pairs & 0x0000_ffff_0000_ffff) + ((pairs >> 16) & 0x0000_ffff_0000_ffff);
        checksum = checksum.wrapping_add((quads & 0x0000_0000_ffff_ffff) + (quads >> 32));
    }
    for &byte in chunks.remainder() {
        checksum = checksum.wrapping_add(byte as u64);
    }
    checksum
}

"#,
    );
}

fn push_columns_api(
    code: &mut String,
    dialect: &crate::formats::Dialect,
    columns: &[Column],
    par_mode: bool,
) {
    // The columns/cells parallel path uses the same record ownership as
    // parse_par. Comment lines are skipped in the sink, so comment-only
    // dialects can expose parallel projection too.
    let col_par = par_mode;
    let any_i64 = columns.iter().any(|c| c.ty == ColumnType::I64);
    let any_f64 = columns.iter().any(|c| c.ty == ColumnType::F64);
    let any_bytes = columns.iter().any(|c| c.ty == ColumnType::Bytes);
    let any_str = columns.iter().any(|c| c.ty == ColumnType::Str);

    // --- struct definition -------------------------------------------------
    code.push_str(
        "/// Typed columnar projection of the declared columns.\n\
         ///\n\
         /// Per column: a values Vec and a validity bitmap (`Vec<u64>`,\n\
         /// LSB-first; bit `r` set = row `r` parsed). A missing, empty, or\n\
         /// malformed cell clears the bit and leaves a zero placeholder in\n\
         /// the values Vec, so every column has exactly `rows` entries.\n\
         /// This layout is deliberately Arrow-primitive-array compatible.\n\
         pub struct Columns<'a> {\n\
         \x20   /// The input buffer; `bytes` column spans index into it.\n\
         \x20   pub data: &'a [u8],\n\
         \x20   /// Total number of records seen, valid or not.\n\
         \x20   pub rows: usize,\n",
    );
    for column in columns {
        let name = column.field_name();
        let idx = column.index;
        if column.ty == ColumnType::Str {
            let _ = writeln!(
                code,
                "    /// Field {idx} of each record, cleaned, in Arrow varbinary\n\
                 \x20   /// layout: `{name}_offsets[r]..{name}_offsets[r + 1]` indexes\n\
                 \x20   /// `{name}_data` (use [`string_at`]). Always rows + 1 entries.\n\
                 \x20   pub {name}_offsets: Vec<i32>,\n\
                 \x20   /// Contiguous cleaned bytes of field {idx} of every record.\n\
                 \x20   pub {name}_data: Vec<u8>,\n\
                 \x20   /// Validity bitmap for `{name}` (missing field = invalid;\n\
                 \x20   /// an empty cell is a valid empty string).\n\
                 \x20   pub {name}_valid: Vec<u64>,"
            );
        } else {
            let ty = column.rust_type();
            let what = match column.ty {
                ColumnType::I64 => "as `i64`",
                ColumnType::F64 => "as `f64`",
                ColumnType::Bytes => "as raw `(start, end)` spans into `data`",
                // String columns emit via the Str branch above; emission branches on the column type before this is called.
                ColumnType::Str => unreachable!(),
            };
            let _ = writeln!(
                code,
                "    /// Field {idx} of each record {what}; zero where invalid.\n\
                 \x20   pub {name}: Vec<{ty}>,\n\
                 \x20   /// Validity bitmap for `{name}`.\n\
                 \x20   pub {name}_valid: Vec<u64>,"
            );
        }
    }
    code.push_str("}\n\n");

    // --- impl: constructor, span resolver, row push ------------------------
    code.push_str("impl<'a> Columns<'a> {\n    fn with_capacity(data: &'a [u8], rows: usize) -> Self {\n        Columns {\n            data,\n            rows: 0,\n");
    for column in columns {
        let name = column.field_name();
        if column.ty == ColumnType::Str {
            // Offsets carry the Arrow invariant of a leading 0 from birth,
            // so serial fill and parallel merge share it.
            let _ = writeln!(
                code,
                "            {name}_offsets: {{ let mut v = Vec::with_capacity(rows + 1); v.push(0); v }},\n\
                 \x20           {name}_data: Vec::new(),\n\
                 \x20           {name}_valid: Vec::with_capacity(rows / 64 + 1),"
            );
        } else {
            let _ = writeln!(
                code,
                "            {name}: Vec::with_capacity(rows),\n\
                 \x20           {name}_valid: Vec::with_capacity(rows / 64 + 1),"
            );
        }
    }
    code.push_str("        }\n    }\n");

    if any_bytes {
        code.push_str(
            "\n    /// Resolve a `bytes` column span to its slice of the input.\n\
             \x20   #[inline]\n\
             \x20   pub fn span(&self, span: (u32, u32)) -> &'a [u8] {\n\
             \x20       &self.data[span.0 as usize..span.1 as usize]\n\
             \x20   }\n",
        );
    }

    code.push_str("}\n\n");

    // --- the projection sink ------------------------------------------------
    // Per-record state is three registers plus a K-slot pending-span
    // array; separators only ever bump a counter (and store a span when
    // the ordinal is declared), terminators flush one row. No tape is
    // materialized anywhere on this path.
    let k_count = columns.len();
    let mut by_ordinal: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (k, column) in columns.iter().enumerate() {
        by_ordinal.entry(column.index).or_default().push(k);
    }
    let arms = |end_expr: &str| -> String {
        let mut out = String::new();
        for (ordinal, ks) in &by_ordinal {
            let mut stores = String::new();
            for &k in ks {
                let _ = write!(
                    stores,
                    "self.pending[{k}] = (self.field_start, {end_expr}); self.found |= {}; ",
                    1u32 << k
                );
            }
            let _ = writeln!(out, "            {ordinal} => {{ {stores}}}");
        }
        out
    };
    let sep_arms = arms("p");
    let last_arms = arms("to");
    // Comment dialects: a record beginning with the comment byte is not a
    // row at all (mirrors the Records iterator's lazy skip).
    let comment_guard = match dialect.comment {
        Some(c) => format!(
            "\x20       // Comment record: not a row.\n\
             \x20       if end > self.record_start && data[self.record_start as usize] == {c}u8 {{\n\
             \x20           return;\n\
             \x20       }}\n"
        ),
        None => String::new(),
    };

    let _ = write!(
        code,
        "/// Streaming projection state: drivers feed structural masks in,\n\
         /// finished rows come out. `end` bounds the record range this sink\n\
         /// owns (records are assigned by terminator position; a sink may\n\
         /// overrun `end` by one record to finish what it started).\n\
         pub(crate) struct ColumnSink<'a> {{\n\
         \x20   cols: Columns<'a>,\n\
         \x20   field_start: u32,\n\
         \x20   record_start: u32,\n\
         \x20   ordinal: u32,\n\
         \x20   pending: [(u32, u32); {k_count}],\n\
         \x20   /// Bit k set = declared column k's span is pending this record.\n\
         \x20   found: u32,\n\
         \x20   end: u32,\n\
         \x20   /// False until the first terminator at or past the sink's start\n\
         \x20   /// (the partial record before it belongs to the previous sink).\n\
         \x20   emitting: bool,\n\
         \x20   pub(crate) done: bool,\n\
         }}\n\n\
         impl<'a> ColumnSink<'a> {{\n\
         \x20   pub(crate) fn new(data: &'a [u8], start: u32, end: u32, emitting: bool) -> ColumnSink<'a> {{\n\
         \x20       ColumnSink {{\n\
         \x20           cols: Columns::with_capacity(data, (end - start) as usize / 32 + 8),\n\
         \x20           field_start: start,\n\
         \x20           record_start: start,\n\
         \x20           ordinal: 0,\n\
         \x20           pending: [(0, 0); {k_count}],\n\
         \x20           found: 0,\n\
         \x20           end,\n\
         \x20           emitting,\n\
         \x20           done: false,\n\
         \x20       }}\n\
         \x20   }}\n\n\
         \x20   #[inline]\n\
         \x20   pub(crate) fn drive(&mut self, mask: u64, term: u64, base: u32) {{\n\
         \x20       let mut m = mask;\n\
         \x20       while m != 0 {{\n\
         \x20           let bit = m & m.wrapping_neg();\n\
         \x20           let p = base + m.trailing_zeros();\n\
         \x20           if term & bit != 0 {{\n\
         \x20               if self.emitting {{\n\
         \x20                   self.flush(p);\n\
         \x20               }} else {{\n\
         \x20                   self.emitting = true;\n\
         \x20               }}\n\
         \x20               self.record_start = p + 1;\n\
         \x20               self.field_start = p + 1;\n\
         \x20               self.ordinal = 0;\n\
         \x20               self.found = 0;\n\
         \x20               if p >= self.end {{\n\
         \x20                   self.done = true;\n\
         \x20                   return;\n\
         \x20               }}\n\
         \x20           }} else if self.emitting {{\n\
         \x20               match self.ordinal {{\n\
         {sep_arms}\
         \x20               _ => {{}}\n\
         \x20               }}\n\
         \x20               self.ordinal += 1;\n\
         \x20               self.field_start = p + 1;\n\
         \x20           }}\n\
         \x20           m &= m - 1;\n\
         \x20       }}\n\
         \x20   }}\n\n\
         \x20   /// Emit the row terminated (exclusively) at `end`.\n\
         \x20   fn flush(&mut self, end: u32) {{\n\
         \x20       let data = self.cols.data;\n\
         {comment_guard}\
         \x20       // Record-level trim of the `\\r` in `\\r\\n` terminators.\n\
         \x20       let to = if end > self.record_start && data[end as usize - 1] == b'\\r' {{\n\
         \x20           end - 1\n\
         \x20       }} else {{\n\
         \x20           end\n\
         \x20       }};\n\
         \x20       match self.ordinal {{\n\
         {last_arms}\
         \x20       _ => {{}}\n\
         \x20       }}\n\
         \x20       let row = self.cols.rows;\n\
         \x20       if row & 63 == 0 {{\n"
    );
    for column in columns {
        let _ = writeln!(
            code,
            "            self.cols.{}_valid.push(0);",
            column.field_name()
        );
    }
    code.push_str("        }\n");
    for (k, column) in columns.iter().enumerate() {
        let name = column.field_name();
        let found_bit = 1u32 << k;
        let body = match column.ty {
            ColumnType::I64 | ColumnType::F64 => {
                let parser = if column.ty == ColumnType::I64 {
                    if dialect.quote.is_some() {
                        "parse_i64_field"
                    } else {
                        "parse_i64_cell"
                    }
                } else if dialect.quote.is_some() {
                    "parse_f64_field"
                } else {
                    "parse_f64_cell"
                };
                let zero = column.zero();
                format!(
                    "        let (cfrom, cto) = self.pending[{k}];\n\
                     \x20       let v = if self.found & {found_bit} != 0 {{\n\
                     \x20           {parser}(&data[cfrom as usize..cto as usize])\n\
                     \x20       }} else {{\n\
                     \x20           None\n\
                     \x20       }};\n\
                     \x20       self.cols.{name}.push(v.unwrap_or({zero}));\n\
                     \x20       self.cols.{name}_valid[row >> 6] |= (v.is_some() as u64) << (row & 63);\n"
                )
            }
            ColumnType::Bytes => format!(
                "        let (cfrom, cto) = self.pending[{k}];\n\
                 \x20       let ok = self.found & {found_bit} != 0 && cfrom != cto;\n\
                 \x20       self.cols.{name}.push(if ok {{ (cfrom, cto) }} else {{ (0, 0) }});\n\
                 \x20       self.cols.{name}_valid[row >> 6] |= (ok as u64) << (row & 63);\n"
            ),
            ColumnType::Str => format!(
                "        let (cfrom, cto) = self.pending[{k}];\n\
                 \x20       let ok = self.found & {found_bit} != 0;\n\
                 \x20       if ok {{\n\
                 \x20           append_clean(&mut self.cols.{name}_data, &data[cfrom as usize..cto as usize]);\n\
                 \x20       }}\n\
                 \x20       assert!(\n\
                 \x20           self.cols.{name}_data.len() <= i32::MAX as usize,\n\
                 \x20           \"string column '{name}' exceeds the 2 GiB Arrow i32-offset limit\"\n\
                 \x20       );\n\
                 \x20       self.cols.{name}_offsets.push(self.cols.{name}_data.len() as i32);\n\
                 \x20       self.cols.{name}_valid[row >> 6] |= (ok as u64) << (row & 63);\n"
            ),
        };
        code.push_str(&body);
    }
    code.push_str(
        "        self.cols.rows = row + 1;\n\
         \x20   }\n\n\
         \x20   /// Flush any trailing unterminated record and surrender the\n\
         \x20   /// columns. Exactly one sink owns the trailer: the one still\n\
         \x20   /// emitting at end of data (sinks past it never saw a\n\
         \x20   /// terminator and never started).\n\
         \x20   pub(crate) fn finish(mut self) -> Columns<'a> {\n\
         \x20       let len = self.cols.data.len() as u32;\n\
         \x20       if self.emitting && !self.done && self.record_start < len {\n\
         \x20           self.flush(len);\n\
         \x20       }\n\
         \x20       self.cols\n\
         \x20   }\n\
         }\n\n",
    );

    // --- bitmap accessor ----------------------------------------------------
    code.push_str(
        "/// Test bit `row` of a validity bitmap.\n\
         #[inline]\n\
         pub fn bitmap_get(bitmap: &[u64], row: usize) -> bool {\n\
         \x20   bitmap[row >> 6] >> (row & 63) & 1 != 0\n\
         }\n\n",
    );
    if any_str {
        code.push_str(
            "/// Slice row `row` of a string column out of its offsets + data\n\
             /// buffers.\n\
             #[inline]\n\
             pub fn string_at<'b>(offsets: &[i32], data: &'b [u8], row: usize) -> &'b [u8] {\n\
             \x20   &data[offsets[row] as usize..offsets[row + 1] as usize]\n\
             }\n\n",
        );
    }

    // --- serial entry point -------------------------------------------------
    let (dispatch_sig, dispatch_serial_args, seed_param) = if col_par {
        (
            "data: &[u8], seed: u64, start: usize, sink: &mut ColumnSink",
            "data, 0, 0, &mut sink",
            "data, seed, start, sink",
        )
    } else {
        (
            "data: &[u8], sink: &mut ColumnSink",
            "data, &mut sink",
            "data, sink",
        )
    };
    let _ = write!(
        code,
        "/// Parse `data` into typed columns: a fused single pass feeds the\n\
         /// structural masks straight into the projection sink, so no tape is\n\
         /// built and only the declared columns' bytes are ever inspected.\n\
         pub fn parse_columns(data: &[u8]) -> Columns<'_> {{\n\
         \x20   let mut sink = ColumnSink::new(data, 0, data.len() as u32, true);\n\
         \x20   index_cells_dispatch({dispatch_serial_args});\n\
         \x20   sink.finish()\n\
         }}\n\n\
         fn index_cells_dispatch({dispatch_sig}) {{\n\
         \x20   #[cfg(target_arch = \"x86_64\")]\n\
         \x20   if std::arch::is_x86_feature_detected!(\"avx512f\")\n\
         \x20       && std::arch::is_x86_feature_detected!(\"avx512bw\")\n\
         \x20       && std::arch::is_x86_feature_detected!(\"avx512vl\")\n\
         \x20       && std::arch::is_x86_feature_detected!(\"pclmulqdq\")\n\
         \x20   {{\n\
         \x20       // SAFETY: the required target features were just detected.\n\
         \x20       unsafe {{ avx512::index_cells({seed_param}) }};\n\
         \x20       return;\n\
         \x20   }}\n\
         \x20   #[cfg(target_arch = \"x86_64\")]\n\
         \x20   if std::arch::is_x86_feature_detected!(\"avx2\")\n\
         \x20       && std::arch::is_x86_feature_detected!(\"pclmulqdq\")\n\
         \x20   {{\n\
         \x20       // SAFETY: the required target features were just detected.\n\
         \x20       unsafe {{ avx2::index_cells({seed_param}) }};\n\
         \x20       return;\n\
         \x20   }}\n\
         \x20   unsupported_cpu();\n\
         }}\n\n"
    );

    // --- parallel entry point ------------------------------------------------
    if col_par {
        let (par_shared, par_refs, par_seed) = par_seed_parts(dialect.quote, "data[start..end]");
        let _ = write!(
            code,
            "/// Parallel [`parse_columns`]: records are assigned to workers by\n\
             /// terminator ownership — worker t skips to the first record\n\
             /// boundary at or past its chunk start, and finishes the record it\n\
             /// is mid-way through at chunk end — so every record is converted\n\
             /// exactly once, with no tape built. Quote context is resolved per\n\
             /// chunk inside the scope (one barrier), as in [`parse_par`]; column\n\
             /// chunks concatenate, validity bitmaps stitch bit-shifted.\n\
             pub fn parse_columns_par(data: &[u8], threads: usize) -> Columns<'_> {{\n\
             \x20   let threads = threads.max(1).min(data.len() / 64 + 1);\n\
             \x20   let chunk = (data.len() / threads + 63) & !63;\n\
             \x20   if threads == 1 || chunk == 0 {{\n\
             \x20       return parse_columns(data);\n\
             \x20   }}\n\
             \x20   let bounds: Vec<usize> = (0..=threads)\n\
             \x20       .map(|t| if t == threads {{ data.len() }} else {{ (t * chunk).min(data.len()) }})\n\
             \x20       .collect();\n\
             {par_shared}\
             \x20   let parts: Vec<Columns<'_>> = std::thread::scope(|s| {{\n\
             \x20       let handles: Vec<_> = (0..threads)\n\
             \x20           .map(|t| {{\n\
             \x20               let (start, end) = (bounds[t], bounds[t + 1]);\n\
             {par_refs}\
             \x20               s.spawn(move || {{\n\
             {par_seed}\
             \x20                   let mut sink = ColumnSink::new(data, start as u32, end as u32, t == 0);\n\
             \x20                   index_cells_dispatch(data, seed, start, &mut sink);\n\
             \x20                   sink.finish()\n\
             \x20               }})\n\
             \x20           }})\n\
             \x20           .collect();\n\
             \x20       handles.into_iter().map(|h| h.join().expect(\"columns thread ok\")).collect()\n\
             \x20   }});\n\
             \x20   let mut cols = Columns::with_capacity(data, parts.iter().map(|p| p.rows).sum::<usize>());\n\
             \x20   for part in &parts {{\n"
        );
        for column in columns {
            let name = column.field_name();
            if column.ty == ColumnType::Str {
                // String chunks: data buffers concatenate, offsets rebase by
                // the merged data length (skipping each part's leading 0).
                let _ = writeln!(
                    code,
                    "        let {name}_base = cols.{name}_data.len();\n\
                     \x20       assert!(\n\
                     \x20           {name}_base + part.{name}_data.len() <= i32::MAX as usize,\n\
                     \x20           \"string column '{name}' exceeds the 2 GiB Arrow i32-offset limit\"\n\
                     \x20       );\n\
                     \x20       cols.{name}_data.extend_from_slice(&part.{name}_data);\n\
                     \x20       cols.{name}_offsets\n\
                     \x20           .extend(part.{name}_offsets[1..].iter().map(|&o| o + {name}_base as i32));\n\
                     \x20       append_bitmap(&mut cols.{name}_valid, cols.rows, &part.{name}_valid, part.rows);"
                );
            } else {
                let _ = writeln!(
                    code,
                    "        cols.{name}.extend_from_slice(&part.{name});\n\
                     \x20       append_bitmap(&mut cols.{name}_valid, cols.rows, &part.{name}_valid, part.rows);"
                );
            }
        }
        code.push_str(
            "        cols.rows += part.rows;\n\
             \x20   }\n\
             \x20   cols\n\
             }\n\n\
             /// Append `src_rows` bits of `src` onto a bitmap currently holding\n\
             /// `dst_rows` bits. Bits past each bitmap's row count are zero, an\n\
             /// invariant `push_row` maintains and this function preserves.\n\
             fn append_bitmap(dst: &mut Vec<u64>, dst_rows: usize, src: &[u64], src_rows: usize) {\n\
             \x20   let words = src_rows.div_ceil(64);\n\
             \x20   let shift = dst_rows & 63;\n\
             \x20   if shift == 0 {\n\
             \x20       dst.extend_from_slice(&src[..words]);\n\
             \x20   } else {\n\
             \x20       for &word in &src[..words] {\n\
             \x20           *dst.last_mut().expect(\"non-aligned dst has a partial word\") |= word << shift;\n\
             \x20           dst.push(word >> (64 - shift));\n\
             \x20       }\n\
             \x20       dst.truncate((dst_rows + src_rows).div_ceil(64));\n\
             \x20   }\n\
             }\n\n",
        );
    }

    // --- cell parsers ---------------------------------------------------------
    if any_i64 {
        code.push_str(INT_CELL_TPL);
    }
    if any_f64 {
        code.push_str(FLOAT_CELL_TPL);
    }
    if any_str {
        code.push_str(&append_clean_fn(dialect));
    }
    // Quoted dialects get a raw-span wrapper per numeric type: the
    // unquoted common case parses in place without constructing a Cow,
    // quoted cells are cleaned first. (Quoteless dialects call the cell
    // parsers directly.)
    if let Some(q) = dialect.quote {
        for (cond, ty, parser) in [
            (any_i64, "i64", "parse_i64_cell"),
            (any_f64, "f64", "parse_f64_cell"),
        ] {
            if cond {
                let _ = write!(
                    code,
                    "/// Parse a numeric cell from its raw field span; only quoted\n\
                     /// cells pay for cleaning.\n\
                     #[inline]\n\
                     fn {parser_field}(raw: &[u8]) -> Option<{ty}> {{\n\
                     \x20   if raw.first() == Some(&{q}u8) {{\n\
                     \x20       {parser}(&clean(raw))\n\
                     \x20   }} else {{\n\
                     \x20       {parser}(raw)\n\
                     \x20   }}\n\
                     }}\n\n",
                    parser_field = parser.replace("_cell", "_field"),
                );
            }
        }
    }
}

fn push_ndjson_lines_api(code: &mut String) {
    code.push_str(
        r#"
/// NDJSON line-framing summary emitted by [`parse_ndjson_lines`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NdjsonLineStats {
    pub records: u64,
}

/// Count NDJSON records directly from the generated quote/escape-aware
/// newline stream without materializing the span tape.
pub fn parse_ndjson_lines(data: &[u8]) -> NdjsonLineStats {
    let mut records = index_ndjson_lines_dispatch(data);
    if ndjson_has_trailing_record(data) {
        records += 1;
    }
    NdjsonLineStats { records }
}

/// Parallel [`parse_ndjson_lines`] over newline-aligned chunks.
///
/// NDJSON records are JSON texts separated by raw newlines; a valid record
/// cannot contain an unescaped newline inside a string. Aligning chunks just
/// after raw newlines therefore lets every worker start from the normal
/// generated quote/escape state while still using the generated scanner.
pub fn parse_ndjson_lines_par(data: &[u8], threads: usize) -> NdjsonLineStats {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    if threads == 1 || data.is_empty() {
        return parse_ndjson_lines(data);
    }
    let bounds = ndjson_line_bounds(data, threads);
    if bounds.len() <= 2 {
        return parse_ndjson_lines(data);
    }
    let records: u64 = std::thread::scope(|s| {
        let handles: Vec<_> = bounds
            .windows(2)
            .map(|range| {
                let slice = &data[range[0]..range[1]];
                s.spawn(move || index_ndjson_lines_dispatch(slice))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("ndjson line thread ok"))
            .sum()
    });
    NdjsonLineStats {
        records: records + ndjson_has_trailing_record(data) as u64,
    }
}

fn index_ndjson_lines_dispatch(data: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        return unsafe { avx512::index_ndjson_lines(data) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        return unsafe { avx2::index_ndjson_lines(data) };
    }
    unsupported_cpu();
}

fn ndjson_has_trailing_record(data: &[u8]) -> bool {
    !data.is_empty() && data.last().copied() != Some(b'\n')
}

fn ndjson_line_bounds(data: &[u8], threads: usize) -> Vec<usize> {
    if data.is_empty() {
        return vec![0];
    }
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        return vec![0, data.len()];
    }
    let mut bounds = Vec::with_capacity(threads + 1);
    bounds.push(0);
    for t in 1..threads {
        let mut bound = (t * chunk).min(data.len());
        while bound < data.len() && data[bound - 1] != b'\n' {
            bound += 1;
        }
        if bound > *bounds.last().expect("initial bound") && bound < data.len() {
            bounds.push(bound);
        }
    }
    bounds.push(data.len());
    bounds
}

"#,
    );
}

fn push_field_byte_stats_api(code: &mut String, dialect: &crate::formats::Dialect) {
    let (par_shared, par_refs, par_seed) = par_seed_parts(dialect.quote, "data[start..end]");
    let field_len_helper = if dialect.quote.is_some() {
        r#"#[inline]
fn field_byte_len(raw: &[u8]) -> u64 {
    const Q: u8 = 34u8;
    if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {
        let inner = &raw[1..raw.len() - 1];
        if !contains_byte(inner, Q) {
            return inner.len() as u64;
        }
        let mut len = 0u64;
        let mut i = 0;
        while i < inner.len() {
            len += 1;
            if inner[i] == Q && i + 1 < inner.len() && inner[i + 1] == Q {
                i += 2;
            } else {
                i += 1;
            }
        }
        return len;
    }
    raw.len() as u64
}

"#
    } else {
        r#"#[inline]
fn field_byte_len(raw: &[u8]) -> u64 {
    raw.len() as u64
}

"#
    };
    let api = r#"
/// Fused field-byte summary emitted by [`parse_field_bytes`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldByteStats {
    pub fields: u64,
    pub bytes: u64,
}

/// Sum decoded field byte lengths directly from the generated structural
/// stream.
///
/// This keeps the benchmark on the same semantic surface as
/// `records().flat_map(fields...)`, but avoids building a tape and then
/// walking it again.
pub fn parse_field_bytes(data: &[u8]) -> FieldByteStats {
    let mut sink = FieldByteSink::new(data, 0, data.len() as u32, true);
    index_field_bytes_dispatch(data, 0, 0, &mut sink);
    sink.finish()
}

/// Parallel [`parse_field_bytes`] using the same record-ownership contract as
/// [`parse_par`].
pub fn parse_field_bytes_par(data: &[u8], threads: usize) -> FieldByteStats {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        return parse_field_bytes(data);
    }
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| {
            if t == threads {
                data.len()
            } else {
                (t * chunk).min(data.len())
            }
        })
        .collect();
@PAR_SHARED@    let parts: Vec<FieldByteStats> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (start, end) = (bounds[t], bounds[t + 1]);
@PAR_REFS@                s.spawn(move || {
@PAR_SEED@                    let mut sink = FieldByteSink::new(data, start as u32, end as u32, t == 0);
                    index_field_bytes_dispatch(data, seed, start, &mut sink);
                    sink.finish()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("field-byte stats thread ok"))
            .collect()
    });
    let mut stats = FieldByteStats::default();
    for part in parts {
        stats.merge(part);
    }
    stats
}

fn index_field_bytes_dispatch(data: &[u8], seed: u64, start: usize, sink: &mut FieldByteSink) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::index_field_bytes(data, seed, start, sink) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_field_bytes(data, seed, start, sink) };
        return;
    }
    unsupported_cpu();
}

pub(crate) struct FieldByteSink<'a> {
    data: &'a [u8],
    stats: FieldByteStats,
    field_start: u32,
    record_start: u32,
    end: u32,
    emitting: bool,
    pub(crate) done: bool,
}

impl<'a> FieldByteSink<'a> {
    fn new(data: &'a [u8], start: u32, end: u32, emitting: bool) -> Self {
        Self {
            data,
            stats: FieldByteStats::default(),
            field_start: start,
            record_start: start,
            end,
            emitting,
            done: false,
        }
    }

    #[inline]
    pub(crate) fn drive(&mut self, mask: u64, term: u64, base: u32) {
        let mut m = mask;
        while m != 0 {
            let bit = m & m.wrapping_neg();
            let p = base + m.trailing_zeros();
            if term & bit != 0 {
                if self.emitting {
                    self.flush_field(p, true);
                } else {
                    self.emitting = true;
                }
                self.record_start = p + 1;
                self.field_start = p + 1;
                if p >= self.end {
                    self.done = true;
                    return;
                }
            } else if self.emitting {
                self.flush_field(p, false);
                self.field_start = p + 1;
            }
            m &= m - 1;
        }
    }

    #[inline]
    fn flush_field(&mut self, end: u32, record_end: bool) {
        let data = self.data;
        let to = if record_end && end > self.record_start && data[end as usize - 1] == b'\r' {
            end - 1
        } else {
            end
        };
        self.stats.fields += 1;
        self.stats.bytes += field_byte_len(&data[self.field_start as usize..to as usize]);
    }

    fn finish(mut self) -> FieldByteStats {
        let len = self.data.len() as u32;
        if self.emitting && !self.done && self.record_start < len {
            self.flush_field(len, true);
        }
        self.stats
    }
}

impl FieldByteStats {
    fn merge(&mut self, other: Self) {
        self.fields += other.fields;
        self.bytes += other.bytes;
    }
}

@FIELD_LEN_HELPER@"#
        .replace("@PAR_SHARED@", &par_shared)
        .replace("@PAR_REFS@", &par_refs)
        .replace("@PAR_SEED@", &par_seed)
        .replace("@FIELD_LEN_HELPER@", field_len_helper);

    code.push_str(&api);
}

fn push_csv_geo_stats_api(code: &mut String, text: bool) {
    let (struct_name, sink_name, parse_fn, parse_par_fn, dispatch_fn, index_fn, thread_msg) =
        if text {
            (
                "CsvGeoTextStats",
                "CsvGeoTextStatsSink",
                "parse_csv_geo_text_stats",
                "parse_csv_geo_text_stats_par",
                "index_csv_geo_text_stats_dispatch",
                "index_csv_geo_text_stats",
                "csv geo text stats thread ok",
            )
        } else {
            (
                "CsvGeoStats",
                "CsvGeoStatsSink",
                "parse_csv_geo_stats",
                "parse_csv_geo_stats_par",
                "index_csv_geo_stats_dispatch",
                "index_csv_geo_stats",
                "csv geo stats thread ok",
            )
        };

    let (
        text_struct_fields,
        pending_count,
        text_drive_arm,
        text_last_arm,
        text_flush_body,
        text_merge_body,
        text_helpers,
        lat_idx,
        lat_bit,
        lon_idx,
        lon_bit,
    ) = if text {
        (
            "\
    pub city_values: u64,\n\
    pub city_bytes: u64,\n\
    pub city_checksum: u64,\n",
            "3",
            "\
                    1 => {\n\
                        self.pending[0] = (self.field_start, p);\n\
                        self.found |= 1;\n\
                    }\n",
            "\
            1 => {\n\
                self.pending[0] = (self.field_start, to);\n\
                self.found |= 1;\n\
            }\n",
            "\
        let (from, to) = self.pending[0];\n\
        if self.found & 1 != 0 {\n\
            let (bytes, checksum) = csv_geo_checksum_clean_field(&data[from as usize..to as usize]);\n\
            self.stats.city_values += 1;\n\
            self.stats.city_bytes += bytes;\n\
            self.stats.city_checksum = self.stats.city_checksum.wrapping_add(checksum);\n\
        }\n",
            "\
        self.city_values += other.city_values;\n\
        self.city_bytes += other.city_bytes;\n\
        self.city_checksum = self.city_checksum.wrapping_add(other.city_checksum);\n",
            r#"
fn csv_geo_checksum_clean_field(raw: &[u8]) -> (u64, u64) {
    const Q: u8 = 34u8;
    if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {
        let inner = &raw[1..raw.len() - 1];
        if !contains_byte(inner, Q) {
            return (inner.len() as u64, csv_geo_checksum_bytes(inner));
        }
        let mut bytes = 0u64;
        let mut checksum = 0u64;
        let mut i = 0;
        while i < inner.len() {
            bytes += 1;
            checksum = checksum.wrapping_add(inner[i] as u64);
            if inner[i] == Q && i + 1 < inner.len() && inner[i + 1] == Q {
                i += 2;
            } else {
                i += 1;
            }
        }
        return (bytes, checksum);
    }
    (raw.len() as u64, csv_geo_checksum_bytes(raw))
}

#[inline]
fn csv_geo_checksum_bytes(bytes: &[u8]) -> u64 {
    match bytes.len() {
        0 => 0,
        1 => bytes[0] as u64,
        2 => bytes[0] as u64 + bytes[1] as u64,
        3 => bytes[0] as u64 + bytes[1] as u64 + bytes[2] as u64,
        4 => bytes[0] as u64 + bytes[1] as u64 + bytes[2] as u64 + bytes[3] as u64,
        5 => {
            bytes[0] as u64
                + bytes[1] as u64
                + bytes[2] as u64
                + bytes[3] as u64
                + bytes[4] as u64
        }
        6 => {
            bytes[0] as u64
                + bytes[1] as u64
                + bytes[2] as u64
                + bytes[3] as u64
                + bytes[4] as u64
                + bytes[5] as u64
        }
        7 => {
            bytes[0] as u64
                + bytes[1] as u64
                + bytes[2] as u64
                + bytes[3] as u64
                + bytes[4] as u64
                + bytes[5] as u64
                + bytes[6] as u64
        }
        8 => csv_geo_checksum_first_8(bytes),
        9 => csv_geo_checksum_first_8(bytes) + bytes[8] as u64,
        10 => csv_geo_checksum_first_8(bytes) + bytes[8] as u64 + bytes[9] as u64,
        11 => csv_geo_checksum_first_8(bytes) + bytes[8] as u64 + bytes[9] as u64 + bytes[10] as u64,
        12 => {
            csv_geo_checksum_first_8(bytes)
                + bytes[8] as u64
                + bytes[9] as u64
                + bytes[10] as u64
                + bytes[11] as u64
        }
        13 => {
            csv_geo_checksum_first_8(bytes)
                + bytes[8] as u64
                + bytes[9] as u64
                + bytes[10] as u64
                + bytes[11] as u64
                + bytes[12] as u64
        }
        14 => {
            csv_geo_checksum_first_8(bytes)
                + bytes[8] as u64
                + bytes[9] as u64
                + bytes[10] as u64
                + bytes[11] as u64
                + bytes[12] as u64
                + bytes[13] as u64
        }
        15 => {
            csv_geo_checksum_first_8(bytes)
                + bytes[8] as u64
                + bytes[9] as u64
                + bytes[10] as u64
                + bytes[11] as u64
                + bytes[12] as u64
                + bytes[13] as u64
                + bytes[14] as u64
        }
        16 => csv_geo_checksum_first_8(bytes) + csv_geo_checksum_first_8(&bytes[8..]),
        _ => csv_geo_checksum_long(bytes),
    }
}

#[inline]
fn csv_geo_checksum_first_8(bytes: &[u8]) -> u64 {
    debug_assert!(bytes.len() >= 8);
    // SAFETY: callers pass at least 8 bytes; unaligned loads are allowed.
    let word = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<u64>()) };
    csv_geo_checksum_word(word)
}

#[inline]
fn csv_geo_checksum_long(bytes: &[u8]) -> u64 {
    let mut checksum = 0u64;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        checksum = checksum.wrapping_add(csv_geo_checksum_word(word));
    }
    for &byte in chunks.remainder() {
        checksum = checksum.wrapping_add(byte as u64);
    }
    checksum
}

#[inline]
fn csv_geo_checksum_word(word: u64) -> u64 {
    let pairs = (word & 0x00ff_00ff_00ff_00ff) + ((word >> 8) & 0x00ff_00ff_00ff_00ff);
    let quads = (pairs & 0x0000_ffff_0000_ffff) + ((pairs >> 16) & 0x0000_ffff_0000_ffff);
    (quads & 0x0000_0000_ffff_ffff) + (quads >> 32)
}

"#,
            "1",
            "2",
            "2",
            "4",
        )
    } else {
        ("", "2", "", "", "", "", "", "0", "1", "1", "2")
    };
    let api = r#"
/// CSV geo projection summary emitted by [`@PARSE@`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct @STRUCT@ {
    pub records: u64,
@TEXT_STRUCT_FIELDS@    pub latitude_values: u64,
    pub longitude_values: u64,
    pub latitude_checksum: u64,
    pub longitude_checksum: u64,
}

/// Parse CSV geo rows and accumulate the benchmark projection directly.
///
/// This generated domain API uses the same SIMD structural stream as
/// [`parse_columns`], but avoids materializing column vectors, string
/// offsets, and validity bitmaps that aggregate benchmarks immediately
/// walk again.
pub fn @PARSE@(data: &[u8]) -> @STRUCT@ {
    let mut sink = @SINK@::new(data, 0, data.len() as u32, true);
    @DISPATCH@(data, 0, 0, &mut sink);
    sink.finish()
}

/// Parallel [`@PARSE@`] using the same record-ownership contract as
/// [`parse_columns_par`].
pub fn @PARSE_PAR@(data: &[u8], threads: usize) -> @STRUCT@ {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        return @PARSE@(data);
    }
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| {
            if t == threads {
                data.len()
            } else {
                (t * chunk).min(data.len())
            }
        })
        .collect();
    let parity: Vec<std::sync::atomic::AtomicU64> =
        (0..threads).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    let barrier = std::sync::Barrier::new(threads);
    let parts: Vec<@STRUCT@> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (start, end) = (bounds[t], bounds[t + 1]);
                let parity = &parity;
                let barrier = &barrier;
                s.spawn(move || {
                    use std::sync::atomic::Ordering::Relaxed;
                    parity[t].store(quote_parity_dispatch(&data[start..end], 0) & 1, Relaxed);
                    barrier.wait();
                    let seed = parity[..t]
                        .iter()
                        .fold(0u64, |acc, p| acc ^ p.load(Relaxed))
                        .wrapping_neg();
                    let mut sink = @SINK@::new(data, start as u32, end as u32, t == 0);
                    @DISPATCH@(data, seed, start, &mut sink);
                    sink.finish()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("@THREAD_MSG@"))
            .collect()
    });
    let mut stats = @STRUCT@::default();
    for part in parts {
        stats.merge(part);
    }
    stats
}

fn @DISPATCH@(data: &[u8], seed: u64, start: usize, sink: &mut @SINK@) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::@INDEX@(data, seed, start, sink) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::@INDEX@(data, seed, start, sink) };
        return;
    }
    unsupported_cpu();
}

pub(crate) struct @SINK@<'a> {
    data: &'a [u8],
    stats: @STRUCT@,
    field_start: u32,
    record_start: u32,
    ordinal: u32,
    pending: [(u32, u32); @PENDING_COUNT@],
    found: u32,
    end: u32,
    emitting: bool,
    pub(crate) done: bool,
}

impl<'a> @SINK@<'a> {
    fn new(data: &'a [u8], start: u32, end: u32, emitting: bool) -> Self {
        Self {
            data,
            stats: @STRUCT@::default(),
            field_start: start,
            record_start: start,
            ordinal: 0,
            pending: [(0, 0); @PENDING_COUNT@],
            found: 0,
            end,
            emitting,
            done: false,
        }
    }

    #[inline]
    pub(crate) fn drive(&mut self, mask: u64, term: u64, base: u32) {
        let mut m = mask;
        while m != 0 {
            let bit = m & m.wrapping_neg();
            let p = base + m.trailing_zeros();
            if term & bit != 0 {
                if self.emitting {
                    self.flush(p);
                } else {
                    self.emitting = true;
                }
                self.record_start = p + 1;
                self.field_start = p + 1;
                self.ordinal = 0;
                self.found = 0;
                if p >= self.end {
                    self.done = true;
                    return;
                }
            } else if self.emitting {
                match self.ordinal {
@TEXT_DRIVE_ARM@                    5 => {
                        self.pending[@LAT_IDX@] = (self.field_start, p);
                        self.found |= @LAT_BIT@;
                    }
                    6 => {
                        self.pending[@LON_IDX@] = (self.field_start, p);
                        self.found |= @LON_BIT@;
                    }
                    _ => {}
                }
                self.ordinal += 1;
                self.field_start = p + 1;
            }
            m &= m - 1;
        }
    }

    fn flush(&mut self, end: u32) {
        let data = self.data;
        let to = if end > self.record_start && data[end as usize - 1] == b'\r' {
            end - 1
        } else {
            end
        };
        match self.ordinal {
@TEXT_LAST_ARM@            5 => {
                self.pending[@LAT_IDX@] = (self.field_start, to);
                self.found |= @LAT_BIT@;
            }
            6 => {
                self.pending[@LON_IDX@] = (self.field_start, to);
                self.found |= @LON_BIT@;
            }
            _ => {}
        }

        self.stats.records += 1;
@TEXT_FLUSH_BODY@        let (from, to) = self.pending[@LAT_IDX@];
        if self.found & @LAT_BIT@ != 0
            && let Some(value) = parse_f64_field(&data[from as usize..to as usize])
        {
            self.stats.latitude_values += 1;
            self.stats.latitude_checksum = self.stats.latitude_checksum.wrapping_add(value.to_bits());
        }
        let (from, to) = self.pending[@LON_IDX@];
        if self.found & @LON_BIT@ != 0
            && let Some(value) = parse_f64_field(&data[from as usize..to as usize])
        {
            self.stats.longitude_values += 1;
            self.stats.longitude_checksum =
                self.stats.longitude_checksum.wrapping_add(value.to_bits());
        }
    }

    fn finish(mut self) -> @STRUCT@ {
        let len = self.data.len() as u32;
        if self.emitting && !self.done && self.record_start < len {
            self.flush(len);
        }
        self.stats
    }
}

impl @STRUCT@ {
    fn merge(&mut self, other: Self) {
        self.records += other.records;
@TEXT_MERGE_BODY@        self.latitude_values += other.latitude_values;
        self.longitude_values += other.longitude_values;
        self.latitude_checksum = self.latitude_checksum.wrapping_add(other.latitude_checksum);
        self.longitude_checksum = self.longitude_checksum.wrapping_add(other.longitude_checksum);
    }
}

@TEXT_HELPERS@"#
        .replace("@STRUCT@", struct_name)
        .replace("@SINK@", sink_name)
        .replace("@PARSE@", parse_fn)
        .replace("@PARSE_PAR@", parse_par_fn)
        .replace("@DISPATCH@", dispatch_fn)
        .replace("@INDEX@", index_fn)
        .replace("@THREAD_MSG@", thread_msg)
        .replace("@TEXT_STRUCT_FIELDS@", text_struct_fields)
        .replace("@PENDING_COUNT@", pending_count)
        .replace("@TEXT_DRIVE_ARM@", text_drive_arm)
        .replace("@TEXT_LAST_ARM@", text_last_arm)
        .replace("@TEXT_FLUSH_BODY@", text_flush_body)
        .replace("@TEXT_MERGE_BODY@", text_merge_body)
        .replace("@TEXT_HELPERS@", text_helpers)
        .replace("@LAT_IDX@", lat_idx)
        .replace("@LAT_BIT@", lat_bit)
        .replace("@LON_IDX@", lon_idx)
        .replace("@LON_BIT@", lon_bit);

    code.push_str(&api);
}

fn push_vcf_stats_api(code: &mut String) {
    code.push_str(
        r#"
/// VCF typed projection summary emitted by [`parse_vcf_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VcfStats {
    pub records: u64,
    pub primary_bytes: u64,
    pub pos_values: u64,
    pub checksum: u64,
}

/// Parse VCF records and accumulate the benchmark projection directly.
///
/// This generated domain API uses the same SIMD structural stream as
/// [`parse_columns`], but avoids materializing column vectors that the
/// sustained benchmark immediately walks again.
pub fn parse_vcf_stats(data: &[u8]) -> VcfStats {
    let mut sink = VcfStatsSink::new(data, 0, data.len() as u32, true);
    index_vcf_stats_dispatch(data, 0, 0, &mut sink);
    sink.finish()
}

/// Parallel [`parse_vcf_stats`] using the same record-ownership contract as
/// [`parse_columns_par`].
pub fn parse_vcf_stats_par(data: &[u8], threads: usize) -> VcfStats {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        return parse_vcf_stats(data);
    }
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| {
            if t == threads {
                data.len()
            } else {
                (t * chunk).min(data.len())
            }
        })
        .collect();
    let parts: Vec<VcfStats> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (start, end) = (bounds[t], bounds[t + 1]);
                s.spawn(move || {
                    let mut sink = VcfStatsSink::new(data, start as u32, end as u32, t == 0);
                    index_vcf_stats_dispatch(data, 0, start, &mut sink);
                    sink.finish()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("vcf stats thread ok"))
            .collect()
    });
    let mut stats = VcfStats::default();
    for part in parts {
        stats.merge(part);
    }
    stats
}

fn index_vcf_stats_dispatch(data: &[u8], seed: u64, start: usize, sink: &mut VcfStatsSink) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::index_vcf_stats(data, seed, start, sink) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_vcf_stats(data, seed, start, sink) };
        return;
    }
    unsupported_cpu();
}

pub(crate) struct VcfStatsSink<'a> {
    data: &'a [u8],
    stats: VcfStats,
    field_start: u32,
    record_start: u32,
    ordinal: u32,
    pending: [(u32, u32); 4],
    found: u32,
    end: u32,
    emitting: bool,
    pub(crate) done: bool,
}

impl<'a> VcfStatsSink<'a> {
    fn new(data: &'a [u8], start: u32, end: u32, emitting: bool) -> Self {
        Self {
            data,
            stats: VcfStats::default(),
            field_start: start,
            record_start: start,
            ordinal: 0,
            pending: [(0, 0); 4],
            found: 0,
            end,
            emitting,
            done: false,
        }
    }

    #[inline]
    pub(crate) fn drive(&mut self, mask: u64, term: u64, base: u32) {
        let mut m = mask;
        while m != 0 {
            let bit = m & m.wrapping_neg();
            let p = base + m.trailing_zeros();
            if term & bit != 0 {
                if self.emitting {
                    self.flush(p);
                } else {
                    self.emitting = true;
                }
                self.record_start = p + 1;
                self.field_start = p + 1;
                self.ordinal = 0;
                self.found = 0;
                if p >= self.end {
                    self.done = true;
                    return;
                }
            } else if self.emitting {
                match self.ordinal {
                    1 => {
                        self.pending[0] = (self.field_start, p);
                        self.found |= 1;
                    }
                    3 => {
                        self.pending[1] = (self.field_start, p);
                        self.found |= 2;
                    }
                    4 => {
                        self.pending[2] = (self.field_start, p);
                        self.found |= 4;
                    }
                    5 => {
                        self.pending[3] = (self.field_start, p);
                        self.found |= 8;
                    }
                    _ => {}
                }
                self.ordinal += 1;
                self.field_start = p + 1;
            }
            m &= m - 1;
        }
    }

    fn flush(&mut self, end: u32) {
        let data = self.data;
        if end > self.record_start && data[self.record_start as usize] == b'#' {
            return;
        }
        let to = if end > self.record_start && data[end as usize - 1] == b'\r' {
            end - 1
        } else {
            end
        };
        match self.ordinal {
            1 => {
                self.pending[0] = (self.field_start, to);
                self.found |= 1;
            }
            3 => {
                self.pending[1] = (self.field_start, to);
                self.found |= 2;
            }
            4 => {
                self.pending[2] = (self.field_start, to);
                self.found |= 4;
            }
            5 => {
                self.pending[3] = (self.field_start, to);
                self.found |= 8;
            }
            _ => {}
        }

        self.stats.records += 1;
        let (from, to) = self.pending[0];
        if self.found & 1 != 0
            && let Some(pos) = parse_vcf_pos_cell(&data[from as usize..to as usize])
        {
            self.stats.pos_values += 1;
            self.stats.checksum = vcf_checksum_u64(self.stats.checksum, pos as u64);
        }
        let (from, to) = self.pending[1];
        if self.found & 2 != 0 && from != to {
            let bytes = &data[from as usize..to as usize];
            self.stats.primary_bytes += bytes.len() as u64;
            self.stats.checksum = vcf_checksum_primary_bytes(self.stats.checksum, bytes);
        }
        let (from, to) = self.pending[2];
        if self.found & 4 != 0 && from != to {
            let bytes = &data[from as usize..to as usize];
            self.stats.primary_bytes += bytes.len() as u64;
            self.stats.checksum = vcf_checksum_primary_bytes(self.stats.checksum, bytes);
        }
        let (from, to) = self.pending[3];
        if self.found & 8 != 0
            && let Some(quality) = parse_f64_cell(&data[from as usize..to as usize])
        {
            self.stats.checksum =
                vcf_checksum_u64(self.stats.checksum, (quality as f32).to_bits() as u64);
        }
    }

    fn finish(mut self) -> VcfStats {
        let len = self.data.len() as u32;
        if self.emitting && !self.done && self.record_start < len {
            self.flush(len);
        }
        self.stats
    }
}

#[inline(always)]
fn parse_vcf_pos_cell(s: &[u8]) -> Option<i64> {
    match s.len() {
        8 => {
            let word = u64::from_le_bytes(s.try_into().unwrap());
            is_8_digits(word).then(|| parse_8_digits(word) as i64)
        }
        9 => {
            let first = s[0].wrapping_sub(b'0');
            if first > 9 {
                return None;
            }
            let tail = u64::from_le_bytes(s[1..9].try_into().unwrap());
            is_8_digits(tail).then(|| first as i64 * 100_000_000 + parse_8_digits(tail) as i64)
        }
        1..=7 => {
            let mut value = 0i64;
            for &byte in s {
                let digit = byte.wrapping_sub(b'0');
                if digit > 9 {
                    return None;
                }
                value = value * 10 + digit as i64;
            }
            Some(value)
        }
        _ => parse_i64_cell(s),
    }
}

impl VcfStats {
    fn merge(&mut self, other: Self) {
        self.records += other.records;
        self.primary_bytes += other.primary_bytes;
        self.pos_values += other.pos_values;
        self.checksum = self.checksum.wrapping_add(other.checksum);
    }
}

fn vcf_checksum_primary_bytes(checksum: u64, bytes: &[u8]) -> u64 {
    match bytes {
        [byte] => checksum.wrapping_add(*byte as u64),
        _ => vcf_checksum_bytes(checksum, bytes),
    }
}

fn vcf_checksum_bytes(mut checksum: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        let pairs = (word & 0x00ff_00ff_00ff_00ff) + ((word >> 8) & 0x00ff_00ff_00ff_00ff);
        let quads = (pairs & 0x0000_ffff_0000_ffff) + ((pairs >> 16) & 0x0000_ffff_0000_ffff);
        checksum = checksum.wrapping_add((quads & 0x0000_0000_ffff_ffff) + (quads >> 32));
    }
    for &byte in chunks.remainder() {
        checksum = checksum.wrapping_add(byte as u64);
    }
    checksum
}

fn vcf_checksum_u64(checksum: u64, value: u64) -> u64 {
    vcf_checksum_bytes(checksum, &value.to_le_bytes())
}

"#,
    );
}

/// Integer cell parser template: SWAR 8-digit blocks (Lemire) for the
/// common lengths, checked scalar arithmetic for 17+ digit tails.
const INT_CELL_TPL: &str = r#"/// Parse an integer cell with the exact acceptance rules of
/// `str::parse::<i64>`: optional sign, ASCII digits, nothing else.
///
/// Cells of up to 16 digits (the overwhelming majority) are parsed as two
/// SWAR 8-digit blocks; longer cells fall back to checked scalar
/// arithmetic, which also rejects overflow exactly like `str::parse`.
fn parse_i64_cell(s: &[u8]) -> Option<i64> {
    let (neg, digits) = match s.split_first() {
        Some((&b'-', rest)) => (true, rest),
        Some((&b'+', rest)) => (false, rest),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    if digits.len() <= 16 {
        // Right-align into a '0'-padded buffer: leading ASCII zeros are
        // digits that contribute nothing, so two fixed 8-digit blocks
        // cover every length without branching on it.
        let mut buf = [b'0'; 16];
        buf[16 - digits.len()..].copy_from_slice(digits);
        let hi = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let lo = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        if !is_8_digits(hi) || !is_8_digits(lo) {
            return None;
        }
        // <= 9_999_999_999_999_999 < i64::MAX: no overflow possible.
        let value = (parse_8_digits(hi) * 100_000_000 + parse_8_digits(lo)) as i64;
        return Some(if neg { -value } else { value });
    }
    // 17+ digits: rare; checked scalar arithmetic, accumulating negative
    // so i64::MIN parses.
    let mut value: i64 = 0;
    for &b in digits {
        if !b.is_ascii_digit() {
            return None;
        }
        value = value
            .checked_mul(10)?
            .checked_sub((b - b'0') as i64)?;
    }
    if neg { Some(value) } else { value.checked_neg() }
}

/// True iff all eight bytes are ASCII digits: high nibbles must be 3, and
/// adding 6 must not carry into the high nibble (i.e. low nibble <= 9).
#[inline]
fn is_8_digits(v: u64) -> bool {
    const HI: u64 = 0xF0F0_F0F0_F0F0_F0F0;
    const THREES: u64 = 0x3030_3030_3030_3030;
    v & HI == THREES && v.wrapping_add(0x0606_0606_0606_0606) & HI == THREES
}

/// Combine eight ASCII digits (loaded little-endian, so the first
/// character is the low byte) into their value: three multiply-and-shift
/// rounds, each merging adjacent digit groups (Lemire's SWAR atoi).
#[inline]
fn parse_8_digits(v: u64) -> u64 {
    let v = (v & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
    let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
    (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32
}

"#;

/// Float cell parser template: Clinger fast path + `str::parse` fallback.
const FLOAT_CELL_TPL: &str = r#"/// Parse a float cell with the exact semantics of `str::parse::<f64>`.
///
/// Fast path (Clinger 1990): when the decimal mantissa has <= 15 digits
/// (so it is exact in an f64) and the decimal exponent is within +/-22
/// (so 10^|e| is exact in an f64), one multiply or divide performs a
/// single correct rounding -- bit-identical to a full parser. Everything
/// else (long mantissas, big exponents, inf/nan spellings, malformed
/// cells) falls through to `str::parse`, which has been Eisel-Lemire in
/// std since Rust 1.55 -- the fallback is rarely taken, not slow.
#[inline(always)]
fn parse_f64_cell(s: &[u8]) -> Option<f64> {
    const POW10: [f64; 23] = [
        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
        1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
    ];
    if let Some((neg, mantissa)) = parse_fixed_6_decimal_signed(s) {
        let value = mantissa as f64 / 1_000_000.0;
        return Some(if neg { -value } else { value });
    }
    let (neg, rest) = match s.split_first() {
        Some((&b'-', rest)) => (true, rest),
        Some((&b'+', rest)) => (false, rest),
        _ => (false, s),
    };
    let mut i = 0;
    let mut mantissa: u64 = 0;
    let mut digits = 0usize;
    while i < rest.len() && rest[i].is_ascii_digit() {
        mantissa = mantissa.wrapping_mul(10).wrapping_add((rest[i] - b'0') as u64);
        digits += 1;
        i += 1;
    }
    let mut exp10: i32 = 0;
    if i < rest.len() && rest[i] == b'.' {
        i += 1;
        while i < rest.len() && rest[i].is_ascii_digit() {
            mantissa = mantissa.wrapping_mul(10).wrapping_add((rest[i] - b'0') as u64);
            digits += 1;
            exp10 -= 1;
            i += 1;
        }
    }
    if i < rest.len() && (rest[i] | 0x20) == b'e' {
        i += 1;
        let esign = match rest.get(i) {
            Some(&b'-') => {
                i += 1;
                -1
            }
            Some(&b'+') => {
                i += 1;
                1
            }
            _ => 1,
        };
        let mut e: i32 = 0;
        let mut exp_digits = 0;
        while i < rest.len() && rest[i].is_ascii_digit() {
            e = e.saturating_mul(10).saturating_add((rest[i] - b'0') as i32);
            exp_digits += 1;
            i += 1;
        }
        if exp_digits == 0 {
            return parse_f64_fallback(s);
        }
        exp10 += esign * e.min(100_000);
    }
    // Anything the strict scanner did not consume -- no digits at all
    // ("inf", "nan", "."), trailing bytes -- goes to the full parser.
    if digits == 0 || i != rest.len() {
        return parse_f64_fallback(s);
    }
    if digits > 15 || !(-22..=22).contains(&exp10) {
        return parse_f64_fallback(s);
    }
    let value = if exp10 < 0 {
        mantissa as f64 / POW10[(-exp10) as usize]
    } else {
        mantissa as f64 * POW10[exp10 as usize]
    };
    Some(if neg { -value } else { value })
}

#[cold]
fn parse_f64_fallback(s: &[u8]) -> Option<f64> {
    std::str::from_utf8(s).ok()?.parse::<f64>().ok()
}

#[inline]
fn parse_fixed_6_decimal_signed(s: &[u8]) -> Option<(bool, u64)> {
    match s.len() {
        8 if s[1] == b'.' => {
            let mantissa = parse_decimal_digit(s[0])? * 1_000_000 + parse_6_digits(&s[2..8])?;
            Some((false, mantissa))
        }
        9 if s[2] == b'.' => {
            let (neg, whole) = if s[0] == b'-' {
                (true, parse_decimal_digit(s[1])?)
            } else {
                (
                    false,
                    parse_decimal_digit(s[0])? * 10 + parse_decimal_digit(s[1])?,
                )
            };
            Some((neg, whole * 1_000_000 + parse_6_digits(&s[3..9])?))
        }
        10 if s[3] == b'.' => {
            let (neg, whole) = if s[0] == b'-' {
                (
                    true,
                    parse_decimal_digit(s[1])? * 10 + parse_decimal_digit(s[2])?,
                )
            } else {
                (
                    false,
                    parse_decimal_digit(s[0])? * 100
                        + parse_decimal_digit(s[1])? * 10
                        + parse_decimal_digit(s[2])?,
                )
            };
            Some((neg, whole * 1_000_000 + parse_6_digits(&s[4..10])?))
        }
        11 if s[0] == b'-' && s[4] == b'.' => {
            let whole = parse_decimal_digit(s[1])? * 100
                + parse_decimal_digit(s[2])? * 10
                + parse_decimal_digit(s[3])?;
            Some((true, whole * 1_000_000 + parse_6_digits(&s[5..11])?))
        }
        _ => None,
    }
}

#[inline]
fn parse_6_digits(s: &[u8]) -> Option<u64> {
    debug_assert_eq!(s.len(), 6);
    Some(
        parse_decimal_digit(s[0])? * 100_000
            + parse_decimal_digit(s[1])? * 10_000
            + parse_decimal_digit(s[2])? * 1_000
            + parse_decimal_digit(s[3])? * 100
            + parse_decimal_digit(s[4])? * 10
            + parse_decimal_digit(s[5])?,
    )
}

#[inline]
fn parse_decimal_digit(b: u8) -> Option<u64> {
    let digit = b.wrapping_sub(b'0');
    if digit <= 9 {
        Some(digit as u64)
    } else {
        None
    }
}

"#;

/// The sequential region resolver emitted into both kernels when the
/// graph contains a `Regions` node. Plain u64 bit math — the same text
/// serves the scalar and AVX2 modules.
const REGIONS_HELPER: &str = r#"
    /// Three-state (normal/quote/comment) region resolution: walks the set
    /// bits of its inputs in position order, filling the inert mask between
    /// region open and close events. Quote bits are ignored inside comments
    /// and comment candidates inside quotes — the interleaving bit-parallel
    /// parity cannot express. `state` carries the region across blocks.
    ///
    /// `pclmulqdq` is required for the carry-less-multiply prefix-XOR on the
    /// comment-free fast path; the SIMD modules that emit this helper already
    /// enable it, and `step` (a feature superset) is the only caller.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    fn resolve_regions(q: u64, s: u64, n: u64, state: &mut u64) -> u64 {
        const NORMAL: u64 = 0;
        const QUOTE: u64 = 1;
        const COMMENT: u64 = 2;
        // Fast path: outside any region with nothing opening this block, the
        // only events are newlines, which are inert here — so there is no work
        // and the state is unchanged. This is the overwhelming majority of
        // blocks for comment dialects whose comments cluster (VCF/BED/SAM
        // headers, etc.), turning per-block region resolution into a no-op.
        if *state == NORMAL && (q | s) == 0 {
            return 0;
        }
        // Fast path: no comment can be active in this block — none starts here
        // (`s == 0`) and we did not enter mid-comment — so the three-state
        // machine collapses to plain quote toggling, which is exactly a
        // prefix-XOR of the quote bits seeded by the entry state. Newlines and
        // stray `#` are irrelevant outside a comment. This keeps the common
        // quoted body (comments cluster in the header) on a branchless path
        // instead of the per-event walk below; it is bit-identical to that walk
        // for every block satisfying this guard (proven by the differential
        // tests, which route interleaved blocks to the walk). The prefix-XOR is
        // one PCLMULQDQ carry-less multiply (`prefix_xor`) rather than the
        // scalar shift cascade — shorter latency on the block-to-block region
        // state dependency chain.
        if *state != COMMENT && s == 0 {
            let mut inert = prefix_xor(q);
            if *state == QUOTE {
                inert = !inert;
            }
            // MSB set iff still inside a quoted region at the block boundary.
            *state = inert >> 63;
            return inert;
        }
        let mut inert = 0u64;
        // A region continuing from the previous block fills from bit 0.
        let mut run_start = 0u32;
        let mut events = q | s | n;
        while events != 0 {
            let p = events.trailing_zeros();
            let bit = 1u64 << p;
            match *state {
                QUOTE => {
                    if q & bit != 0 {
                        inert |= range_mask(run_start, p);
                        *state = NORMAL;
                    }
                }
                COMMENT => {
                    if n & bit != 0 {
                        inert |= range_mask(run_start, p);
                        *state = NORMAL;
                    }
                }
                _ => {
                    if q & bit != 0 {
                        *state = QUOTE;
                        run_start = p;
                    } else if s & bit != 0 {
                        *state = COMMENT;
                        run_start = p;
                    }
                }
            }
            events &= events - 1;
        }
        if *state != NORMAL {
            inert |= range_mask(run_start, 64);
        }
        inert
    }

    /// Bits `[from, to)` set.
    #[inline]
    fn range_mask(from: u32, to: u32) -> u64 {
        let hi = if to >= 64 { !0u64 } else { (1u64 << to) - 1 };
        hi & !((1u64 << from) - 1)
    }
"#;

/// Streaming parser template; `@K@` is the kernel carry count.
const STREAM_TPL: &str = r#"/// Incremental parser for unbounded input: feed chunks, receive complete
/// records via callback. Kernel carries persist across feeds, so quoted
/// regions and escape runs spanning chunk boundaries are handled exactly.
/// Bytes of the unfinished trailing record are buffered internally and
/// compacted amortized; a single record must fit in memory (< 4 GiB).
pub struct StreamParser {
    buf: Vec<u8>,
    seps: Vec<u32>,
    ends: Vec<u64>,
    indexed: usize,
    emitted: usize,
    emitted_seps: usize,
    record_start: usize,
    carries: [u64; @K@],
}

/// Create a [`StreamParser`].
pub fn stream() -> StreamParser {
    StreamParser {
        buf: Vec::new(),
        seps: Vec::new(),
        ends: Vec::new(),
        indexed: 0,
        emitted: 0,
        emitted_seps: 0,
        record_start: 0,
        carries: @CINIT@,
    }
}

impl StreamParser {
    /// Feed the next chunk; `on_record` is called once per record completed
    /// by this chunk.
    pub fn feed(&mut self, chunk: &[u8], mut on_record: impl FnMut(Record<'_>)) {
        self.buf.extend_from_slice(chunk);
        index_tape_partial_dispatch(
            &self.buf[self.indexed..],
            &mut self.carries,
            self.indexed as u32,
            &mut self.seps,
            &mut self.ends,
        );
        self.indexed += (self.buf.len() - self.indexed) & !63;
        self.emit_ready(&mut on_record);
        self.compact();
    }

    /// Signal end of input; emits any records completed by the final
    /// partial block plus a trailing unterminated record.
    pub fn finish(mut self, mut on_record: impl FnMut(Record<'_>)) {
        let rem = self.buf.len() - self.indexed;
        if rem > 0 {
            // True end of stream: zero padding is safe here, exactly as in
            // the batch tail.
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&self.buf[self.indexed..]);
            let live = (1u64 << rem) - 1;
            index_tape_block_dispatch(
                &block,
                live,
                &mut self.carries,
                self.indexed as u32,
                &mut self.seps,
                &mut self.ends,
            );
        }
        self.emit_ready(&mut on_record);
        if self.record_start < self.buf.len() {
            on_record(Record {
                data: &self.buf,
                start: self.record_start,
                end: self.buf.len(),
                seps: &self.seps[self.emitted_seps..],
            });
        }
    }

    fn emit_ready(&mut self, on_record: &mut impl FnMut(Record<'_>)) {
        while self.emitted < self.ends.len() {
            let entry = self.ends[self.emitted];
            let end = (entry & 0xFFFF_FFFF) as usize;
            let cum = (entry >> 32) as usize;
            on_record(Record {
                data: &self.buf,
                start: self.record_start,
                end,
                seps: &self.seps[self.emitted_seps..cum],
            });
            self.record_start = end + 1;
            self.emitted_seps = cum;
            self.emitted += 1;
        }
    }

    /// Drop consumed bytes/tape once they dominate the buffer. The cut is
    /// 64-byte aligned: it keeps block boundaries and, critically, byte
    /// position parity stable (the escape machinery's even/odd constant
    /// masks are parity-dependent).
    fn compact(&mut self) {
        let base = self.record_start.min(self.indexed) & !63;
        if base < 4096 || base < self.buf.len() / 2 {
            return;
        }
        self.buf.copy_within(base.., 0);
        self.buf.truncate(self.buf.len() - base);
        self.indexed -= base;
        self.seps.drain(..self.emitted_seps);
        for p in &mut self.seps {
            *p -= base as u32;
        }
        let rebase = ((self.emitted_seps as u64) << 32) | base as u64;
        self.ends.drain(..self.emitted);
        for e in &mut self.ends {
            *e -= rebase;
        }
        self.record_start -= base;
        self.emitted = 0;
        self.emitted_seps = 0;
    }
}

fn index_tape_partial_dispatch(data: &[u8], carries: &mut [u64; @K@], base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::index_tape_partial(data, carries, base, seps, ends) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_tape_partial(data, carries, base, seps, ends) };
        return;
    }
    unsupported_cpu();
}

fn index_tape_block_dispatch(block: &[u8; 64], live: u64, carries: &mut [u64; @K@], base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::index_tape_block(block, live, carries, base, seps, ends) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_tape_block(block, live, carries, base, seps, ends) };
        return;
    }
    unsupported_cpu();
}

"#;

/// The dialect-specific appending cleaner for string columns: same
/// semantics as `clean`, but writing into the column's data buffer so the
/// unquoted common case is a single `extend_from_slice` and no
/// intermediate allocation ever happens.
fn append_clean_fn(dialect: &crate::formats::Dialect) -> String {
    use crate::formats::Escape;
    match (dialect.quote, dialect.escape) {
        (None, _) => r#"/// No quote convention in this dialect: cells append verbatim.
#[inline]
fn append_clean(out: &mut Vec<u8>, raw: &[u8]) {
    out.extend_from_slice(raw);
}

"#
        .to_string(),
        (Some(q), Escape::None) => format!(
            r#"/// Append `raw` cleaned: outer quotes stripped, doubled quotes
/// collapsed. Unquoted cells are one memcpy.
fn append_clean(out: &mut Vec<u8>, raw: &[u8]) {{
    const Q: u8 = {q}u8;
    if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {{
        let inner = &raw[1..raw.len() - 1];
        let mut i = 0;
        while i < inner.len() {{
            out.push(inner[i]);
            if inner[i] == Q && i + 1 < inner.len() && inner[i + 1] == Q {{
                i += 2;
            }} else {{
                i += 1;
            }}
        }}
    }} else {{
        out.extend_from_slice(raw);
    }}
}}

"#
        ),
        (Some(q), Escape::Backslash(e)) => format!(
            r#"/// Append `raw` cleaned: outer quotes stripped, backslash escapes
/// resolved (`\x` -> `x`). Escape-free cells are one memcpy.
fn append_clean(out: &mut Vec<u8>, raw: &[u8]) {{
    const Q: u8 = {q}u8;
    const E: u8 = {e}u8;
    let body = if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {{
        &raw[1..raw.len() - 1]
    }} else {{
        raw
    }};
    if !body.contains(&E) {{
        out.extend_from_slice(body);
        return;
    }}
    let mut i = 0;
    while i < body.len() {{
        if body[i] == E && i + 1 < body.len() {{
            out.push(body[i + 1]);
            i += 2;
        }} else {{
            out.push(body[i]);
            i += 1;
        }}
    }}
}}

"#
        ),
    }
}

/// The dialect-specific field-cleaning function for the span API.
fn clean_fn(dialect: &crate::formats::Dialect) -> String {
    use crate::formats::Escape;
    let swar = SWAR_CONTAINS;
    match (dialect.quote, dialect.escape) {
        (None, _) => r#"/// No quote convention in this dialect: fields are returned verbatim.
#[inline]
fn clean(raw: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    std::borrow::Cow::Borrowed(raw)
}
"#
        .to_string(),
        (Some(q), Escape::None) => format!(
            r#"/// Strip surrounding quotes and collapse doubled escape quotes. In a
/// valid doubled-quote field every interior quote is half of a pair, so
/// "contains the quote byte" is the collapse test; a malformed stray
/// quote merely takes the copying path and comes out byte-identical.
///
#[inline]
fn clean(raw: &[u8]) -> std::borrow::Cow<'_, [u8]> {{
    const Q: u8 = {q}u8;
    if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {{
        let inner = &raw[1..raw.len() - 1];
        if !contains_byte(inner, Q) {{
            return std::borrow::Cow::Borrowed(inner);
        }}
        let mut out = Vec::with_capacity(inner.len());
        let mut i = 0;
        while i < inner.len() {{
            out.push(inner[i]);
            if inner[i] == Q && i + 1 < inner.len() && inner[i + 1] == Q {{
                i += 2;
            }} else {{
                i += 1;
            }}
        }}
        return std::borrow::Cow::Owned(out);
    }}
    std::borrow::Cow::Borrowed(raw)
}}
{swar}"#
        ),
        (Some(q), Escape::Backslash(e)) => format!(
            r#"/// Strip surrounding quotes and resolve backslash escapes (`\x` -> `x`).
#[inline]
fn clean(raw: &[u8]) -> std::borrow::Cow<'_, [u8]> {{
    const Q: u8 = {q}u8;
    const E: u8 = {e}u8;
    let body = if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {{
        &raw[1..raw.len() - 1]
    }} else {{
        raw
    }};
    if !contains_byte(body, E) {{
        return std::borrow::Cow::Borrowed(body);
    }}
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {{
        if body[i] == E && i + 1 < body.len() {{
            out.push(body[i + 1]);
            i += 2;
        }} else {{
            out.push(body[i]);
            i += 1;
        }}
    }}
    std::borrow::Cow::Owned(out)
}}
{swar}"#
        ),
    }
}

/// SWAR byte search emitted next to `clean` (std-only memchr): the
/// has-zero-byte trick on XORed 8-byte chunks, byte-order agnostic.
const SWAR_CONTAINS: &str = r#"
/// SWAR byte search, 8 bytes per step (std-only stand-in for memchr).
#[inline]
fn contains_byte(hay: &[u8], needle: u8) -> bool {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let pat = (needle as u64).wrapping_mul(LO);
    let mut chunks = hay.chunks_exact(8);
    for chunk in &mut chunks {
        let x = u64::from_le_bytes(chunk.try_into().unwrap()) ^ pat;
        if x.wrapping_sub(LO) & !x & HI != 0 {
            return true;
        }
    }
    chunks.remainder().contains(&needle)
}
"#;

/// Nodes reachable from `roots` through operand edges: a step variant only
/// emits the lines its return tuple needs, keeping every variant
/// warning-free even though variants share one graph.
fn live_nodes(graph: &Graph, roots: &[crate::ir::NodeId]) -> Vec<bool> {
    let mut live = vec![false; graph.nodes().len()];
    let mut work: Vec<u32> = roots.iter().map(|r| r.0).collect();
    while let Some(i) = work.pop() {
        if live[i as usize] {
            continue;
        }
        live[i as usize] = true;
        match graph.nodes()[i as usize] {
            Op::Class(_) | Op::Const(_) => {}
            Op::Not(a) | Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) | Op::PrefixXor(a) => {
                work.push(a.0)
            }
            Op::And(a, b) | Op::Or(a, b) | Op::Xor(a, b) | Op::Add(a, b) => {
                work.push(a.0);
                work.push(b.0);
            }
            Op::Regions(a, b, c) => {
                work.push(a.0);
                work.push(b.0);
                work.push(c.0);
            }
        }
    }
    live
}

/// Emit one `let vN = ...;` line per node reachable from `roots`. Shared
/// between kernels except for the `Class` byte-comparison primitive.
fn emit_step_body(
    code: &mut String,
    graph: &Graph,
    carry_slot: &[usize],
    flavor: Flavor,
    roots: &[crate::ir::NodeId],
) {
    let live = live_nodes(graph, roots);
    let class_args = match flavor {
        Flavor::Avx512 => "lo, hi",
        Flavor::Avx2 => "lo, hi",
        Flavor::Neon => "b0, b1, b2, b3",
    };
    for (i, op) in graph.nodes().iter().enumerate() {
        if !live[i] {
            continue;
        }
        let line = match *op {
            Op::Class(class) => {
                let n = class.members().count();
                let label: String = if n > 16 {
                    format!("{n} bytes")
                } else {
                    class
                        .members()
                        .map(|b| b.escape_ascii().to_string())
                        .collect()
                };
                if n <= MAX_CLASS_BYTES || flavor == Flavor::Avx512 {
                    let compares: Vec<String> = class
                        .members()
                        .map(|b| format!("eq_mask({class_args}, {b}u8)"))
                        .collect();
                    format!("let v{i} = {}; // class \"{label}\"", compares.join(" | "))
                } else {
                    match flavor {
                        Flavor::Avx2 => {
                            let (lo_tbl, hi_tbl) =
                                nibble_tables(&class).expect("validated before emission");
                            let setr = |t: [u8; 16]| -> String {
                                t.iter()
                                    .chain(t.iter())
                                    .map(|&v| format!("{v}u8 as i8"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            };
                            format!(
                                "let v{i} = {{ let lo_tbl = _mm256_setr_epi8({}); let hi_tbl = _mm256_setr_epi8({}); table_mask(lo, hi, lo_tbl, hi_tbl) }}; // class \"{label}\"",
                                setr(lo_tbl),
                                setr(hi_tbl)
                            )
                        }
                        Flavor::Neon => {
                            let (lo_tbl, hi_tbl) =
                                nibble_tables(&class).expect("validated before emission");
                            // NEON has no `setr`; pack each 16-byte table into
                            // two little-endian u64 halves and rebuild it
                            // branchlessly via vcreate/vcombine (no memory load).
                            let dup = |t: [u8; 16]| -> String {
                                let half = |bytes: &[u8]| -> u64 {
                                    bytes
                                        .iter()
                                        .enumerate()
                                        .fold(0u64, |acc, (k, &v)| acc | ((v as u64) << (8 * k)))
                                };
                                format!(
                                    "vreinterpretq_u8_u64(vcombine_u64(vcreate_u64({:#018x}u64), vcreate_u64({:#018x}u64)))",
                                    half(&t[..8]),
                                    half(&t[8..])
                                )
                            };
                            format!(
                                "let v{i} = {{ let lo_tbl = {}; let hi_tbl = {}; table_mask(b0, b1, b2, b3, lo_tbl, hi_tbl) }}; // class \"{label}\"",
                                dup(lo_tbl),
                                dup(hi_tbl)
                            )
                        }
                        Flavor::Avx512 => unreachable!("AVX-512 emits class compares directly"),
                    }
                }
            }
            Op::Const(pattern) => format!("let v{i} = {pattern:#018x}u64;"),
            Op::Not(a) => format!("let v{i} = !v{};", a.0),
            Op::And(a, b) => format!("let v{i} = v{} & v{};", a.0, b.0),
            Op::Or(a, b) => format!("let v{i} = v{} | v{};", a.0, b.0),
            Op::Xor(a, b) => format!("let v{i} = v{} ^ v{};", a.0, b.0),
            Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) => {
                let k = carry_slot[i];
                format!(
                    "let v{i} = {{ let shifted = (v{a} << 1) | carries[{k}]; \
                     carries[{k}] = v{a} >> 63; shifted }};",
                    a = a.0
                )
            }
            Op::Regions(q, s, n) => {
                let k = carry_slot[i];
                format!(
                    "let v{i} = resolve_regions(v{q}, v{s}, v{n}, &mut carries[{k}]);",
                    q = q.0,
                    s = s.0,
                    n = n.0
                )
            }
            Op::PrefixXor(a) => {
                let k = carry_slot[i];
                format!(
                    "let v{i} = {{ let parity = prefix_xor(v{a}) ^ carries[{k}]; \
                     carries[{k}] = ((parity as i64) >> 63) as u64; parity }};",
                    a = a.0
                )
            }
            Op::Add(a, b) => {
                let k = carry_slot[i];
                format!(
                    "let v{i} = {{ let (partial, c1) = v{a}.overflowing_add(v{b}); \
                     let (sum, c2) = partial.overflowing_add(carries[{k}]); \
                     carries[{k}] = (c1 | c2) as u64; sum }};",
                    a = a.0,
                    b = b.0
                )
            }
        };
        let _ = writeln!(code, "        {line}");
    }
}

/// Emit the nested-tape API for dialects with bracket pairs: a tape builder
/// that matches brackets over the structural index (each bracket entry
/// carries the tape index of its partner, so skipping a container is O(1)),
/// plus item iterators for walking one nesting level at a time.
///
/// The template is brace-heavy Rust, so dialect-specific fragments are
/// substituted as placeholders rather than fighting `format!` escaping.
fn push_nested_api(code: &mut String, dialect: &crate::formats::Dialect, carry_count: usize) {
    fn byte_lit(b: u8) -> String {
        if b.is_ascii_graphic() && b != b'\'' && b != b'\\' {
            format!("b'{}'", b as char)
        } else {
            format!("0x{b:02x}")
        }
    }
    let open_pat = dialect
        .nesting
        .iter()
        .map(|&(open, _)| byte_lit(open))
        .collect::<Vec<_>>()
        .join(" | ");
    // One match arm per pair: pushing the expected close byte at open time
    // is what lets the pop validate without re-reading input or tape. The
    // catch-all is unreachable (the opens mask covers only open bytes) but
    // keeps the match exhaustive; 0 never equals a close byte.
    let expected_close_arms = dialect
        .nesting
        .iter()
        .map(|&(open, close)| {
            format!(
                "                    {} => {},\n",
                byte_lit(open),
                byte_lit(close)
            )
        })
        .chain(std::iter::once("                    _ => 0,\n".to_string()))
        .collect::<String>();

    const TEMPLATE: &str = r##"/// How bracket matching failed. Positions are byte offsets into the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestError {
    /// A close bracket with no matching open, or closing the wrong kind.
    UnmatchedClose(u32),
    /// Input ended with brackets still open; the innermost open's position.
    UnclosedOpen(u32),
}

/// The nested structural tape over borrowed input: one `u64` entry per
/// structural byte. The low 32 bits are the byte position; for bracket
/// entries the high 32 bits are the tape index of the matching partner
/// (`u32::MAX` while unclosed), making container skips O(1). Separator
/// entries leave the high bits zero and are classified by re-reading the
/// input byte, never by the partner field.
pub struct Nested<'a> {
    data: &'a [u8],
    tape: Vec<u64>,
    /// First bracket-matching error. Tape building stops at an unmatched
    /// close; navigation over an errored tape is best-effort, never panics.
    pub error: Option<NestError>,
}

/// Index `data` and match nesting brackets into a navigable tape.
/// Brackets inside quoted regions are inert and never reach the tape.
/// Fused: per-block masks stream from the kernel straight into the
/// bracket matcher; no intermediate position vector is materialized.
pub fn parse_nested(data: &[u8]) -> Nested<'_> {
    parse_nested_into(data, Nested { data: &[], tape: Vec::new(), error: None })
}

/// Like [`parse_nested`], recycling the tape allocation of a previous
/// parse (its contents are discarded). Steady-state callers avoid paying
/// the allocation — and, more important at GiB/s, the soft page faults of
/// a fresh tape — on every document batch.
pub fn parse_nested_into<'a>(data: &'a [u8], recycle: Nested<'_>) -> Nested<'a> {
    let mut tape = recycle.tape;
    tape.clear();
    let mut stack: Vec<u64> = Vec::with_capacity(64);
    let mut error = nested_tape(data, &mut tape, &mut stack);
    // Edition-agnostic single condition: generated files compile under the
    // consumer's edition, so no let-chains.
    if let (None, Some(&top)) = (error, stack.last()) {
        error = Some(NestError::UnclosedOpen(tape[(top >> 8) as usize] as u32));
    }
    Nested { data, tape, error }
}

fn nested_tape(data: &[u8], tape: &mut Vec<u64>, stack: &mut Vec<u64>) -> Option<NestError> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        return unsafe { avx512::nested_tape(data, tape, stack) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        return unsafe { avx2::nested_tape(data, tape, stack) };
    }
    unsupported_cpu()
}

/// Consume one block's structural masks in one ordered pass. The masks
/// classify each event without touching input bytes — separators (the
/// majority) are a mask test and a sequential store; only opens read their
/// byte, for pair identity. Stack entries pack (open's tape index << 8) |
/// expected close byte, so a pop validates with a single compare and no
/// dependent loads. Entries go through raw pointers after one reserve per
/// block; per-push capacity checks would otherwise dominate the event cost.
#[inline]
fn push_nested(
    data: &[u8],
    mut mask: u64,
    opens: u64,
    closes: u64,
    base: u32,
    tape: &mut Vec<u64>,
    stack: &mut Vec<u64>,
) -> Option<NestError> {
    let events = mask.count_ones() as usize;
    tape.reserve(events);
    stack.reserve(events);
    let tape_ptr = tape.as_mut_ptr();
    let stack_ptr = stack.as_mut_ptr();
    let mut tlen = tape.len();
    let mut slen = stack.len();
    let mut error = None;
    let brackets = opens | closes;
    while mask != 0 {
        let lowest = mask & mask.wrapping_neg();
        let pos = base + mask.trailing_zeros();
        mask &= mask - 1;
        if lowest & brackets == 0 {
            // SAFETY: tlen stays below the reserved bound (one entry per
            // event); same for every tape/stack write below.
            unsafe { *tape_ptr.add(tlen) = pos as u64 };
            tlen += 1;
        } else if lowest & opens != 0 {
            // SAFETY: a set structural bit always indexes a live input
            // byte (drivers mask off tail padding).
            unsafe {
                let close = match *data.get_unchecked(pos as usize) {
__EXPECTED_CLOSE_ARMS__                };
                *stack_ptr.add(slen) = ((tlen as u64) << 8) | close as u64;
                slen += 1;
                *tape_ptr.add(tlen) = ((u32::MAX as u64) << 32) | pos as u64;
            }
            tlen += 1;
        } else {
            if slen == 0 {
                error = Some(NestError::UnmatchedClose(pos));
                break;
            }
            slen -= 1;
            // SAFETY: slen indexes a live stack entry; the patched open
            // slot is an earlier, already-written tape entry.
            unsafe {
                let top = *stack_ptr.add(slen);
                if top as u8 != *data.get_unchecked(pos as usize) {
                    error = Some(NestError::UnmatchedClose(pos));
                    break;
                }
                let open = (top >> 8) as usize;
                *tape_ptr.add(open) =
                    ((tlen as u64) << 32) | (*tape_ptr.add(open) as u32) as u64;
                *tape_ptr.add(tlen) = ((open as u64) << 32) | pos as u64;
            }
            tlen += 1;
        }
    }
    // SAFETY: tlen/slen count initialized entries within reserved capacity.
    unsafe {
        tape.set_len(tlen);
        stack.set_len(slen);
    }
    error
}

/// Trim ASCII whitespace from both ends of `data[start..end]`, as offsets.
fn trim_ws(data: &[u8], mut start: usize, mut end: usize) -> (usize, usize) {
    while start < end && matches!(data[start], b' ' | b'\t' | b'\r' | b'\n') {
        start += 1;
    }
    while end > start && matches!(data[end - 1], b' ' | b'\t' | b'\r' | b'\n') {
        end -= 1;
    }
    (start, end)
}

impl<'a> Nested<'a> {
    /// Raw tape access; see the type docs for the entry layout.
    pub fn tape(&self) -> &[u64] {
        &self.tape
    }

    /// The top-level items: every bracketed container and every
    /// non-whitespace scalar run between top-level structural bytes.
    pub fn items(&self) -> Items<'a, '_> {
        Items {
            nested: self,
            next: 0,
            end: self.tape.len(),
            cursor: 0,
            limit: self.data.len(),
        }
    }

    /// Every scalar span in the document, flat and depth-independent, in input
    /// order: keys and values at any nesting level, whitespace-trimmed, with
    /// quotes and escapes intact (zero-copy). Aggregate queries — sum, count,
    /// search — that do not need the object/array structure run as a single
    /// O(tape) pass here, far cheaper than recursive [`Self::items`] descent:
    /// every scalar is the gap between consecutive structural-byte positions,
    /// which the tape already stores in ascending order.
    pub fn scalars(&self) -> Scalars<'a, '_> {
        Scalars {
            nested: self,
            next: 0,
            cursor: 0,
        }
    }
}

/// One value: a bracketed container or a whitespace-trimmed scalar span.
#[derive(Clone, Copy)]
pub struct Node<'a, 'p> {
    nested: &'p Nested<'a>,
    repr: Repr,
}

#[derive(Clone, Copy)]
enum Repr {
    /// Tape index of the opening bracket.
    Container(usize),
    /// Trimmed byte span.
    Scalar(usize, usize),
}

impl<'a, 'p> Node<'a, 'p> {
    /// The container's opening bracket byte; `None` for scalars.
    pub fn open(&self) -> Option<u8> {
        match self.repr {
            Repr::Container(i) => {
                Some(self.nested.data[(self.nested.tape[i] as u32) as usize])
            }
            Repr::Scalar(..) => None,
        }
    }

    /// The full input span: brackets included for containers, trimmed bytes
    /// for scalars (quotes and escapes intact — spans are zero-copy).
    pub fn bytes(&self) -> &'a [u8] {
        match self.repr {
            Repr::Container(i) => {
                let entry = self.nested.tape[i];
                let start = (entry as u32) as usize;
                let close = (entry >> 32) as u32;
                let end = if close == u32::MAX {
                    self.nested.data.len()
                } else {
                    (self.nested.tape[close as usize] as u32) as usize + 1
                };
                &self.nested.data[start..end]
            }
            Repr::Scalar(start, end) => &self.nested.data[start..end],
        }
    }

    /// The container's items in input order; empty for scalars. Every
    /// separator byte splits items, so formats that separate keys from
    /// values with a second separator (JSON objects' `:`) yield keys and
    /// values as consecutive items.
    pub fn items(&self) -> Items<'a, 'p> {
        match self.repr {
            Repr::Container(i) => {
                let entry = self.nested.tape[i];
                let close = (entry >> 32) as u32;
                let (end, limit) = if close == u32::MAX {
                    (self.nested.tape.len(), self.nested.data.len())
                } else {
                    (
                        close as usize,
                        (self.nested.tape[close as usize] as u32) as usize,
                    )
                };
                Items {
                    nested: self.nested,
                    next: i + 1,
                    end,
                    cursor: (entry as u32) as usize + 1,
                    limit,
                }
            }
            Repr::Scalar(..) => Items {
                nested: self.nested,
                next: 0,
                end: 0,
                cursor: 0,
                limit: 0,
            },
        }
    }
}

/// Iterator over the items of one nesting level.
pub struct Items<'a, 'p> {
    nested: &'p Nested<'a>,
    /// Next tape index to inspect; `end` is this level's exclusive bound.
    next: usize,
    end: usize,
    /// Byte position where the pending scalar gap starts.
    cursor: usize,
    /// Byte end of this level's contents.
    limit: usize,
}

impl<'a, 'p> Iterator for Items<'a, 'p> {
    type Item = Node<'a, 'p>;

    fn next(&mut self) -> Option<Node<'a, 'p>> {
        loop {
            if self.next >= self.end {
                let (s, e) = trim_ws(self.nested.data, self.cursor, self.limit);
                self.cursor = self.limit;
                if s < e {
                    return Some(Node { nested: self.nested, repr: Repr::Scalar(s, e) });
                }
                return None;
            }
            let entry = self.nested.tape[self.next];
            let pos = (entry as u32) as usize;
            match self.nested.data[pos] {
                __OPEN_PAT__ => {
                    // A pending scalar gap is yielded before the bracket
                    // (well-formed input always separates siblings, so the
                    // gap is normally whitespace).
                    let (s, e) = trim_ws(self.nested.data, self.cursor, pos);
                    if s < e {
                        self.cursor = pos;
                        return Some(Node { nested: self.nested, repr: Repr::Scalar(s, e) });
                    }
                    let node = Node { nested: self.nested, repr: Repr::Container(self.next) };
                    let close = (entry >> 32) as u32;
                    if close == u32::MAX || close as usize >= self.end {
                        // Unclosed (errored tape): the container swallows
                        // the rest of this level.
                        self.next = self.end;
                        self.cursor = self.limit;
                    } else {
                        self.next = close as usize + 1;
                        self.cursor = (self.nested.tape[close as usize] as u32) as usize + 1;
                    }
                    return Some(node);
                }
                _ => {
                    self.next += 1;
                    let (s, e) = trim_ws(self.nested.data, self.cursor, pos);
                    self.cursor = pos + 1;
                    if s < e {
                        return Some(Node { nested: self.nested, repr: Repr::Scalar(s, e) });
                    }
                }
            }
        }
    }
}

/// Flat iterator over every scalar span in the document; see
/// [`Nested::scalars`]. One linear pass over the tape: the gap between each pair
/// of consecutive structural-byte positions (and the trailing gap) is a scalar,
/// trimmed of whitespace and skipped if empty. No recursion, no per-level state.
pub struct Scalars<'a, 'p> {
    nested: &'p Nested<'a>,
    /// Next tape index to inspect.
    next: usize,
    /// Byte position where the pending scalar gap starts.
    cursor: usize,
}

impl<'a, 'p> Iterator for Scalars<'a, 'p> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let data = self.nested.data;
        let tape = &self.nested.tape;
        loop {
            if self.next >= tape.len() {
                // The trailing gap after the last structural byte, yielded at
                // most once (cursor is pushed past the input to stop).
                if self.cursor <= data.len() {
                    let (s, e) = trim_ws(data, self.cursor, data.len());
                    self.cursor = data.len() + 1;
                    if s < e {
                        return Some(&data[s..e]);
                    }
                }
                return None;
            }
            let pos = (tape[self.next] as u32) as usize;
            self.next += 1;
            let (s, e) = trim_ws(data, self.cursor, pos);
            self.cursor = pos + 1;
            if s < e {
                return Some(&data[s..e]);
            }
        }
    }
}

"##;
    code.push_str(
        &TEMPLATE
            .replace("__OPEN_PAT__", &open_pat)
            .replace("__EXPECTED_CLOSE_ARMS__", &expected_close_arms),
    );

    const PAR_TEMPLATE: &str = r##"/// Like [`parse_nested`], built across `threads` chunks; see
/// [`parse_nested_par_into`] for the steady-state variant.
pub fn parse_nested_par(data: &[u8], threads: usize) -> Nested<'_> {
    parse_nested_par_into(data, threads, Nested { data: &[], tape: Vec::new(), error: None })
}

/// Parallel [`parse_nested_into`]. A serial prepass replays the kernel,
/// snapshotting chunk-entry carries (exact for any dialect — no parity
/// tricks) and counting each chunk's tape slots; chunks then index and
/// match brackets concurrently, writing globally-indexed entries straight
/// into their disjoint ranges of the recycled master tape — no rebase or
/// concatenation pass exists. The few brackets that cross chunk
/// boundaries reconcile through an ordered residue merge (the classic
/// parenthesis reduction). Output is identical to the serial path;
/// malformed input falls back to serial so error and truncation
/// semantics match exactly.
pub fn parse_nested_par_into<'a>(
    data: &'a [u8],
    threads: usize,
    recycle: Nested<'_>,
) -> Nested<'a> {
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {
        return parse_nested_into(data, recycle);
    }
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| if t == threads { data.len() } else { (t * chunk).min(data.len()) })
        .collect();
    let mut entries: Vec<[u64; __K__]> = Vec::with_capacity(threads);
    entries.push(__ENTRY_INIT__);
    let mut counts: Vec<usize> = Vec::with_capacity(threads);
    nested_prepass_dispatch(data, &bounds, &mut entries, &mut counts);
    let mut bases: Vec<usize> = Vec::with_capacity(threads);
    let mut total = 0usize;
    for &count in &counts {
        bases.push(total);
        total += count;
    }

    let mut tape = recycle.tape;
    tape.clear();
    tape.reserve(total);
    // Workers write through this address into disjoint slot ranges; the
    // Vec itself is not touched again until after the scope.
    let master_addr = tape.as_mut_ptr() as usize;
    struct ChunkOutcome {
        error: Option<NestError>,
        written: usize,
        /// Leftover open stack, bottom to top: (global tape idx << 8) | close byte.
        opens: Vec<u64>,
        /// Closes with no local open, in order: (global tape idx << 32) | pos.
        pending: Vec<u64>,
    }
    let results: Vec<ChunkOutcome> =
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let slice = &data[bounds[t]..bounds[t + 1]];
                    let seed = entries[t];
                    let pos_base = bounds[t] as u32;
                    let tape_base = bases[t];
                    s.spawn(move || {
                        let master = master_addr as *mut u64;
                        let mut stack = Vec::with_capacity(64);
                        let mut pending = Vec::new();
                        // SAFETY: the prepass counted this chunk's slots
                        // exactly, the master capacity covers the total,
                        // and slot ranges are disjoint by prefix sum.
                        let (error, written) = unsafe {
                            nested_tape_seeded_dispatch(
                                slice, seed, pos_base, master, tape_base,
                                &mut stack, &mut pending,
                            )
                        };
                        ChunkOutcome { error, written, opens: stack, pending }
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("nested parse thread ok")).collect()
        });
    let consistent = results
        .iter()
        .zip(&counts)
        .all(|(outcome, &count)| outcome.error.is_none() && outcome.written == count);
    if !consistent {
        // A chunk found a definite mismatch (wrong close against an open
        // in the same chunk). Serial reproduces the exact first-error
        // truncation semantics; correctness over speed on malformed input.
        return parse_nested_into(data, Nested { data: &[], tape, error: None });
    }
    // SAFETY: every chunk wrote exactly its counted slots, so all `total`
    // entries are initialized.
    unsafe { tape.set_len(total) };

    // Residue merge: each chunk's pending closes match opens left by
    // earlier chunks, in order; leftover opens stack up for later chunks.
    // All indexes are already global.
    let mut gstack: Vec<u64> = Vec::new();
    let mut mismatch = false;
    'merge: for outcome in &results {
        for &pend in &outcome.pending {
            let close_idx = (pend >> 32) as usize;
            let close_pos = (pend as u32) as usize;
            match gstack.pop() {
                Some(top) if top as u8 == data[close_pos] => {
                    let open_idx = (top >> 8) as usize;
                    let open_pos = tape[open_idx] as u32;
                    tape[open_idx] = ((close_idx as u64) << 32) | open_pos as u64;
                    tape[close_idx] = ((open_idx as u64) << 32) | close_pos as u64;
                }
                _ => {
                    mismatch = true;
                    break 'merge;
                }
            }
        }
        gstack.extend_from_slice(&outcome.opens);
    }
    if mismatch {
        return parse_nested_into(data, Nested { data: &[], tape, error: None });
    }
    let error = gstack
        .last()
        .map(|&top| NestError::UnclosedOpen(tape[(top >> 8) as usize] as u32));
    Nested { data, tape, error }
}

/// # Safety
/// See `nested_tape_seeded`: the master slot range must be exclusively
/// owned and sized by the prepass count.
unsafe fn nested_tape_seeded_dispatch(
    data: &[u8],
    seed: [u64; __K__],
    pos_base: u32,
    master: *mut u64,
    tape_base: usize,
    stack: &mut Vec<u64>,
    pending: &mut Vec<u64>,
) -> (Option<NestError>, usize) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: features detected; slot contract forwarded to caller.
        return unsafe {
            avx512::nested_tape_seeded(data, seed, pos_base, master, tape_base, stack, pending)
        };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: features detected; slot contract forwarded to caller.
        return unsafe {
            avx2::nested_tape_seeded(data, seed, pos_base, master, tape_base, stack, pending)
        };
    }
    unsupported_cpu()
}

fn nested_prepass_dispatch(
    data: &[u8],
    bounds: &[usize],
    entries: &mut Vec<[u64; __K__]>,
    counts: &mut Vec<usize>,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx512::nested_prepass(data, bounds, entries, counts) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::nested_prepass(data, bounds, entries, counts) };
        return;
    }
    unsupported_cpu();
}

/// `push_nested` twin for parallel chunks: byte reads are chunk-local
/// (`offset` indexes the chunk slice, `pos_base + offset` is the global
/// position), entries land in the master tape at `tape_base + tlen` with
/// globally-indexed partners, and a close with no local open becomes a
/// pending-residue entry rather than an error — its open lives in an
/// earlier chunk.
///
/// # Safety
/// The caller owns master slots `tape_base..tape_base + (this chunk's
/// prepass count)`; `tlen` stays within that range by construction.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn push_nested_par(
    data: &[u8],
    mut mask: u64,
    opens: u64,
    closes: u64,
    offset: u32,
    pos_base: u32,
    master: *mut u64,
    tape_base: usize,
    tlen: &mut usize,
    stack: &mut Vec<u64>,
    pending: &mut Vec<u64>,
) -> Option<NestError> {
    let events = mask.count_ones() as usize;
    stack.reserve(events);
    let stack_ptr = stack.as_mut_ptr();
    let mut slen = stack.len();
    let mut t = *tlen;
    let mut error = None;
    let brackets = opens | closes;
    while mask != 0 {
        let lowest = mask & mask.wrapping_neg();
        let local = offset + mask.trailing_zeros();
        let pos = pos_base + local;
        mask &= mask - 1;
        if lowest & brackets == 0 {
            // SAFETY: tape_base + t stays in the owned slot range (one
            // entry per structural event, prepass-counted); same for all
            // master writes below.
            unsafe { *master.add(tape_base + t) = pos as u64 };
            t += 1;
        } else if lowest & opens != 0 {
            // SAFETY: as above; `local` indexes the chunk slice (drivers
            // mask off tail padding), stack writes are reserved.
            unsafe {
                let close = match *data.get_unchecked(local as usize) {
__EXPECTED_CLOSE_ARMS__                };
                *stack_ptr.add(slen) = (((tape_base + t) as u64) << 8) | close as u64;
                slen += 1;
                *master.add(tape_base + t) = ((u32::MAX as u64) << 32) | pos as u64;
            }
            t += 1;
        } else if slen == 0 {
            pending.push((((tape_base + t) as u64) << 32) | pos as u64);
            // SAFETY: as above; the partner stays pending until the merge.
            unsafe { *master.add(tape_base + t) = ((u32::MAX as u64) << 32) | pos as u64 };
            t += 1;
        } else {
            slen -= 1;
            // SAFETY: as above; the patched open slot is an earlier entry
            // this same chunk wrote (global indexes, own range).
            unsafe {
                let top = *stack_ptr.add(slen);
                if top as u8 != *data.get_unchecked(local as usize) {
                    error = Some(NestError::UnmatchedClose(pos));
                    break;
                }
                let open = (top >> 8) as usize;
                *master.add(open) =
                    (((tape_base + t) as u64) << 32) | (*master.add(open) as u32) as u64;
                *master.add(tape_base + t) = ((open as u64) << 32) | pos as u64;
            }
            t += 1;
        }
    }
    *tlen = t;
    // SAFETY: slen counts initialized entries within the reserve.
    unsafe { stack.set_len(slen) };
    error
}

"##;
    let entry_init = if carry_count > 0 {
        "CARRY_INIT"
    } else {
        "[0u64; 0]"
    };
    code.push_str(
        &PAR_TEMPLATE
            .replace("__EXPECTED_CLOSE_ARMS__", &expected_close_arms)
            .replace("__ENTRY_INIT__", entry_init)
            .replace("__K__", &carry_count.to_string()),
    );
}
