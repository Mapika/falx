//! IR-to-Rust code generation.
//!
//! [`emit`] turns a bitstream [`Graph`] into a self-contained Rust source
//! file with no dependency on this crate: a public `index_structurals`
//! entry point that runtime-dispatches between an AVX2+PCLMULQDQ kernel and
//! a portable scalar fallback. Both kernels run the same blockwise node
//! schedule the interpreter does, so the three implementations are
//! differential-testable against each other.
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

/// Largest class emitted as an OR of byte compares; bigger classes need the
/// (future) shuffle-based classifier.
const MAX_CLASS_BYTES: usize = 8;

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

    fn rust_type(&self) -> &'static str {
        match self.ty {
            ColumnType::I64 => "i64",
            ColumnType::F64 => "f64",
            ColumnType::Bytes => "(u32, u32)",
        }
    }

    fn zero(&self) -> &'static str {
        match self.ty {
            ColumnType::I64 => "0",
            ColumnType::F64 => "0.0",
            ColumnType::Bytes => "(0, 0)",
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
        // Each column claims `name` and `name_valid`; collisions between
        // any of them make the generated struct uncompilable.
        if !seen.insert(name.clone()) || !seen.insert(format!("{name}_valid")) {
            return Err(CodegenError(format!(
                "column name '{name}' collides with another column"
            )));
        }
    }
    Ok(())
}

/// Which kernel a step body is being emitted for.
#[derive(Clone, Copy, PartialEq)]
enum Flavor {
    Avx2,
    Fallback,
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
    validate_columns(columns)?;
    let parts = crate::formats::delimited_parts(dialect);
    emit_with(
        &parts.graph,
        format_name,
        Some((dialect, parts.terminators)),
        columns,
    )
}

fn emit_with(
    graph: &Graph,
    format_name: &str,
    parser: Option<(&crate::formats::Dialect, crate::ir::NodeId)>,
    columns: &[Column],
) -> Result<String, CodegenError> {
    let dialect = parser.map(|(d, _)| d);
    let output = graph.output();

    // Assign carry slots to stateful nodes.
    let mut carry_slot = vec![usize::MAX; graph.nodes().len()];
    let mut carry_count = 0usize;
    for (i, op) in graph.nodes().iter().enumerate() {
        if matches!(op, Op::ShiftLeft1(_) | Op::PrefixXor(_) | Op::Add(_, _)) {
            carry_slot[i] = carry_count;
            carry_count += 1;
        }
    }

    let uses_class = graph.nodes().iter().any(|op| matches!(op, Op::Class(_)));
    let uses_prefix_xor = graph
        .nodes()
        .iter()
        .any(|op| matches!(op, Op::PrefixXor(_)));

    for op in graph.nodes() {
        if let Op::Class(class) = op {
            let n = class.members().count();
            if n > MAX_CLASS_BYTES {
                return Err(CodegenError(format!(
                    "character class with {n} bytes exceeds the compare-based \
                     limit of {MAX_CLASS_BYTES}"
                )));
            }
        }
    }

    // Pieces that differ depending on whether the graph carries state.
    let carry_decl = if carry_count > 0 {
        format!("        let mut carries = [0u64; {carry_count}];\n")
    } else {
        String::new()
    };
    let carry_param = if carry_count > 0 {
        format!(", carries: &mut [u64; {carry_count}]")
    } else {
        String::new()
    };
    let carry_arg = if carry_count > 0 { ", &mut carries" } else { "" };

    // In parser mode the step function also returns the record-terminator
    // subset of the structural mask, so tape indexing gets record boundaries
    // for free; the plain indexer selects the first element.
    let step_ret_ty = if parser.is_some() { "(u64, u64)" } else { "u64" };
    let sel = if parser.is_some() { ".0" } else { "" };
    let step_ret = match parser {
        Some((_, term)) => format!("(v{out}, v{out} & v{term})", out = output.0, term = term.0),
        None => format!("v{}", output.0),
    };

    // Parallel indexing (emitted for doubled-quote/no-escape dialects):
    // a chunk's entry state is one bit — the parity of quote bytes before
    // it — so a counting prepass makes chunks independent.
    let par_mode = matches!(
        dialect,
        Some(d) if d.escape == crate::formats::Escape::None
    );
    let seed_init = if carry_count == 1 {
        "        let mut carries = [seed];\n".to_string()
    } else if carry_count == 0 {
        "        let _ = seed;\n".to_string()
    } else {
        String::new() // par_mode never emits with >1 carry
    };
    let seeded_kernel = |loads: &str, tail_loads: &str, attr: &str| {
        format!(
            r#"
    /// Like `index_structurals` but with seeded quote-parity carry and an
    /// absolute base offset: the building block for parallel indexing.
{attr}    pub fn index_structurals_seeded(data: &[u8], seed: u64, base: u32, out: &mut Vec<u32>) {{
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
    }}
"#
        )
    };
    let fallback_seeded = if par_mode {
        seeded_kernel(
            &format!("let block: &[u8; 64] = data[offset..offset + 64].try_into().unwrap();\n            let mask = step(block{carry_arg}){sel};"),
            &format!("let mask = step(&block{carry_arg}){sel} & ((1u64 << rem) - 1);"),
            "",
        )
    } else {
        String::new()
    };
    let avx2_seeded = if par_mode {
        seeded_kernel(
            &format!("// SAFETY: offset + 64 <= data.len().\n            let mask = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }}{sel};"),
            &format!("// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let mask = unsafe {{ step(block.as_ptr(){carry_arg}) }}{sel} & ((1u64 << rem) - 1);"),
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
        )
    } else {
        String::new()
    };
    let fallback_tape_seeded = if par_mode {
        format!(
            r#"
    /// Seeded-carry, based variant of `index_tape` for parallel parsing.
    pub fn index_tape_seeded(data: &[u8], seed: u64, base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {{
{seed_init}        let mut offset = 0usize;
        while offset + 64 <= data.len() {{
            let block: &[u8; 64] = data[offset..offset + 64].try_into().unwrap();
            let (mask, term) = step(block{carry_arg});
            push_tape(mask, term, base + offset as u32, seps, ends);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            let (mask, term) = step(&block{carry_arg});
            push_tape(mask & live, term & live, base + offset as u32, seps, ends);
        }}
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
    pub fn index_tape_seeded(data: &[u8], seed: u64, base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {{
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
    }}
"#
        )
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
        let (doc, params, init, start) = if par_mode {
            (
                " Scans 64-byte blocks from\n    /// `start` (block-aligned) onward, until end of data or until the\n    /// sink completes its record range.",
                "data: &[u8], seed: u64, start: usize, ",
                seed_init.as_str(),
                "        let mut offset = start;\n",
            )
        } else {
            ("", "data: &[u8], ", carry_decl.as_str(), "        let mut offset = 0usize;\n")
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
    let fallback_cells = if has_columns {
        cells_fill(
            "",
            &format!("let block: &[u8; 64] = data[offset..offset + 64].try_into().unwrap();\n            let (mask, term) = step(block{carry_arg});"),
            &format!("let (mask, term) = step(&block{carry_arg});"),
        )
    } else {
        String::new()
    };
    let avx2_cells = if has_columns {
        cells_fill(
            "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n",
            &format!("// SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_arg}) }};"),
            &format!("// SAFETY: block is a readable 64-byte buffer. Pad bits masked.\n            let (mask, term) = unsafe {{ step(block.as_ptr(){carry_arg}) }};"),
        )
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
    let fallback_partial = if parser.is_some() {
        partial_tpl
            .replace("@ATTR@", "")
            .replace("@K@", &carry_count.to_string())
            .replace("@LOAD@", &format!("            let block: &[u8; 64] = data[offset..offset + 64].try_into().unwrap();\n            let (mask, term) = step(block{carry_fwd});\n"))
            .replace("@LOAD2@", &format!("        let (mask, term) = step(block{carry_fwd});\n"))
    } else {
        String::new()
    };
    let avx2_partial = if parser.is_some() {
        partial_tpl
            .replace("@ATTR@", "    #[target_feature(enable = \"avx2\", enable = \"pclmulqdq\")]\n")
            .replace("@K@", &carry_count.to_string())
            .replace("@LOAD@", &format!("            // SAFETY: offset + 64 <= data.len().\n            let (mask, term) = unsafe {{ step(data.as_ptr().add(offset){carry_fwd}) }};\n"))
            .replace("@LOAD2@", &format!("        // SAFETY: block is a readable 64-byte buffer.\n        let (mask, term) = unsafe {{ step(block.as_ptr(){carry_fwd}) }};\n"))
    } else {
        String::new()
    };

    let par_block = if par_mode {
        let prepass = prepass_snippet(dialect.and_then(|d| d.quote));
        format!(
            r#"
/// Parallel structural indexing: byte-identical to [`index_structurals`],
/// split across `threads` chunks. Quote context is reconstructed with a
/// counting prepass, so both passes run fully parallel.
pub fn index_structurals_par(data: &[u8], threads: usize, out: &mut Vec<u32>) {{
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {{
        index_structurals(data, out);
        return;
    }}
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| if t == threads {{ data.len() }} else {{ (t * chunk).min(data.len()) }})
        .collect();
{prepass}    // Pass 2: index chunks concurrently with seeded entry state.
    std::thread::scope(|s| {{
        let handles: Vec<_> = (0..threads)
            .map(|t| {{
                let slice = &data[bounds[t]..bounds[t + 1]];
                let seed = entry[t];
                let base = bounds[t] as u32;
                s.spawn(move || {{
                    let mut part = Vec::with_capacity(slice.len() / 16 + 8);
                    index_structurals_seeded_dispatch(slice, seed, base, &mut part);
                    part
                }})
            }})
            .collect();
        for handle in handles {{
            out.extend_from_slice(&handle.join().expect("index thread ok"));
        }}
    }});
}}

fn index_structurals_seeded_dispatch(data: &[u8], seed: u64, base: u32, out: &mut Vec<u32>) {{
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        unsafe {{ avx2::index_structurals_seeded(data, seed, base, out) }};
        return;
    }}
    fallback::index_structurals_seeded(data, seed, base, out);
}}

/// Parallel [`parse`]: identical tape, built across `threads` chunks.
/// Chunk tapes concatenate directly; each end entry's cumulative separator
/// count is rebased with one add during the merge.
pub fn parse_par(data: &[u8], threads: usize) -> Parsed<'_> {{
    let threads = threads.max(1).min(data.len() / 64 + 1);
    let chunk = (data.len() / threads + 63) & !63;
    if threads == 1 || chunk == 0 {{
        return parse(data);
    }}
    let bounds: Vec<usize> = (0..=threads)
        .map(|t| if t == threads {{ data.len() }} else {{ (t * chunk).min(data.len()) }})
        .collect();
{prepass}    let parts: Vec<(Vec<u32>, Vec<u64>)> = std::thread::scope(|s| {{
        let handles: Vec<_> = (0..threads)
            .map(|t| {{
                let slice = &data[bounds[t]..bounds[t + 1]];
                let seed = entry[t];
                let base = bounds[t] as u32;
                s.spawn(move || {{
                    let mut seps = Vec::with_capacity(slice.len() / 16 + 8);
                    let mut ends = Vec::with_capacity(slice.len() / 32 + 8);
                    index_tape_seeded_dispatch(slice, seed, base, &mut seps, &mut ends);
                    (seps, ends)
                }})
            }})
            .collect();
        handles.into_iter().map(|h| h.join().expect("parse thread ok")).collect()
    }});
    let mut seps = Vec::with_capacity(parts.iter().map(|p| p.0.len()).sum::<usize>());
    let mut ends = Vec::with_capacity(parts.iter().map(|p| p.1.len()).sum::<usize>());
    for (part_seps, part_ends) in &parts {{
        let rebase = (seps.len() as u64) << 32;
        seps.extend_from_slice(part_seps);
        ends.extend(part_ends.iter().map(|&e| e + rebase));
    }}
    Parsed {{ data, seps, ends }}
}}

fn index_tape_seeded_dispatch(data: &[u8], seed: u64, base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {{
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        unsafe {{ avx2::index_tape_seeded(data, seed, base, seps, ends) }};
        return;
    }}
    fallback::index_tape_seeded(data, seed, base, seps, ends);
}}
"#
        )
    } else {
        String::new()
    };

    // Record-aware tape indexers, emitted only in parser mode. Identical in
    // both kernels except for how a block reaches step().
    let fallback_tape = if parser.is_some() {
        format!(
            r#"
    /// Record-aware indexing for [`crate::parse`]-style use: separator
    /// positions into `seps`, record ends into `ends` encoded as
    /// (cumulative separator count << 32) | byte position.
    pub fn index_tape(data: &[u8], seps: &mut Vec<u32>, ends: &mut Vec<u64>) {{
{carry_decl}        let mut offset = 0usize;
        while offset + 64 <= data.len() {{
            let block: &[u8; 64] = data[offset..offset + 64].try_into().unwrap();
            let (mask, term) = step(block{carry_arg});
            push_tape(mask, term, offset as u32, seps, ends);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            let live = (1u64 << rem) - 1;
            let (mask, term) = step(&block{carry_arg});
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
    /// Record-aware indexing; see the fallback twin for the tape encoding.
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
/// Index the structural positions of `data` into `out`.
pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {{
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {{
        // SAFETY: the required target features were just detected.
        unsafe {{ avx2::index_structurals(data, out) }};
        return;
    }}
    fallback::index_structurals(data, out);
}}

"#
    );

    code.push_str(&par_block);

    if let Some(dialect) = dialect {
        push_span_api(&mut code, dialect, carry_count);
        if !columns.is_empty() {
            push_columns_api(&mut code, dialect, columns, par_mode);
        }
    }

    let _ = write!(
        code,
        r#"/// Portable kernel, public so the dispatch-bypassed path stays testable.
pub mod fallback {{
    pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {{
{carry_decl}        let mut offset = 0usize;
        while offset + 64 <= data.len() {{
            let block: &[u8; 64] = data[offset..offset + 64].try_into().unwrap();
            let mask = step(block{carry_arg}){sel};
            push_indexes(mask, offset as u32, out);
            offset += 64;
        }}
        let rem = data.len() - offset;
        if rem > 0 {{
            let mut block = [0u8; 64];
            block[..rem].copy_from_slice(&data[offset..]);
            // Mask off bits that fall in the zero padding.
            let mask = step(&block{carry_arg}){sel} & ((1u64 << rem) - 1);
            push_indexes(mask, offset as u32, out);
        }}
    }}
{fallback_tape}{fallback_seeded}{fallback_tape_seeded}{fallback_partial}{fallback_cells}
    #[inline]
    fn step(block: &[u8; 64]{carry_param}) -> {step_ret_ty} {{
"#
    );
    emit_step_body(&mut code, graph, &carry_slot, Flavor::Fallback);
    let _ = write!(code, "        {step_ret}\n    }}\n");

    if uses_class {
        code.push_str(
            r#"
    #[inline]
    fn eq_mask(block: &[u8; 64], byte: u8) -> u64 {
        let mut mask = 0u64;
        let mut i = 0;
        while i < 64 {
            mask |= ((block[i] == byte) as u64) << i;
            i += 1;
        }
        mask
    }
"#,
        );
    }
    if uses_prefix_xor {
        code.push_str(
            r#"
    #[inline]
    fn prefix_xor(mut x: u64) -> u64 {
        x ^= x << 1;
        x ^= x << 2;
        x ^= x << 4;
        x ^= x << 8;
        x ^= x << 16;
        x ^= x << 32;
        x
    }
"#,
        );
    }
    code.push_str(
        r#"
    #[inline]
    fn push_indexes(mut mask: u64, base: u32, out: &mut Vec<u32>) {
        while mask != 0 {
            out.push(base + mask.trailing_zeros());
            mask &= mask - 1;
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

{avx2_tape}{avx2_seeded}{avx2_tape_seeded}{avx2_partial}{avx2_cells}
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
    emit_step_body(&mut code, graph, &carry_slot, Flavor::Avx2);
    let _ = write!(code, "        {step_ret}\n    }}\n");

    if uses_class {
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
    if uses_prefix_xor {
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

    Ok(code)
}

/// The quote-parity counting prepass shared by every parallel entry
/// point; the emitted code leaves `entry[t]` as the carry seed for chunk t.
fn prepass_snippet(quote: Option<u8>) -> String {
    match quote {
        Some(q) => format!(
            r#"    // Pass 1: parity of quote bytes per chunk (doubled-quote escapes
    // self-cancel, so raw parity is exact). Auto-vectorizes to ~memory speed.
    let mut entry = vec![0u64; threads];
    std::thread::scope(|s| {{
        let handles: Vec<_> = (0..threads)
            .map(|t| {{
                let slice = &data[bounds[t]..bounds[t + 1]];
                s.spawn(move || slice.iter().filter(|&&b| b == {q}u8).count() as u64 & 1)
            }})
            .collect();
        let mut parity = 0u64;
        for (t, h) in handles.into_iter().enumerate() {{
            entry[t] = parity.wrapping_neg();
            parity ^= h.join().expect("prepass thread ok");
        }}
    }});
"#
        ),
        None => r#"    // No quote context in this dialect: chunks are independent.
    let entry = vec![0u64; threads];
"#
        .to_string(),
    }
}

/// Emit the records/fields span API: lazy walking of the structural index
/// into record and field spans, with dialect-specific field cleaning.
fn push_span_api(code: &mut String, dialect: &crate::formats::Dialect, carry_count: usize) {
    code.push_str(
        r#"/// Record-aware tape indexing used by [`parse`].
fn index_tape(data: &[u8], seps: &mut Vec<u32>, ends: &mut Vec<u64>) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_tape(data, seps, ends) };
        return;
    }
    fallback::index_tape(data, seps, ends);
}

/// Index `data` and return a lazy record/field view over it.
pub fn parse(data: &[u8]) -> Parsed<'_> {
    let mut seps = Vec::with_capacity(data.len() / 16 + 8);
    let mut ends = Vec::with_capacity(data.len() / 32 + 8);
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

impl<'p> Iterator for Records<'p> {
    type Item = Record<'p>;

    fn next(&mut self) -> Option<Record<'p>> {
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

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_sep < self.seps.len() {
            let to = self.seps[self.next_sep] as usize;
            self.next_sep += 1;
            let span = &self.data[self.from..to];
            self.from = to + 1;
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

"#,
    );
    code.push_str(
        &STREAM_TPL.replace("@K@", &carry_count.to_string()),
    );
    code.push_str(&clean_fn(dialect));
    code.push('\n');
}

/// Emit the typed columnar projection API: a `Columns` struct with one
/// values Vec plus an Arrow-style validity bitmap per declared column,
/// filled by walking the structural tape so unrequested fields are never
/// read, cleaned, or copied.
fn push_columns_api(
    code: &mut String,
    dialect: &crate::formats::Dialect,
    columns: &[Column],
    par_mode: bool,
) {
    let any_i64 = columns.iter().any(|c| c.ty == ColumnType::I64);
    let any_f64 = columns.iter().any(|c| c.ty == ColumnType::F64);
    let any_bytes = columns.iter().any(|c| c.ty == ColumnType::Bytes);

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
        let ty = column.rust_type();
        let what = match column.ty {
            ColumnType::I64 => "as `i64`",
            ColumnType::F64 => "as `f64`",
            ColumnType::Bytes => "as raw `(start, end)` spans into `data`",
        };
        let _ = writeln!(
            code,
            "    /// Field {} of each record {what}; zero where invalid.\n\
             \x20   pub {name}: Vec<{ty}>,\n\
             \x20   /// Validity bitmap for `{name}`.\n\
             \x20   pub {name}_valid: Vec<u64>,",
            column.index
        );
    }
    code.push_str("}\n\n");

    // --- impl: constructor, span resolver, row push ------------------------
    code.push_str("impl<'a> Columns<'a> {\n    fn with_capacity(data: &'a [u8], rows: usize) -> Self {\n        Columns {\n            data,\n            rows: 0,\n");
    for column in columns {
        let name = column.field_name();
        let _ = writeln!(
            code,
            "            {name}: Vec::with_capacity(rows),\n\
             \x20           {name}_valid: Vec::with_capacity(rows / 64 + 1),"
        );
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
        let _ = writeln!(code, "            self.cols.{}_valid.push(0);", column.field_name());
    }
    code.push_str("        }\n");
    for (k, column) in columns.iter().enumerate() {
        let name = column.field_name();
        let found_bit = 1u32 << k;
        let body = match column.ty {
            ColumnType::I64 | ColumnType::F64 => {
                let parser = if column.ty == ColumnType::I64 {
                    if dialect.quote.is_some() { "parse_i64_field" } else { "parse_i64_cell" }
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

    // --- serial entry point -------------------------------------------------
    let (dispatch_sig, dispatch_serial_args, seed_param) = if par_mode {
        (
            "data: &[u8], seed: u64, start: usize, sink: &mut ColumnSink",
            "data, 0, 0, &mut sink",
            "data, seed, start, sink",
        )
    } else {
        ("data: &[u8], sink: &mut ColumnSink", "data, &mut sink", "data, sink")
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
         \x20   if std::arch::is_x86_feature_detected!(\"avx2\")\n\
         \x20       && std::arch::is_x86_feature_detected!(\"pclmulqdq\")\n\
         \x20   {{\n\
         \x20       // SAFETY: the required target features were just detected.\n\
         \x20       unsafe {{ avx2::index_cells({seed_param}) }};\n\
         \x20       return;\n\
         \x20   }}\n\
         \x20   fallback::index_cells({seed_param});\n\
         }}\n\n"
    );

    // --- parallel entry point ------------------------------------------------
    if par_mode {
        let prepass = prepass_snippet(dialect.quote);
        let _ = write!(
            code,
            "/// Parallel [`parse_columns`]: records are assigned to workers by\n\
             /// terminator ownership — worker t skips to the first record\n\
             /// boundary at or past its chunk start, and finishes the record it\n\
             /// is mid-way through at chunk end — so every record is converted\n\
             /// exactly once, with no tape built. Quote context comes from the\n\
             /// same counting prepass as [`parse_par`]; column chunks\n\
             /// concatenate, validity bitmaps stitch bit-shifted.\n\
             pub fn parse_columns_par(data: &[u8], threads: usize) -> Columns<'_> {{\n\
             \x20   let threads = threads.max(1).min(data.len() / 64 + 1);\n\
             \x20   let chunk = (data.len() / threads + 63) & !63;\n\
             \x20   if threads == 1 || chunk == 0 {{\n\
             \x20       return parse_columns(data);\n\
             \x20   }}\n\
             \x20   let bounds: Vec<usize> = (0..=threads)\n\
             \x20       .map(|t| if t == threads {{ data.len() }} else {{ (t * chunk).min(data.len()) }})\n\
             \x20       .collect();\n\
             {prepass}\
             \x20   let parts: Vec<Columns<'_>> = std::thread::scope(|s| {{\n\
             \x20       let handles: Vec<_> = (0..threads)\n\
             \x20           .map(|t| {{\n\
             \x20               let seed = entry[t];\n\
             \x20               let (start, end) = (bounds[t], bounds[t + 1]);\n\
             \x20               s.spawn(move || {{\n\
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
            let _ = writeln!(
                code,
                "        cols.{name}.extend_from_slice(&part.{name});\n\
                 \x20       append_bitmap(&mut cols.{name}_valid, cols.rows, &part.{name}_valid, part.rows);"
            );
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
fn parse_f64_cell(s: &[u8]) -> Option<f64> {
    const POW10: [f64; 23] = [
        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
        1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
    ];
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
    if digits > 15 || exp10 < -22 || exp10 > 22 {
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
        carries: [0u64; @K@],
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
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_tape_partial(data, carries, base, seps, ends) };
        return;
    }
    fallback::index_tape_partial(data, carries, base, seps, ends);
}

fn index_tape_block_dispatch(block: &[u8; 64], live: u64, carries: &mut [u64; @K@], base: u32, seps: &mut Vec<u32>, ends: &mut Vec<u64>) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
    {
        // SAFETY: the required target features were just detected.
        unsafe { avx2::index_tape_block(block, live, carries, base, seps, ends) };
        return;
    }
    fallback::index_tape_block(block, live, carries, base, seps, ends);
}

"#;

/// The dialect-specific field-cleaning function for the span API.
fn clean_fn(dialect: &crate::formats::Dialect) -> String {
    use crate::formats::Escape;
    match (dialect.quote, dialect.escape) {
        (None, _) => r#"/// No quote convention in this dialect: fields are returned verbatim.
fn clean(raw: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    std::borrow::Cow::Borrowed(raw)
}
"#
        .to_string(),
        (Some(q), Escape::None) => format!(
            r#"/// Strip surrounding quotes and collapse doubled escape quotes.
fn clean(raw: &[u8]) -> std::borrow::Cow<'_, [u8]> {{
    const Q: u8 = {q}u8;
    if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {{
        let inner = &raw[1..raw.len() - 1];
        if !inner.windows(2).any(|w| w[0] == Q && w[1] == Q) {{
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
"#
        ),
        (Some(q), Escape::Backslash(e)) => format!(
            r#"/// Strip surrounding quotes and resolve backslash escapes (`\x` -> `x`).
fn clean(raw: &[u8]) -> std::borrow::Cow<'_, [u8]> {{
    const Q: u8 = {q}u8;
    const E: u8 = {e}u8;
    let body = if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {{
        &raw[1..raw.len() - 1]
    }} else {{
        raw
    }};
    if !body.contains(&E) {{
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
"#
        ),
    }
}

/// Emit one `let vN = ...;` line per node. Shared between kernels except for
/// the `Class` byte-comparison primitive.
fn emit_step_body(code: &mut String, graph: &Graph, carry_slot: &[usize], flavor: Flavor) {
    let class_args = match flavor {
        Flavor::Avx2 => "lo, hi",
        Flavor::Fallback => "block",
    };
    for (i, op) in graph.nodes().iter().enumerate() {
        let line = match *op {
            Op::Class(class) => {
                let compares: Vec<String> = class
                    .members()
                    .map(|b| format!("eq_mask({class_args}, {b}u8)"))
                    .collect();
                let label: String = class.members().map(|b| b.escape_ascii().to_string()).collect();
                format!("let v{i} = {}; // class \"{label}\"", compares.join(" | "))
            }
            Op::Const(pattern) => format!("let v{i} = {pattern:#018x}u64;"),
            Op::Not(a) => format!("let v{i} = !v{};", a.0),
            Op::And(a, b) => format!("let v{i} = v{} & v{};", a.0, b.0),
            Op::Or(a, b) => format!("let v{i} = v{} | v{};", a.0, b.0),
            Op::Xor(a, b) => format!("let v{i} = v{} ^ v{};", a.0, b.0),
            Op::ShiftLeft1(a) => {
                let k = carry_slot[i];
                format!(
                    "let v{i} = {{ let shifted = (v{a} << 1) | carries[{k}]; \
                     carries[{k}] = v{a} >> 63; shifted }};",
                    a = a.0
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
