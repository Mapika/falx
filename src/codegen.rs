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

/// Which kernel a step body is being emitted for.
#[derive(Clone, Copy, PartialEq)]
enum Flavor {
    Avx2,
    Fallback,
}

/// Emit a generated source file exposing only the structural indexer.
pub fn emit(graph: &Graph, format_name: &str) -> Result<String, CodegenError> {
    emit_with(graph, format_name, None)
}

/// Emit a full generated parser for a delimited dialect: the structural
/// indexer, a record-aware tape indexer (separator stream + record-end
/// stream), and a records/fields span API with quote stripping and
/// escape-aware field cleaning.
pub fn emit_parser(
    dialect: &crate::formats::Dialect,
    format_name: &str,
) -> Result<String, CodegenError> {
    let parts = crate::formats::delimited_parts(dialect);
    emit_with(&parts.graph, format_name, Some((dialect, parts.terminators)))
}

fn emit_with(
    graph: &Graph,
    format_name: &str,
    parser: Option<(&crate::formats::Dialect, crate::ir::NodeId)>,
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

    if let Some(dialect) = dialect {
        push_span_api(&mut code, dialect);
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
{fallback_tape}
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

{avx2_tape}
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

/// Emit the records/fields span API: lazy walking of the structural index
/// into record and field spans, with dialect-specific field cleaning.
fn push_span_api(code: &mut String, dialect: &crate::formats::Dialect) {
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

    /// Raw span of field `i`: quotes and escapes intact.
    pub fn field_raw(&self, i: usize) -> Option<&'p [u8]> {
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
        Some(&self.data[from..to])
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
    code.push_str(&clean_fn(dialect));
    code.push('\n');
}

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
