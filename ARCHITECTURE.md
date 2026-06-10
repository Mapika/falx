# Falx Architecture

## Pipeline Overview

The falx parser generator transforms declarative format specifications into self-contained SIMD parsing kernels:

1. **Input**: Format spec (e.g., `specs/csv.toml`) defines structural bytes, quote character, and escape convention.
2. **Dialect**: Parsed into `formats::Dialect` describing the delimited format family.
3. **IR Graph**: Built by `formats::delimited_parts()`, a DAG of bitstream operations (`src/ir.rs`) on which every execution strategy can operate.
4. **Execution**:
   - `interp::run()` — reference interpreter (ground truth, byte-at-a-time execution for testing).
   - `codegen::emit_parser()` — generates a self-contained Rust file with two kernels (AVX2+PCLMULQDQ and scalar fallback), runtime dispatch, and a span API with quote stripping and unescaping.
5. **Output**: A single `.rs` file (std-only) exposing `index_structurals(data, out)`, a records/fields span API, and — when the spec declares typed columns — a columnar `parse_columns` API.

All three executors (interp, codegen AVX2, codegen scalar) run the same blockwise node schedule and are differential-testable against each other.

## The Bitstream IR

Operations over 64-bit vectors with one bit per input byte, executed 64 bytes per block. The algebra has eight operations:

- **Class(CharClass)**: Stateless. Bit i is set iff input byte i matches the character class (e.g., commas, quotes, newlines).
- **Const(u64)**: A constant pattern repeated in every block (e.g., even/odd position masks for escape handling).
- **Not(a), And(a,b), Or(a,b), Xor(a,b)**: Bitwise logic, all stateless.
- **ShiftLeft1(a)**: Bit i of output is bit i-1 of operand ("the previous byte matched"). Carries one bit across blocks; used for lookahead like "escaped positions."
- **PrefixXor(a)**: Bit i is the XOR of operand bits 0..=i — running parity via log-step shifts. Carries one parity bit across blocks. Implements the quote-context trick and is the only arithmetic needed for doubled-quote (RFC 4180) escaping.
- **Add(a,b)**: 64-bit binary addition with carry propagated across blocks. Used for odd/even run detection in backslash-escape handling (the simdjson algorithm).

Each op maps to one or two machine instructions. Stateful ops (`ShiftLeft1`, `PrefixXor`, `Add`) thread one or two bits of state between blocks—the entire kernel's memory. A graph is topologically sorted by construction; all evaluators run a single forward pass per block with no backtracking or allocation.

## Generated Kernel Anatomy

The emitted code exports `index_structurals(data: &[u8], out: &mut Vec<u32>)`, which appends byte offsets of set bits in the output stream.

### Structure

1. **Dispatch wrapper**: Runtime detection of AVX2 + PCLMULQDQ; falls back to scalar.
2. **Scalar fallback** (`fallback` module):
   - `step(block: &[u8; 64], carries) -> u64`: Evaluates every IR node over one block, returning the output mask. Inlined.
   - Implements `Class` via `eq_mask()` (loop over 64 bytes, compare and set bits).
   - Implements `PrefixXor` via `prefix_xor()` (log-step XOR cascade).
3. **AVX2 kernel** (`avx2` module):
   - Same `step()` logic but with target-feature attributes.
   - `Class` via `_mm256_cmpeq_epi8` on five u64s (covering all 256 byte values).
   - `PrefixXor` via PCLMULQDQ-based carryless multiply.
4. **Drivers**:
   - Unrolled 128-byte loops (two blocks per iteration) for AVX2 throughput.
   - Single-block loops for scalar and AVX2 tail handling.
5. **Tape indexing** (parser mode only): `push_tape()` flattens structurals into two streams:
   - Separator positions (`seps`).
   - Record ends, encoded as `(cumulative_sep_count << 32) | position`.
6. **Span API** (parser mode): `parse(data) -> records() -> field(i)` with quote stripping and escape resolution via lazy `Cow<[u8]>` slices that only allocate if escapes are present.

### Typed Columnar Projection

Specs may declare typed columns (`[[columns]]` with `index`, `type` of
`i64`/`f64`/`bytes`, optional `name`); the generated parser then also
exposes `parse_columns(data) -> Columns` (and `parse_columns_par` for
parallel-capable dialects).

**Layout** — per column, a values `Vec<T>` plus a validity bitmap
(`Vec<u64>`, LSB-first, bit `r` = row `r` parsed). A missing, empty, or
malformed cell clears its bit and leaves a zero placeholder in the values
Vec; nothing panics, and every column always has exactly `rows` entries.
This is deliberately the Arrow primitive-array layout (values buffer +
null bitmap, LSB-first), so handing a column to Arrow is a buffer wrap,
not a conversion — see `examples/arrow_interop.rs`. Generated files stay
std-only; the Arrow dependency lives in that example, never in the kernel.

**Projection** — rows are filled by walking the structural tape: field
boundaries come from `seps` positions, so only the bytes of *declared*
columns are read (and only those cells are quote-cleaned). A 20-field CSV
with two declared columns never touches the other 18 fields' bytes after
indexing. `bytes` columns store raw `(start, end)` spans into the input —
zero-copy, quotes and escapes intact.

**Number parsing** —
- `i64`: cells of ≤16 digits (effectively all real data) are parsed as two
  SWAR 8-digit blocks (Lemire's multiply-and-shift atoi) after a SWAR
  all-digits check; longer cells take a checked scalar path that rejects
  overflow exactly like `str::parse`. Acceptance rules are identical to
  `str::parse::<i64>`.
- `f64`: the Clinger (1990) fast path — when the decimal mantissa fits in
  15 digits and the decimal exponent is within ±22, both mantissa and
  power of ten are exact f64s, so one multiply or divide performs a single
  correct rounding, bit-identical to a full parser. Everything else
  (longer mantissas, larger exponents, `inf`/`nan` spellings, malformed
  cells) falls back to `str::parse`, which *is* Eisel-Lemire in std since
  Rust 1.55 — the fallback is rarely taken, not slow. A from-scratch
  Eisel-Lemire would only duplicate std; the remaining headroom is SWAR
  digit scanning in the fast path, tracked as an issue.

`parse_columns_par` reuses the `parse_par` tape, converts disjoint record
ranges on separate threads, then concatenates column chunks; validity
bitmaps are stitched with a bit shift, so chunk row counts need not be
multiples of 64.

### Parallel Variants

For doubled-quote dialects (no backslash escapes, so quote parity is independent):
- `index_structurals_par(data, threads, out)`: Counting prepass over chunks to extract per-chunk quote parity, then parallel indexing with `index_structurals_seeded()`.
- `parse_par(data, threads)`: Parallel tape building; end entries carry cumulative separator counts, so chunk tapes concatenate with one add per end entry.
- Output is byte-identical to serial (tested across thread counts).

## Testing Strategy

**Layered differential testing**: every kernel must agree at every layer.

1. **Scalar reference** (`scalar::index_structurals_spec()`): The ground truth, byte-at-a-time baseline.
2. **IR interpreter** (`interp::run()`): Reference implementation of the bitstream algebra.
3. **Generated AVX2 kernel**: The fast path.
4. **Generated scalar fallback**: Portable, same IR semantics as AVX2 but portable primitives.
5. **Oracle**: For CSV, cross-validated against the `csv` crate for real-world data.

Tests cover:
- **Randomized differential** (`tests/codegen.rs`): 800+ random inputs per format with controlled alphabets (structurals, quotes, escapes, filler).
- **Drift test** (`tests/codegen.rs::generated_kernels_match_codegen`): Checked-in kernels match current codegen output.
- **IR block-boundary cases** (`tests/ir.rs`): Carries across 64-byte seams (quote parity, backslash runs).
- **Span API** (`tests/spans.rs`): Quote stripping, escape resolution, parallel iteration.
- **Typed columns** (`tests/columns.rs`): A dumb scalar reference (quote-parity split + `str::parse` per cell) must agree with `parse_columns` — values, placeholders, and validity bitmaps, f64 compared by bit pattern — on thousands of randomized inputs including quoted/escaped cells and block-boundary placements; `parse_columns_par` must equal serial for several thread counts.
- **Hand-picked cases** (`src/lib.rs`): Escaped quotes, unclosed strings, edge cases.

## Invariants for Contributors

1. **Regenerate checked-in kernels**: After any change to IR, codegen, or format definitions, run `cargo run --example generate` and commit the regenerated files in `src/kernels/`. The drift test enforces this.
2. **Every new IR op** must include:
   - Definition in `ir.rs` enum.
   - Builder method on `Graph`.
   - Evaluation in `interp.rs` with correct carry threading.
   - Codegen emission in both `Flavor::Avx2` and `Flavor::Fallback` branches in `codegen.rs`.
   - Carry-slot allocation if stateful.
   - Differential tests covering block boundaries.
3. **Semantics changes**: The scalar reference is ground truth. Update `scalar.rs` first, then verify all three executors agree via differential tests.
4. **Zero-warning policy**: All generated code must compile warning-free.
