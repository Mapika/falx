# falx

falx is a parser **generator**: it takes a declarative format spec and emits
simdjson-style branchless SIMD parsing kernels. Hand-written SIMD parsers
(simdjson, simdzone, simdutf, Sep) each take an expert months; the techniques
they share are mechanical enough to compile from a spec. No other such
generator currently exists.

```
$ cargo run --features cli --bin falx -- build specs/logfmt.toml -o logfmt_parser.rs
```

The output is a single self-contained Rust file (std only): native x86 SIMD
structural indexers (AVX-512F/BW/VL+PCLMULQDQ preferred over AVX2+PCLMULQDQ at
runtime) plus a zero-copy record/field API, ready to drop into any project:

```rust
let parsed = logfmt_parser::parse(&data);
for record in parsed.records() {
    for field in record.fields() {   // Cow<[u8]>: unquoted, unescaped,
        handle(&field);              // borrows unless an escape forced a copy
    }
}
```

Specs can also declare **typed columns**; the generated parser then emits
columnar buffers directly — CSV to Arrow-layout typed arrays in one pass,
skipping every undeclared field:

```toml
[[columns]]
index = 5
type = "f64"      # also: "i64", "string" (cleaned, Arrow varbinary
name = "latitude" #   layout), "bytes" (zero-copy raw spans)
```

```rust
let cols = parser::parse_columns(&data);          // or parse_columns_par
let lat: &[f64] = &cols.latitude;                 // one Vec per column
let ok = parser::bitmap_get(&cols.latitude_valid, row);  // Arrow-style
// validity bitmap: empty/malformed cells clear a bit, never panic
```

## Performance

Xeon w7-3455 (Sapphire Rapids, AVX-512F/BW/VL + PCLMULQDQ, 24 cores /
48 threads, 8-channel DDR5); **1 GiB** synthetic file per format, **median of
9 runs** (2 warmup) via `cargo run --release --example bench_sustained`, AVX-512
path selected at runtime. These kernels are execution-port/bandwidth bound and
**peak at 24 threads (physical cores)** — driving all 48 logical threads is
slower — so every parallel figure uses 24 threads. Each row is gated by an
equal-`Work` check: falx and the library it is timed against must emit identical
record/byte/checksum counters before any timing is reported, so the comparisons
are like-for-like, not a structural framer measured against a full semantic
parser.

**Same work on both sides** (enforced by the `Work` gate), 1 GiB per format:

| format (equal work) | falx serial | falx @24 cores | fastest library | per-core | @24 cores |
|---|---|---|---|---|---|
| FASTQ records | 7.8 | **111** | seq_io / needletail 2.7 | 2.8x | 40x |
| VCF typed projection | 2.0 | **26** | noodles-vcf 0.34 | 5.8x | 77x |
| TSV field bytes | 3.0 | **39** | csv crate 0.30 | 10x | 130x |
| logfmt pairs | 1.0 | **14** | logfmt-zerocopy 0.21 | 4.9x | 66x |
| CSV field bytes | 1.2 | **15** | csv crate 0.30 | 4.1x | 49x |
| CSV geo, text + lat/lon | 0.86 | **12** | arrow-csv 0.27 | 3.2x | 44x |
| CSV geo, lat/lon f64 | 1.1 | **13** | arrow-csv 0.31 | 3.4x | 43x |

Throughput in GiB/s. The libraries are single-threaded, so **per-core** is the
fairest kernel-efficiency comparison and **@24 cores** is end-to-end throughput
(kernel + parallelism). FASTQ is the toughest field — `seq_io`/`needletail` are
real SIMD-class parsers at ~2.7 GiB/s, so the 2.8x per-core win there is a
genuine kernel result; the 130x over the `csv` crate partly reflects how slow
that crate is.

**Reported separately, *not* a faster-than claim:** falx frames NDJSON lines at
8.4 GiB/s serial / 112 GiB/s @24, but that is line framing — `serde_json` (0.15)
and `simd-json` (0.24) do full DOM/tape parsing, which is not equal work.
Pure CSV structural indexing (every field + record boundary, no value work) runs
6.0 / 40 GiB/s — less work than a parse, so it is the headroom indexing creates,
not a library speedup.

The generated API also exposes `parse_into` (recycles the tape buffer — a fresh
per-GiB tape soft-page-faults, so batch callers hand the previous parse back),
a `fields_raw()` zero-copy span iterator that skips quote-stripping, and a
`stream()` incremental mode for unbounded input; see `examples/`.

### Typed columns: CSV → Arrow-layout buffers

Extract Latitude/Longitude as `f64` columns with validity bitmaps, skipping the
other five fields — a distinct column-*materialization* run (`bench_columns`,
64 MiB synthetic, parallel figure at 48 threads):

| | throughput | vs csv crate |
|---|---|---|
| falx `parse_columns` | 0.77 GiB/s | 3.3x |
| falx `parse_columns_par` | **5.9 GiB/s** | **26x** |
| csv crate + `str::parse` | 0.23 GiB/s | 1.0x |
| arrow-csv (projection enabled) | 0.29 GiB/s | 1.3x |

All four agree exactly on valid-row counts and value checksums (`cargo run
--release --example bench_columns`). The output layout *is* the Arrow
primitive-array layout, so handing columns to Arrow is a buffer wrap —
`examples/arrow_interop.rs` does it without copying. The path is *fused*:
structural masks feed a projection sink directly, no tape is materialized, and
undeclared fields cost one counter increment each. `string` columns materialize
cleaned cells (quotes stripped, escapes resolved) straight into Arrow varbinary
buffers in the same pass.

### Beyond CSV/JSON: formats researchers actually use

The generator is the point — a new delimited or record format is a few lines of
spec, not months of expert SIMD — so the same engine that does CSV does the
formats genomics, ML, and scientific pipelines run on. VCF/BED/GFF/SAM are
tab-delimited with `#` header lines (a comment dialect; parallelized by line
ownership). FASTQ packs each read into 4 lines (`@header` / sequence / `+` /
quality), and the quality line can contain `@`/`+`, so framing must count line
boundaries rather than split on a sigil — falx finds them with the generated
newline kernel and groups by 4 (`examples/fastq.rs`), 2.8x per core / 40x @24 vs
the real SIMD-class `seq_io` and `needletail`, with sequence- and quality-byte
checksums identical on all three. Matrix Market (sparse sci-compute) generates
from a 4-line spec.

CSV-with-`#`-comments (`csv_hash`) is the one dialect carrying both quotes and
comments — a quote can hide a comment-start and a multi-line quoted field can
hide a newline, so neither quote-parity nor line-ownership parallelism applies
directly. Its region resolver fast-paths comment-free blocks through a
PCLMULQDQ quote-parity (3.8 GiB/s serial), and it parallelizes via a per-chunk
region *transfer function* (~5x at 24 threads); see [Parallelism](#parallelism).

### Versus the real simdjson (C++)

From the earlier Core Ultra 9 285H laptop session (simdjson C++ not rebuilt on
the Xeon); the *relative* standing is the point. simdjson 4.6.4 (haswell kernel,
`g++ -O3 -march=native`), same machine, byte-identical NDJSON, document counts
matching exactly:

| | throughput |
|---|---|
| falx NDJSON framing kernel | **6.61 GiB/s** |
| falx JSON nested tape (`parse_nested_into`, recycled) | **2.84 GiB/s** |
| simdjson `iterate_many`, count documents | 2.75 GiB/s |
| simdjson `iterate_many`, read one field per doc | 2.48 GiB/s |
| falx JSON nested tape, fresh allocation per call | 1.40 GiB/s |
| simd-json (Rust port), full tape per document | 0.41 GiB/s |

The nested rows are the bracket-matched tape (structure only — matched brackets
and separator positions, values sliced lazily), built fused from the kernel's
masks; serde_json parses the same stream at 0.30 GiB/s. simdjson parses document
internals while falx frames records and slices fields lazily — the gap is what
skipping in-document parsing buys on record-streaming workloads. (A parallel
nested builder exists but currently only ties serial here: dense nested parsing
saturates memory bandwidth and a serial prepass caps it,
[#6](https://github.com/Mapika/falx/issues/6).)

## How it works

A format spec compiles to a graph in a small **bitstream IR** (`src/ir.rs`):
operations over bit vectors with one bit per input byte, executed 64 bytes per
step. The algebra is small — character-class membership (byte compares or PSHUFB
nibble tables for big classes), the four bitwise ops, `ShiftLeft1` ("previous
byte matched", one carried bit), `PrefixXor` (running parity via carry-less
multiply — the quote-context trick), carry-propagating `Add` (odd/even run
detection, the simdjson backslash-escape trick), and one three-state resolver,
`Regions`, that lets comment lines and quoted fields interleave exactly (its
comment-free common path is bit-parallel via a PCLMULQDQ prefix-XOR; only
genuinely interleaved blocks walk events). Bit-parallel ops map to one or two
machine instructions; stateful ops carry a few bits across blocks — the kernel's
entire memory, no lookback, no backtracking, no allocation.

Three executors share these semantics:

- `src/interp.rs` — reference interpreter (ground truth, deliberately slow)
- `src/codegen.rs` — emits the self-contained Rust kernel file
- `src/avx2.rs` — the original hand-written CSV kernel, kept as the fidelity
  baseline

Format specs (TOML, see `specs/`) describe the *delimited* family: a structural
byte set, an optional quote byte, an escape convention (RFC 4180 doubled quotes
or JSON-style backslash), and an optional line-start comment byte
(`comment = "#"` skips comment lines exactly, quotes-in-comments and
comments-in-quotes included). Specs can also declare nesting bracket pairs
(`nesting = ["{}", "[]"]`), adding a `parse_nested` API: the structural index
feeds a bracket-matching tape with O(1) container skips and a zero-copy
navigation API (the generated JSON parser, `specs/json.toml`, is differentially
tested against serde_json). That one family covers CSV dialects (with or without
`#` comments), TSV, logfmt, NDJSON framing, and separator-rich formats with
arbitrarily large structural byte sets.

Generated parsers also include a streaming API for unbounded input (pipes, log
tails, larger-than-RAM files): `stream()` accepts arbitrary chunks and emits
complete records through a callback. Kernel state carries across feeds, so
quoted regions and escape runs split across chunk boundaries are handled
exactly. Stream-vs-batch equivalence is differentially tested down to 1-byte
feeds.

### Parallelism

Parallel indexing of quoted formats is normally blocked by quote context (a
chunk can't know if it starts inside a string). falx handles each dialect family
with no prepass over the data:

- **Quoted (CSV/TSV/logfmt):** index each chunk **speculatively** as if it began
  outside a quote; the chunk's quote parity is the kernel's final carry, so a
  prefix-XOR of carries gives each chunk's true entry state — only the rare
  chunk that began mid-quote is re-indexed.
- **Comment-without-quote (VCF/BED/GFF/SAM):** chunk by **line ownership** —
  each worker starts on a fresh line, so comment-region state never crosses a
  boundary.
- **Comment+quote (`csv_hash`):** the 3-state region machine isn't XOR-linear
  and quoted fields span newlines, so neither scheme applies. Each chunk
  computes its region **transfer function** (`state → state`) in parallel, the
  true entry states follow from an O(threads) serial composition, and every
  chunk is then indexed once in parallel with its known entry state.

Chunk tapes scatter into one master tape concurrently. Every scheme is
byte-identical to serial, differentially tested across thread counts with
quoted/comment regions spanning every chunk boundary. (Dropping an earlier
counting prepass for the speculative scheme was a measured interleaved-A/B win —
1.27x at 48 threads, 2.34x at 24 — with unchanged-kernel controls held flat.)

Correctness: every kernel is differential-tested — generated SIMD dispatch, the
IR interpreter, and an independent byte-at-a-time reference must agree
bit-for-bit across thousands of randomized inputs, including escape runs and
quotes spanning 64-byte block boundaries. The benchmark additionally
cross-checks structural counts between implementations at runtime.

## falx-synth: the generator discovers its own kernels

`src/synth.rs` inverts the generation direction. Instead of compiling a fixed
set of known bit-parallel tricks, it searches for them: given a byte-at-a-time
reference (the state machine a person would naively write), it enumerates the
bitstream IR bottom-up to find an equivalent branchless graph. The search uses
observational-equivalence dedup, CEGIS verification loops (a differential
mismatch on fresh random inputs becomes a new corpus entry and the search
restarts), and cost-weighted settling (the cheapest verified form, not the first
match). It includes automatic abstraction discovery: exhausted rounds promote
high-scoring banked terms to leaves and mine single-hole templates by
anti-unification. Enumeration runs sharded across threads with a deterministic
merge. Demo: `cargo run --release --example synth_demo`.

Starting from only the escape-byte class and the even-position constant, the
system re-derived the simdjson odd-backslash-run escape trick by inventing its
own intermediate abstractions, and found a 9-node form (two carried states) that
beats the 16-node hand derivation falx originally shipped — now the
`escaped_positions` kernel in `src/formats.rs`, feeding JSON, NDJSON, and logfmt.

Verification is exhaustive: every candidate passes 4,000-input CEGIS and
50,000-input differential comparison with the hand graph. Beyond that, every IR
op has an exact byte-serial form carrying at most one bit of state, so a graph
is a finite automaton over bytes, and complete equivalence is proven via
product-automaton reachability against the spec machine — equality for all
inputs, no SMT solver (the escape kernel's proof has 224 product states). The
prover also certifies impossibilities: CR-before-LF needs one byte of lookahead
and every IR op is causal, so no graph of any size computes it
([#3](https://github.com/Mapika/falx/issues/3)).

Multi-output synthesis solves several specs against one corpus, later outputs
reusing earlier ones in a shared-CSE graph. Before emission the selected graph
passes a deterministic cost-weighted optimizer (prune unreachable nodes,
canonicalize commutative ops, fold boolean identities); codegen accepts the
rewrite only when the AVX2 cost model is lower, so equal-cost normalizations
don't perturb instruction order.

## Building on falx

The intended integration is a build script: keep a `spec.toml` in your repo,
generate the parser at compile time, ship no falx dependency at runtime (the
generated file is std-only):

```toml
# Cargo.toml
[build-dependencies]
falx = { git = "https://github.com/Mapika/falx", features = ["spec"] }
```

```rust
// build.rs
let spec = falx::spec::parse(&std::fs::read_to_string("spec.toml")?)?;
let code = falx::codegen::emit_parser_with_columns(&spec.dialect, &spec.name, &spec.columns)?;
std::fs::write(format!("{}/parser.rs", std::env::var("OUT_DIR")?), code)?;
```

```rust
// src/main.rs
mod parser { include!(concat!(env!("OUT_DIR"), "/parser.rs")); }
```

A complete runnable version lives in `examples/build-integration/`. The spec
gallery in `specs/` covers CSV (comma and semicolon dialects), TSV, PSV, logfmt,
and NDJSON framing; a new delimited format is usually a four-line TOML file.

Want to extend falx itself? `ARCHITECTURE.md` explains the bitstream IR and the
generated-kernel anatomy; `CONTRIBUTING.md` has recipes for adding a format
preset (~20 lines), an IR op, or a backend — the
[issue tracker](https://github.com/Mapika/falx/issues) has scoped starter
projects, including the ARM NEON backend (CI already verifies ARM correctness,
so it's pure speed work).

## Running

```
cargo test                          # differential + drift tests
cargo run --release --example bench # multi-format throughput benchmark
cargo run --release --example bench_columns  # typed-column extraction vs csv/arrow-csv
cargo run --example generate        # regenerate src/kernels/ from dialects
cargo run --features cli --bin falx -- build specs/csv-typed.toml -o parser.rs
```

Kernel generation defaults to weighted auto-discovery plus cost-weighted graph
optimization for supported dialects; unsupported dialects such as comment-region
CSV stay on the handwritten graph path but still use the optimizer before
emission. To force handwritten graphs for every target, run
`cargo run --example generate -- --manual`.

## Roadmap

- M0–M3 (done): hand-written AVX2 CSV indexer + benchmark methodology;
  bitstream IR + interpreter + differential fuzzing; Rust codegen from the IR
  (within 2% of hand-written); declarative TOML spec + CLI emitting
  self-contained parser files.
- M4–M5 (done): escape machinery and the wider delimited family (TSV, logfmt,
  NDJSON framing); records/fields span API with lazy spans, quote stripping, and
  `Cow` unescaping that allocates only on a real escape.
- M6 (done): typed projection — specs declare typed columns, parsers emit
  Arrow-layout columnar buffers (values + validity bitmap), SWAR int and
  Clinger-fast-path float parsing, serial and parallel.
- M7 (done): nested structure — specs declare bracket pairs, parsers add a
  matched-bracket tape with O(1) container skips and a navigation API; JSON
  structural parsing differentially tested against serde_json.
- M8–M9 (done): parallelism — speculative entry-state (no prepass) + parallel
  scatter merge; comment-without-quote dialects parallelize by line ownership
  (VCF/BED/GFF/SAM, Matrix Market); FASTQ framing via the newline kernel.
- M10 (done): the region resolver fast-paths comment-free blocks through a
  PCLMULQDQ prefix-XOR (csv_hash serial 1.96 → 3.84 GiB/s), and `csv_hash`
  (comment+quote) gains a parallel parse via per-chunk region transfer functions
  — the last serial-only kernel now scales (~5x at 24 threads).
- Next: fold csv_hash's three transfer-function scans into one combined 3-state
  pass and add `parse_par_into` (its remaining parallel levers); per-field
  clean/Cow cost (~2.5 ns/field span-layer headroom); a declarative
  `lines_per_record` so other fixed-line formats get FASTQ's generated record
  API; ARM NEON backend; full equality-saturation graph extraction over the
  local cost-weighted optimizer.

## License

MIT.
