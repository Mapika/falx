# falx

falx is a parser **generator** that takes a declarative format specification
and emits simdjson-style branchless SIMD parsing kernels. Hand-written SIMD
parsers (simdjson, simdzone, simdutf, Sep) each take an expert months; the
techniques they share are mechanical enough to compile from a spec. No other
such generator currently exists.

```
$ cargo run --features cli --bin falx -- build specs/logfmt.toml -o logfmt_parser.rs
```

The output is a single self-contained Rust file (std only): native x86 SIMD
structural indexers with AVX-512F/BW/VL+PCLMULQDQ preferred over
AVX2+PCLMULQDQ at runtime, plus a zero-copy record/field API — ready to
drop into any project:

```rust
let parsed = logfmt_parser::parse(&data);
for record in parsed.records() {
    for field in record.fields() {   // Cow<[u8]>: unquoted, unescaped,
        handle(&field);              // borrows unless an escape forced a copy
    }
}
```

Specs can also declare **typed columns**, and the generated parser then
emits columnar buffers directly — CSV to Arrow-style typed arrays in one
pass, skipping every undeclared field:

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
48 threads, 8-channel DDR5), 64 MiB synthetic data per format, best of 7
runs in one session; the AVX-512 path is selected at runtime.

**Structural indexing** — find every field and record boundary. This is less
work than the baselines, which also materialize values, so these speedups
show the headroom indexing creates rather than a like-for-like parse:

| format | falx serial | falx parallel | ecosystem baseline |
|---|---|---|---|
| CSV | 5.8 GiB/s | **36 GiB/s** | csv crate full parse 0.26 |
| TSV | 7.2 GiB/s | **42 GiB/s** | — |
| logfmt | 5.1 GiB/s | — | — |
| NDJSON framing | 5.1 GiB/s | — | serde_json 0.15 (34x), simd-json tape 0.19 (27x) |

**Like-for-like** — full record/field iteration with quote stripping and
unescaping on both sides, byte-identical output to the csv crate:

| | throughput | vs csv crate |
|---|---|---|
| falx `parse()` + field iteration | 0.83 GiB/s | 3.1x |
| falx `parse_into()` + fields (recycled tape) | 1.16 GiB/s | 4.3x |
| falx `parse_par()` + parallel fields (48 threads) | **9.6 GiB/s** | **36x** |
| falx `stream()` incremental, 64 KiB feeds | 1.08 GiB/s | 4.0x |
| falx `fields_raw()` (zero-copy spans, no cleaning) | **7.8 GiB/s** | — |
| csv crate `byte_records()` | 0.27 GiB/s | 1.0x |

The recycled-tape row is the steady-state number (a fresh ~40 MB tape per
parse soft-page-faults at GiB/s, so batch callers hand the previous parse
back via `parse_into`). `fields_raw()` is the zero-copy fast path for
callers that don't need quote stripping (~6x the cleaning path). Parallel
rows use all 48 threads; the csv crate and arrow-csv are single-threaded, so
the single-core columns are the fairest "we're faster" figures and the
parallel columns are the end-to-end throughput.

Parallelism falls out of the tape design plus a **speculative** entry-state
trick: each chunk is indexed as if it began outside any quoted region, and
its quote parity falls out of the kernel's final carry — so there is *no
prepass over the data*; only the rare chunk that truly began mid-quote is
re-indexed. Chunk tapes then scatter into one master tape concurrently
(replacing a single-threaded merge). All std-only; byte-identical to serial,
differentially tested per thread count with quoted regions spanning every
chunk boundary.

### Typed columns: CSV → Arrow-layout buffers

The projection benchmark — extract Latitude/Longitude as `f64` columns
with validity bitmaps, skipping the other five fields, 64 MiB synthetic:

| | throughput | vs csv crate |
|---|---|---|
| falx `parse_columns` | 0.77 GiB/s | 3.3x |
| falx `parse_columns_par` (48 threads) | **5.9 GiB/s** | **26x** |
| csv crate + `str::parse` | 0.23 GiB/s | 1.0x |
| arrow-csv (projection enabled) | 0.29 GiB/s | 1.3x |

All four contenders agree exactly on valid-row counts and value checksums
(`cargo run --release --example bench_columns`). The output layout *is*
the Arrow primitive-array layout, so handing columns to Arrow is a buffer
wrap — `examples/arrow_interop.rs` does it without copying either buffer.

The columnar path is *fused*: structural masks feed a projection sink
directly, so no tape is materialized and undeclared fields cost one
counter increment each. Parallel workers reconcile chunk boundaries by
terminator ownership rather than tape splitting (see ARCHITECTURE.md) —
removing the tape pass is what took parallel extraction from 1.9 to
3+ GiB/s.

Text projects too: `string` columns materialize cleaned cells (quotes
stripped, escapes resolved) straight into Arrow varbinary buffers during
the same pass. Extracting *City as string + lat/lon as f64* runs at
0.64 GiB/s serial / 2.15 GiB/s parallel vs the csv crate's 0.22 and
arrow-csv's 0.27 — same `bench_columns` example, city byte totals matching
across contenders. (arrow-csv is run with projection enabled, its
like-for-like configuration, but still materializes through its own record
reader; serial falx here is conversion-bound — float parsing runs ~8 ns/cell
at the scalar frontier.)

### Beyond CSV/JSON: formats researchers actually use

The generator is the point — a new delimited or record format is a few
lines of spec, not months of expert SIMD — so the same engine that does CSV
does the formats genomics, ML, and scientific pipelines run on:

| format | falx | baseline |
|---|---|---|
| **VCF / BED / GFF / SAM** (tab + `#`, genomics) | 1.6 serial, **8.6 parallel** GiB/s | scalar readers ~1–2 GiB/s |
| **FASTQ** (4-line reads, genomics) | **~21 GiB/s** framed (newline index 28–36) | scalar reader 2.5 GiB/s |
| Matrix Market (sparse sci-compute) | generated from a 4-line spec | — |

VCF/BED/GFF/SAM are tab-delimited with `#` header lines — a
comment-without-quote dialect, which now parallelizes by *line ownership*
(each worker starts on a fresh line, so comment-region state never crosses a
chunk boundary). FASTQ packs each read into 4 lines (`@header` / sequence /
`+` / quality), and the quality line can contain `@` and `+`, so framing
must count line boundaries rather than split on a sigil — falx finds them
with the generated newline kernel at memory speed and groups them by 4
(`examples/fastq.rs`), **~8–18x a scalar reader**, sequence-byte checksums
identical. (Comment dialects still run their three-state region resolution
scalar; vectorizing it is the next lever.)

### Versus the real simdjson (C++)

The following table is from the earlier Core Ultra 9 285H laptop session
(simdjson C++ has not been rebuilt on the Xeon); the *relative* standing is
the point. simdjson 4.6.4 (the C++ original, haswell kernel,
g++ -O3 -march=native), same machine, byte-identical NDJSON, document counts
matching exactly:

| | throughput |
|---|---|
| falx NDJSON framing kernel | **6.61 GiB/s** |
| falx JSON nested tape (`parse_nested_into`, recycled) | **2.84 GiB/s** |
| simdjson `iterate_many`, count documents | 2.75 GiB/s |
| simdjson `iterate_many`, read one field per doc | 2.48 GiB/s |
| falx JSON nested tape, fresh allocation per call | 1.40 GiB/s |
| simd-json (Rust port), full tape per document | 0.41 GiB/s |

The nested rows are the M7 bracket-matched tape (structure only — matched
brackets and separator positions, values sliced lazily), built fused from
the kernel's masks with no intermediate position vector; serde_json parses
the same stream of documents at 0.30 GiB/s.

A parallel builder exists (`parse_nested_par[_into]`): a serial prepass
replays the kernel to hand each chunk its exact entry carries and tape
slot range, chunks then write globally-indexed entries into disjoint
ranges of one master tape, and the few brackets crossing chunk boundaries
reconcile through an ordered residue merge. Stated plainly: on this WSL2
machine it only ties serial (~3.4 GiB/s at x16) — dense nested parsing
saturates memory bandwidth here, and the serial prepass caps sparse
inputs. The prepass is the part
[#6](https://github.com/Mapika/falx/issues/6) removes: with parallel
entry-state reconstruction for backslash dialects, chunks seed themselves
and the serial pass disappears.

Caveat, stated plainly: simdjson parses document internals while falx
frames records and slices fields lazily — the gap is what skipping
in-document parsing buys on record-streaming workloads. The C++ benchmark
lives in this repo's history (`/tmp/simdjson_bench` recipe in the commit
message) and is reproducible with the released amalgamation.

### Parallel structural indexing

Parallel indexing of quoted formats is normally blocked by quote context
(a chunk can't know if it starts inside a string). falx indexes each chunk
**speculatively** as if it began outside any quoted region; the chunk's
quote parity is the kernel's final carry, so a prefix-XOR of carries gives
each chunk's true entry state with no prepass over the data — only the rare
chunk that actually began mid-quote (a quoted field spanning the boundary)
is re-indexed. Comment dialects (genomics: tab/space + `#`) instead chunk by
**line ownership** — each worker starts on a fresh line, so region state
never crosses a boundary. Both are byte-identical to serial, tested across
thread counts with quoted/comment regions spanning every chunk boundary.

Earlier versions ran a separate counting prepass (a full extra read of the
data) and merged chunk tapes single-threaded; dropping both — speculative
entry-state plus a parallel scatter — is what took CSV parallel indexing
from ~13 to ~36 GiB/s on this box.

Correctness: every kernel is differential-tested — generated SIMD dispatch,
the IR interpreter, and an independent byte-at-a-time reference must agree
bit-for-bit across thousands of randomized inputs, including escape runs
and quotes spanning 64-byte block boundaries. The benchmark additionally
cross-checks structural counts between implementations at runtime.

## How it works

A format spec compiles to a graph in a small **bitstream IR** (`src/ir.rs`):
operations over bit vectors with one bit per input byte, executed 64 bytes
per step. The algebra is small — character-class membership (byte compares
or PSHUFB nibble tables for big classes), the four bitwise ops,
`ShiftLeft1` ("previous byte matched", one carried bit), `PrefixXor`
(running parity via carry-less multiply — the quote-context trick),
carry-propagating `Add` (odd/even run detection, the simdjson
backslash-escape trick), and one deliberate exception to bit-parallelism:
`Regions`, an event-walking three-state resolver that lets comment lines
and quoted fields interleave exactly. Bit-parallel ops map to one or two
machine instructions; stateful ops carry a few bits across blocks, which
is the kernel's entire memory — no lookback, no backtracking, no
allocation.

Three executors share these semantics:

- `src/interp.rs` — reference interpreter (ground truth, deliberately slow)
- `src/codegen.rs` — emits the self-contained Rust kernel file
- `src/avx2.rs` — the original hand-written CSV kernel, kept as the fidelity
  baseline

Format specs (TOML, see `specs/`) currently describe the *delimited* family:
a structural byte set, an optional quote byte, an escape convention
(RFC 4180 doubled quotes or JSON-style backslash), and an optional
line-start comment byte (`comment = "#"` skips comment lines exactly,
quotes-in-comments and comments-in-quotes included). Specs can also declare
nesting bracket pairs (`nesting = ["{}", "[]"]`), which adds a `parse_nested`
API: the structural index feeds a bracket-matching tape with O(1) container
skips and a zero-copy navigation API; the generated JSON structural parser
(`specs/json.toml`) demonstrates this, differentially tested against
serde_json. That one family already covers CSV dialects (with or without `#`
comments), TSV, logfmt, NDJSON record framing, and separator-rich formats
with arbitrarily large structural byte sets.

Generated parsers also include a streaming API for unbounded input (pipes,
log tails, larger-than-RAM files): `stream()` accepts arbitrary chunks and
emits complete records through a callback. Kernel state carries across
feeds, so quoted regions and escape runs split across chunk boundaries are
handled exactly — and the small hot working set makes it faster than
single-threaded batch parsing. Stream-vs-batch equivalence is
differentially tested down to 1-byte feeds, including forced compaction.

## falx-synth: the generator discovers its own kernels

`src/synth.rs` inverts the generation direction. Instead of compiling a fixed set of known bit-parallel tricks, it searches for them: given a byte-at-a-time reference implementation (the state machine a person would naively write), it enumerates the bitstream IR bottom-up to find an equivalent branchless graph. The search uses observational-equivalence dedup (terms are pruned if their behavior on a corpus equals an earlier term), CEGIS verification loops (a differential mismatch on fresh random inputs becomes a new corpus entry and the search restarts), and cost-weighted settling (the cheapest verified form, not the first match).

The search includes automatic abstraction discovery: when a round of enumeration exhausts, banked terms are scored by gate (precision × recall against the target bits), generativity (how many novel terms build on them), and near-miss subterm frequency, and the best are promoted to leaves for the next round; single-hole templates mined by anti-unification join the grammar as ops. Enumeration itself runs sharded across threads with a deterministic merge.

Demo: `cargo run --release --example synth_demo`.

Starting from only the escape-byte class and the even-position constant, the system re-derived the simdjson odd-backslash-run escape trick by inventing its own intermediate abstractions. It then found a 9-node form (two carried states) that beats the 16-node hand derivation falx originally shipped — that form now is the `escaped_positions` kernel in `src/formats.rs`, feeding JSON, NDJSON, and logfmt.

Verification is exhaustive. Every candidate passes 4,000-input differential CEGIS verification and 50,000-input differential comparison with the hand graph. Beyond that, every IR op has an exact byte-serial form carrying at most one bit of state, so a graph is a finite automaton over bytes. Complete equivalence is proven via product-automaton reachability against the spec machine — equality for all inputs, no SMT solver. The escape kernel's proof has 224 product states.

Caveats, stated plainly. An earlier discovery (a PrefixXor-based 9-node form) was 5–8% slower in the kernels despite being smaller; PrefixXor is a carry-less multiply on AVX2. The search now optimizes a per-backend cost model instead of node count. Throughput of the escape kernels is flat either way on this WSL2 box (memory-bandwidth-bound), so the practical win is smaller generated code and one fewer carried state, not GiB/s. With don't-care masks (the stream is only read at quote bytes), the search found a 6-node form. The prover also certifies impossibilities: CR-before-LF needs one byte of lookahead, every IR op is causal, so no graph of any size computes it ([#3](https://github.com/Mapika/falx/issues/3) context).

Multi-output synthesis solves several specs against one corpus, later outputs reusing earlier ones and merging into a shared-CSE graph. Before native code emission, the selected `DelimitedParts` pass through a deterministic cost-weighted graph optimizer: it prunes unreachable nodes, canonicalizes commutative bit ops, folds boolean identities, and remaps the structural/terminator/nesting output roles onto the optimized graph. Codegen accepts the rewrite only when the AVX2 cost model is lower, so equal-cost normalizations do not perturb instruction order in generated kernels.

## Building on falx

The intended integration is a build script: keep a `spec.toml` in your
repo, generate the parser at compile time, ship no falx dependency at
runtime (the generated file is std-only):

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

A complete runnable version lives in `examples/build-integration/`. The
spec gallery in `specs/` covers CSV (comma and semicolon dialects), TSV,
PSV, logfmt, and NDJSON framing; a new delimited format is usually a
four-line TOML file.

Want to extend falx itself? `ARCHITECTURE.md` explains the bitstream IR
and the generated-kernel anatomy; `CONTRIBUTING.md` has recipes for adding
a format preset (~20 lines), an IR op, or a backend — the
[issue tracker](https://github.com/Mapika/falx/issues) has scoped starter
projects, including the ARM NEON backend (CI already verifies ARM
correctness, so it's pure speed work).

## Running

```
cargo test                          # differential + drift tests
cargo run --release --example bench # multi-format throughput benchmark
cargo run --release --example bench_columns  # typed-column extraction vs csv/arrow-csv
cargo run --example generate        # regenerate src/kernels/ from dialects
cargo run --features cli --bin falx -- build specs/csv-typed.toml -o parser.rs
```

Kernel generation defaults to weighted auto-discovery plus cost-weighted graph optimization for supported dialects; unsupported dialects such as comment-region CSV stay on the handwritten graph path but still use the optimizer before emission. To force handwritten graphs for every target, run `cargo run --example generate -- --manual`.

## Roadmap

- M0 (done): hand-written AVX2 CSV structural indexer, benchmark methodology
- M1 (done): bitstream IR + interpreter, differential fuzzing harness
- M2 (done): Rust codegen from the IR — generated CSV kernel within 2% of
  hand-written
- M3 (done): declarative TOML spec + CLI emitting self-contained parser files
- M4 (done): escape machinery (`Add`/`Const` ops) and the wider delimited
  family: TSV, logfmt, NDJSON framing
- M5 (done): records/fields span API emitted into generated parsers — lazy
  spans, quote stripping, `Cow`-based unescaping that allocates only when an
  escape is present
- M6 (done): typed projection — specs declare typed columns, parsers emit
  Arrow-layout columnar buffers (values + validity bitmap), SWAR int and
  Clinger-fast-path float parsing, serial and parallel
- M7 (done): nested structure — specs declare bracket pairs, generated parsers
  add a matched-bracket nested tape with O(1) container skips and a navigation
  API; JSON structural parsing differentially tested against serde_json
- M8 (done): parallelism overhaul — parallel scatter merge, speculative
  entry-state (no quote-parity prepass), `fields_raw()` zero-copy iterator;
  CSV parallel indexing ~13 → ~36 GiB/s on the Sapphire Rapids box
- M9 (done): comment dialects parallelize by line ownership — genomics and
  scientific formats (VCF/BED/GFF/SAM, Matrix Market) get the parallel path;
  FASTQ framing (4-line reads) via the newline kernel (`examples/fastq.rs`)
- Next: vectorize the three-state region resolver (the scalar `Regions` pass
  caps comment dialects — the genomics speed lever), per-field clean/Cow cost
  (~2.5 ns/field span-layer headroom), `lines_per_record` so fixed-line
  formats like FASTQ get a generated record API, ARM NEON backend, full
  equality-saturation graph extraction over the local cost-weighted optimizer

## License

MIT.
