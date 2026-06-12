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

Intel Core Ultra 9 285H (AVX2 + PCLMULQDQ, WSL2, Rust 1.93), 64 MiB
synthetic data per format, best of 7 runs:

| format | generated kernel | ecosystem baseline |
|---|---|---|
| CSV | 5.61 GiB/s serial, **9.14 GiB/s parallel x16** | csv crate 0.47 GiB/s |
| TSV | 7.33 GiB/s serial, **13.17 GiB/s parallel x16** | — |
| logfmt | **4.99 GiB/s** | — |
| NDJSON framing | **6.61 GiB/s** | serde_json 0.24 GiB/s (27x slower), simd-json tape 0.41 GiB/s (16x slower) |

Structural indexing is less work than the baselines do (they materialize
values), so those speedups show headroom. The **like-for-like** comparison —
full record/field iteration with quote stripping and unescaping on both
sides, byte-identical output — is:

| | throughput | speedup |
|---|---|---|
| falx `parse()` + field iteration | 0.93 GiB/s | 2.1x |
| falx `parse_into()` + fields, recycled tape buffers | 1.39 GiB/s | 3.2x |
| falx `parse_par()` + parallel fields (16 threads) | **4.27 GiB/s** | **9.8x** |
| falx `stream()` incremental, 64 KiB feeds | 1.47 GiB/s | 3.4x |
| csv crate `byte_records()` | 0.44 GiB/s | 1.0x |

The recycled-tape row is the steady-state number: at GiB/s, the soft page
faults of allocating ~40 MB of fresh tape per parse are a measurable share
of the run, so batch callers should hand the previous parse back via
`parse_into` (streaming reuses its buffers internally, which is why it
matches). All rows above are from one session on the same machine.

On real data (worldcitiespop.csv, 145 MB, the csv crate's canonical
benchmark file): indexing 2.49 GiB/s, single-threaded field iteration
1.33 GiB/s vs the csv crate's 0.78 — with field byte totals matching the
csv crate exactly.

Parallelism falls out of the tape design: the record tape's end entries
carry cumulative separator counts, so `records_range(a..b)` yields disjoint
O(1) chunks for threads to walk, and `parse_par(data, threads)` builds the
tape itself in parallel — chunk tapes concatenate directly, with one add
per end entry to rebase counts. All std-only; output is byte-identical to
the serial path (differentially tested per thread count).

### Typed columns: CSV → Arrow-layout buffers

The projection benchmark — extract Latitude/Longitude as `f64` columns
with validity bitmaps, skipping the other five fields — median of 9 runs
(best runs are 2–5% faster; WSL2 is memory-bandwidth-limited, so medians
are the honest figure):

| | 64 MiB synthetic | worldcitiespop.csv (144 MiB) |
|---|---|---|
| falx `parse_columns` | 1.05 GiB/s | 1.28 GiB/s |
| falx `parse_columns_par` (16 threads) | **4.87 GiB/s** | **3.05 GiB/s** |
| csv crate + `str::parse` | 0.42 GiB/s | 0.52 GiB/s |
| arrow-csv (projection enabled) | 0.49 GiB/s | 0.61 GiB/s |

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
the same pass. Extracting *City as string + lat/lon as f64* from
worldcitiespop runs at 1.07 GiB/s serial / 1.79 GiB/s parallel vs the
csv crate's 0.48 and arrow-csv's 0.56 — same `bench_columns` example,
city byte totals matching across contenders.

Caveats, stated plainly: arrow-csv is benchmarked with projection enabled
(its like-for-like configuration) but still materializes through its own
record reader; serial falx is now conversion-bound — float parsing runs
~8 ns/cell at the scalar frontier, and SWAR digit scanning was prototyped
and measured *slower* on real short-mantissa data (closed
[#8](https://github.com/Mapika/falx/issues/8) has the numbers). The
144 MiB real file parallelizes to 3.05 GiB/s versus the 64 MiB synthetic
input's 4.87 — the larger working set runs into WSL2's memory-bandwidth
ceiling sooner.

### Versus the real simdjson (C++)

simdjson 4.6.4 (the C++ original, haswell kernel, g++ -O3 -march=native),
same machine, byte-identical NDJSON, document counts matching exactly:

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
(a chunk can't know if it starts inside a string). For doubled-quote
dialects the entry state collapses to one bit — the parity of quote bytes
before the chunk — so `index_structurals_par` runs a trivially parallel
counting prepass, prefix-combines parities serially (nanoseconds), then
indexes all chunks concurrently. Output is byte-identical to the serial
indexer (tested across thread counts with quoted regions spanning every
chunk boundary).

Codegen fidelity: after two-block unrolling in the emitter, the generated
CSV kernel outruns the hand-written kernel it was modeled on
(5.61 vs 5.12 GiB/s).

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

Multi-output synthesis solves several specs against one corpus, later outputs reusing earlier ones and merging into a shared-CSE graph.

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

Kernel generation defaults to the weighted auto-discovery path for supported dialects; unsupported dialects such as comment-region CSV stay on the handwritten graph path. To force handwritten graphs for every target, run `cargo run --example generate -- --manual`.

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
- Next: parallel entry-state for backslash dialects
  ([#6](https://github.com/Mapika/falx/issues/6) — removes the parallel
  nested builder's serial prepass), per-field clean/Cow cost (the
  remaining span-layer headroom, ~2.5 ns/field), configurable record
  terminators ([#3](https://github.com/Mapika/falx/issues/3), in
  progress), ARM NEON backend, cost-weighted e-graph simplification of format graphs (the synthesizer's cost models are the seed)

## License

MIT.
