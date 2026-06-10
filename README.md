# falx

falx is a parser **generator** that takes a declarative format specification
and emits simdjson-style branchless SIMD parsing kernels. Hand-written SIMD
parsers (simdjson, simdzone, simdutf, Sep) each take an expert months; the
techniques they share are mechanical enough to compile from a spec. No other
such generator currently exists.

```
$ cargo run --features cli --bin falx -- build specs/logfmt.toml -o logfmt_parser.rs
```

The output is a single self-contained Rust file (std only): an
AVX2+PCLMULQDQ structural indexer with a portable scalar fallback, runtime
dispatch, and a zero-copy record/field API — ready to drop into any project:

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
type = "f64"      # also: "i64", "bytes" (zero-copy spans)
name = "latitude"
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

| format | generated kernel | generated scalar fallback | ecosystem baseline |
|---|---|---|---|
| CSV | 5.61 GiB/s serial, **9.14 GiB/s parallel x16** | 0.67 GiB/s | csv crate 0.47 GiB/s |
| TSV | 7.33 GiB/s serial, **13.17 GiB/s parallel x16** | 0.98 GiB/s | — |
| logfmt | **4.99 GiB/s** | 0.42 GiB/s | — |
| NDJSON framing | **6.61 GiB/s** | 0.76 GiB/s | serde_json 0.24 GiB/s (27x slower), simd-json tape 0.41 GiB/s (16x slower) |

Structural indexing is less work than the baselines do (they materialize
values), so those speedups show headroom. The **like-for-like** comparison —
full record/field iteration with quote stripping and unescaping on both
sides, byte-identical output — is:

| | throughput | speedup |
|---|---|---|
| falx `parse()` + field iteration | 0.97 GiB/s | 2.1x |
| falx `parse_par()` + parallel fields (16 threads) | **4.80 GiB/s** | **10.3x** |
| falx `stream()` incremental, 64 KiB feeds | 1.29 GiB/s | 2.8x |
| csv crate `byte_records()` | 0.46 GiB/s | 1.0x |

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
| falx `parse_columns` | 0.82 GiB/s | 0.98 GiB/s |
| falx `parse_columns_par` (16 threads) | **1.45 GiB/s** | **1.94 GiB/s** |
| csv crate + `str::parse` | 0.41 GiB/s | 0.53 GiB/s |
| arrow-csv (projection enabled) | 0.49 GiB/s | 0.62 GiB/s |

All four contenders agree exactly on valid-row counts and value checksums
(`cargo run --release --example bench_columns`). The output layout *is*
the Arrow primitive-array layout, so handing columns to Arrow is a buffer
wrap — `examples/arrow_interop.rs` does it without copying either buffer.

Caveats, stated plainly: arrow-csv is benchmarked with projection enabled
(its like-for-like configuration) but still materializes through its own
record reader; the parallel speedup tops out near 2x here because the
structural tape build is already memory-bandwidth-bound on this machine
(stage breakdown: on worldcitiespop, 60 ms of the 144 ms serial total is
tape construction, and 16 hyperthreads lose to 8 cores). Float conversion
costs ~12 ns/cell via the Clinger fast path; SWAR digit scanning
([#8](https://github.com/Mapika/falx/issues/8)) is the known headroom.

### Versus the real simdjson (C++)

simdjson 4.6.4 (the C++ original, haswell kernel, g++ -O3 -march=native),
same machine, byte-identical NDJSON, document counts matching exactly:

| | throughput |
|---|---|
| falx NDJSON framing kernel | **6.61 GiB/s** |
| simdjson `iterate_many`, count documents | 2.75 GiB/s |
| simdjson `iterate_many`, read one field per doc | 2.48 GiB/s |
| simd-json (Rust port), full tape per document | 0.41 GiB/s |

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

Correctness: every kernel is differential-tested — generated AVX2, generated
scalar fallback, the IR interpreter, and an independent byte-at-a-time
reference must agree bit-for-bit across thousands of randomized inputs,
including escape runs and quotes spanning 64-byte block boundaries. The
benchmark additionally cross-checks structural counts between
implementations at runtime.

## How it works

A format spec compiles to a graph in a small **bitstream IR** (`src/ir.rs`):
operations over bit vectors with one bit per input byte, executed 64 bytes
per step. The whole algebra is seven ops — character-class membership,
the four bitwise ops, `ShiftLeft1` ("previous byte matched", one carried
bit), `PrefixXor` (running parity via carry-less multiply — the quote-context
trick), and carry-propagating `Add` (odd/even run detection, the
simdjson backslash-escape trick). Each op maps to one or two machine
instructions; stateful ops carry a few bits across blocks, which is the
kernel's entire memory — no lookback, no backtracking, no allocation.

Three executors share these semantics:

- `src/interp.rs` — reference interpreter (ground truth, deliberately slow)
- `src/codegen.rs` — emits the self-contained Rust kernel file
- `src/avx2.rs` — the original hand-written CSV kernel, kept as the fidelity
  baseline

Format specs (TOML, see `specs/`) currently describe the *delimited* family:
a structural byte set, an optional quote byte, and an escape convention
(RFC 4180 doubled quotes or JSON-style backslash). That one family already
covers CSV dialects, TSV, logfmt, and NDJSON record framing.

Generated parsers also include a streaming API for unbounded input (pipes,
log tails, larger-than-RAM files): `stream()` accepts arbitrary chunks and
emits complete records through a callback. Kernel state carries across
feeds, so quoted regions and escape runs split across chunk boundaries are
handled exactly — and the small hot working set makes it faster than
single-threaded batch parsing. Stream-vs-batch equivalence is
differentially tested down to 1-byte feeds, including forced compaction.

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
- Next: faster span walking (the scalar record/field layer now dominates
  end-to-end time), shuffle-based classification (large character classes),
  comment/line-start context, ARM NEON backend, e-graph simplification of
  format graphs

## License

MIT OR Apache-2.0.
