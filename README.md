# falx

falx is a parser generator for high-throughput record and column parsers. It
takes a declarative format spec and emits a self-contained Rust parser with
runtime SIMD dispatch:

- x86: AVX-512F/BW/VL + PCLMULQDQ first, then AVX2 + PCLMULQDQ
- ARM: NEON, with PMULL where carry-less multiply is needed
- std-only generated parser files for normal use

The goal is not another hand-written parser. The goal is to compile the same
bit-parallel tricks used by expert SIMD parsers from small specs, then reuse
them across CSV, TSV, logfmt, NDJSON, FASTQ, VCF, and related delimited formats.

```bash
cargo run --features cli --bin falx -- build specs/logfmt.toml -o logfmt_parser.rs
```

```rust
let parsed = logfmt_parser::parse(&data);
for record in parsed.records() {
    for field in record.fields() {
        handle(&field);
    }
}
```

## Current Status

On the benchmark matrix we have actually run, falx is the fastest tested
solution across CSV projection/materialization, CSV aggregation, NDJSON
schema-aware aggregation, and BGZF block streaming. The claim is deliberately
bounded: these are local, reproducible results on the datasets and libraries
listed below, not a universal claim about every parser workload.

Hardware for the cross-library runs: Xeon w7-3455, Sapphire Rapids, 24 physical
cores, 48 logical threads, AVX-512F/BW/VL + PCLMULQDQ. Parallel falx figures use
24 threads unless noted. CSV and NDJSON files are 1 GiB.

| Lane | falx | Fastest external baseline | Result |
|---|---:|---:|---:|
| CSV Latitude/Longitude materialization, chunked | 9.63 GiB/s | Polars 1.41.2: 4.64 GiB/s | 2.1x |
| CSV City + Latitude/Longitude materialization | 2.13 GiB/s | PyArrow 24.0.0: 1.78 GiB/s | 1.2x |
| CSV count/sum(Latitude)/sum(Longitude), fused | 9.80 GiB/s | Polars 1.41.2: 2.04 GiB/s | 4.8x |
| NDJSON `sum(id + nested.score)` | 38.19 GiB/s | simdjson C++: 2.76 GiB/s | 13.9x |
| BGZF block streaming, 1 GiB raw VCF | 0.074 s median | htslib bgzip: 0.10-0.12 s | ~1.4x |

The falx column above predates the 2026-08 optimization pass below and is
conservative for the CSV lanes: the same falx lanes have since improved by
+23-34% serial and up to 3.1x on parallel materialization (measured on a
different machine — see the next section). The external baselines have not been
re-run since, so the table keeps the older, matched-hardware falx numbers until
the full matrix is repeated on the reference box.

## 2026-08 Optimization Pass

falx-before vs falx-after on one machine (dual EPYC 9575F, Zen 5, 128 physical
cores, AVX-512 incl. VBMI2; 1 GiB generated datasets; parallel figures use 24
threads unless noted; `bench_columns` materialization rows use the example's
default thread count). Both builds ran the same lanes on the same box, so these
ratios are hardware-independent falx improvements:

| Lane | falx before | falx after | Delta |
|---|---:|---:|---:|
| csv_hash (`#` comments + quotes) typed columns, chunked | 1.55 GiB/s serial-only | 16.11 GiB/s | 10.4x |
| CSV City + Latitude/Longitude materialization, parallel | 5.13 GiB/s | 16.03 GiB/s | 3.1x |
| CSV Latitude/Longitude materialization, parallel | 9.36 GiB/s | 17.57 GiB/s | 1.9x |
| CSV City + Latitude/Longitude materialization, serial | 0.99 GiB/s | 1.32 GiB/s | +34% |
| CSV count/sum(lat,lon) fused, 1 thread | 1.24 GiB/s | 1.63 GiB/s | +31% |
| CSV Latitude/Longitude materialization, serial | 1.17 GiB/s | 1.43 GiB/s | +23% |
| CSV count/sum(lat,lon) fused, 24 threads | 27.44 GiB/s | 32.38 GiB/s | +18% |
| NDJSON `sum(id + nested.score)`, 24 threads | 114.5 GiB/s | 114.4 GiB/s | held |

Same-box context: the `csv` crate materializes the same columns at 0.42 GiB/s
and `arrow-csv` at 0.53 GiB/s, checksum-identical to falx on every lane.

The csv_hash row is a new capability rather than a tuning win: comment+quote
dialects previously had no fused parallel projection at all, because a chunk's
entry context there is a region (NORMAL/QUOTE/COMMENT) that no parity prefix can
recover — a `"` can hide a `#` and vice versa. They now reach the same fused
sinks through the three-phase transfer-function scheme `parse_par` already used
(each chunk's region transfer function computed in parallel, composed serially
in O(threads), then one seeded pass per chunk), so no chunk is parsed twice. At
96 threads that lane reaches 61.97 GiB/s.

What changed (all in the code generator; kernels regenerated):

- comment+quote dialects gain `parse_columns_par` /
  `parse_columns_chunks_par` via the region transfer-function scheme; their
  fused sink drivers take both entry carries packed into one seed
- fixed-shape decimal cells (`[-]d{1,3}.d{6}`) parse via one unaligned load and
  a SWAR all-digits test instead of digit-at-a-time branching, with the sign
  applied by ORing the sign bit (nonnegative mantissa) — random-sign data was
  costing ~2 branch mispredicts per row
- the general float scanner is outlined so the fixed-shape fast path stays
  icache-dense
- `parse_columns_par` scatters worker chunks into exact-sized disjoint slices
  concurrently instead of a single-threaded concat (the prior parallel-scaling
  ceiling for string columns)
- string arenas are seeded at 8 bytes/cell instead of growing from empty
- the AVX-512 structural step uses one 512-bit load and one `vpcmpb` per byte
  class instead of paired 256-bit halves
- hosts with AVX-512 VBMI2 + BMI2 (Ice Lake+, Zen 4+) get a compress-based
  drive tier: `vpcompressb` over a byte iota extracts every structural position
  of a block at once and `pext` compacts terminator flags, replacing the serial
  trailing-zero/clear-bit walk; older AVX-512 hosts keep the prior path
- short unquoted string cells append as one unconditional 16-byte copy into
  reserved spare capacity

Important benchmark boundaries:

- CSV materialization compares table output to table output. falx exposes
  `parse_columns_chunks_par` so callers can keep worker chunks, matching the way
  Polars and Arrow avoid a final flattening copy.
- CSV fused aggregation compares against database-style projected aggregation.
  falx does not build `Vec<f64>` columns for that lane.
- NDJSON is schema-aware and matches the benchmark shape: read `id` and
  `nested.score` and sum them. It is not claiming to beat a full JSON DOM parse
  while doing less work.
- BGZF streaming measures decompressed blocks delivered to a callback. The
  older fully materialized decompression path still exists when callers need one
  contiguous output buffer.

## Why It Is Fast

falx represents formats as bitstream graphs: one bit per input byte, processed
64 bytes at a time. The IR contains byte-class membership, bitwise ops,
carry-aware shifts, PCLMULQDQ prefix XOR for quote parity, and a small region
resolver for comment and quoted-field dialects.

Generated kernels do the expensive parts once:

- structural bytes and quote/comment state are computed as SIMD masks
- projected typed columns are filled directly from those masks
- undeclared CSV fields are skipped instead of parsed
- aggregate sinks can fuse parse + reduction without building a table
- parallel parsers split by record ownership, so every record is converted once
- chunked table output avoids the final memory-bandwidth-heavy concat pass

The result is a generated parser that behaves like a hand-tuned SIMD parser but
comes from a spec.

## Typed Columns

Specs can declare typed columns. The generated parser emits Arrow-compatible
value buffers and validity bitmaps directly:

```toml
[[columns]]
index = 5
name = "latitude"
type = "f64" # also: i64, string, bytes
```

```rust
let cols = parser::parse_columns(&data);
let latitudes: &[f64] = &cols.latitude;
let valid = parser::bitmap_get(&cols.latitude_valid, row);
```

For parallel table output, prefer chunks when your downstream can accept them:

```rust
let chunks = parser::parse_columns_chunks_par(&data, 24);
for chunk in &chunks {
    consume(&chunk.latitude, &chunk.latitude_valid);
}
```

`parse_columns_par` remains available and returns the legacy single `Columns`
layout; worker chunks are scattered into the exact-sized destination buffers
concurrently, so the flatten no longer serializes on one thread.

String columns are cleaned into Arrow varbinary-style buffers. Byte columns are
zero-copy raw spans into the source.

## Generated Format Coverage

The same generator covers:

- CSV/TSV/PSV-style delimited records
- quoted CSV, doubled quotes, and JSON-style escapes
- logfmt pairs
- NDJSON line records and schema-aware reductions
- nested bracket tapes for JSON-like structural navigation
- comment-line dialects such as VCF/BED/GFF/SAM
- FASTQ records via declarative `lines_per_record = 4`
- VCF typed projections, including selected INFO sub-columns

Optional modules extend this into genomics pipelines:

- `bgzf`: block-parallel BGZF inflate with pure-Rust miniz_oxide
- `bgzf-libdeflate`: the faster libdeflate backend
- fused `.vcf.gz` inflate -> parse paths that avoid materializing the full text
- Python/Arrow integration work under `python/`

## Correctness

The benchmark results are only useful because the outputs are checked.

The test suite compares generated SIMD kernels against independent references:

- scalar record/field parsers
- the bitstream IR interpreter
- codegen drift tests for checked-in generated kernels
- randomized CSV, quote, escape, comment, and nesting boundary cases
- noodles parity for VCF and BGZF genomics paths
- simdjson/Polars/PyArrow/DuckDB benchmark harness checksums where applicable

Run the main verification set:

```bash
cargo test --lib --tests
cargo test --features bgzf-libdeflate --lib --tests
```

## Reproducing The Latest Benchmark Lanes

CSV typed materialization:

```bash
cargo run --release --example bench_columns -- <csv-geo-1g.csv>
```

CSV projected aggregation:

```bash
cargo run --release --features mmap --example csv_geo_aggregate -- <csv-geo-1g.csv> 7 24
```

NDJSON schema-aware sum:

```bash
cargo run --release --example json_sum_par -- <ndjson-1g.ndjson> 9 24 full
```

BGZF decompression and streaming:

```bash
cargo run --release --features bgzf-libdeflate --example bgzf_bench -- 1024 7
```

Regenerate checked-in kernels:

```bash
cargo run --example generate
```

## Building With falx

The intended integration is a build script. Keep a spec in your repo, generate
the parser at compile time, and ship only the generated Rust file at runtime.

```toml
[build-dependencies]
falx = { git = "https://github.com/Mapika/falx", features = ["spec"] }
```

```rust
// build.rs
let spec = falx::spec::parse(&std::fs::read_to_string("spec.toml")?)?;
let code = falx::codegen::emit_parser_with_columns(
    &spec.dialect,
    &spec.name,
    &spec.columns,
)?;
std::fs::write(format!("{}/parser.rs", std::env::var("OUT_DIR")?), code)?;
```

```rust
// src/main.rs
mod parser {
    include!(concat!(env!("OUT_DIR"), "/parser.rs"));
}
```

A complete runnable version lives in `examples/build-integration/`.

## How The Generator Works

`src/ir.rs` defines the bitstream IR. `src/interp.rs` is the slow reference
interpreter. `src/codegen.rs` emits the production Rust kernels. `src/synth.rs`
can rediscover branchless kernels from byte-at-a-time reference machines using
CEGIS-style differential search and finite-automaton proof.

The synthesis path has already rediscovered and improved escape handling for
JSON/NDJSON/logfmt, and the e-graph optimizer in `src/egraph.rs` extracts
cheaper equivalent graphs before codegen. The string-template codegen is the
production path; `src/emit/` is an experimental typed-AST emitter that can render
Rust or CUDA-C from the same lowered graph.

## Development Commands

```bash
cargo test
cargo test --features bgzf-libdeflate --lib --tests
cargo run --release --example bench_sustained -- --formats csv-geo
cargo run --example generate
cargo run --features cli --bin falx -- build specs/csv-typed.toml -o parser.rs
```

Useful optional features:

- `spec`: TOML spec parsing
- `cli`: command-line parser generator
- `mmap`: memory-mapped file helpers
- `bgzf`: pure-Rust BGZF inflate
- `bgzf-libdeflate`: fastest BGZF inflate backend
- `gpu`: experimental CUDA/NVRTC backend

## Roadmap

Done:

- declarative specs for delimited formats
- generated AVX-512, AVX2, and NEON kernels
- typed CSV/VCF projections into Arrow-style buffers
- parallel parsing with quote/comment correctness
- nested JSON structural tapes
- FASTQ fixed-line records
- VCF INFO sub-column extraction
- block-parallel BGZF decompression
- fused `.vcf.gz` parse paths
- schema-aware NDJSON reductions
- chunked parallel table materialization

Experimental:

- typed-AST emitter for Rust and CUDA-C
- GPU-resident parsing/decompression/query pipelines
- broader public SOTA benchmark harness

## License

MIT.
