# vexel

vexel is a parser **generator** that takes a declarative format specification
and emits simdjson-style branchless SIMD parsing kernels. Hand-written SIMD
parsers (simdjson, simdzone, simdutf, Sep) each take an expert months; the
techniques they share are mechanical enough to compile from a spec. No other
such generator currently exists.

```
$ cargo run --features cli --bin vexel -- build specs/logfmt.toml -o logfmt_parser.rs
```

The output is a single self-contained Rust file (std only): an
AVX2+PCLMULQDQ structural indexer with a portable scalar fallback and
runtime dispatch, ready to drop into any project.

## Performance

Intel Core Ultra 9 285H (AVX2 + PCLMULQDQ, WSL2, Rust 1.93), 64 MiB
synthetic data per format, best of 7 runs:

| format | generated kernel | generated scalar fallback | ecosystem baseline |
|---|---|---|---|
| CSV | **5.23 GiB/s** | 0.66 GiB/s | csv crate 0.47 GiB/s (11.1x slower) |
| TSV | **6.66 GiB/s** | 0.98 GiB/s | — |
| logfmt | **4.28 GiB/s** | 0.42 GiB/s | — |
| NDJSON framing | **6.14 GiB/s** | 0.75 GiB/s | serde_json 0.26 GiB/s (23.6x slower) |

Two honesty notes. The baselines do more work (field/value
materialization) — the comparison shows the headroom structural indexing
creates, not a like-for-like parse. And the codegen fidelity check: the
generated CSV kernel runs within 2% of the hand-written kernel it was
modeled on (5.23 vs 5.32 GiB/s).

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

## Running

```
cargo test                          # differential + drift tests
cargo run --release --example bench # multi-format throughput benchmark
cargo run --example generate        # regenerate src/kernels/ from dialects
cargo run --features cli --bin vexel -- build specs/csv.toml -o parser.rs
```

## Roadmap

- M0 (done): hand-written AVX2 CSV structural indexer, benchmark methodology
- M1 (done): bitstream IR + interpreter, differential fuzzing harness
- M2 (done): Rust codegen from the IR — generated CSV kernel within 2% of
  hand-written
- M3 (done): declarative TOML spec + CLI emitting self-contained parser files
- M4 (done): escape machinery (`Add`/`Const` ops) and the wider delimited
  family: TSV, logfmt, NDJSON framing
- Next: shuffle-based classification (large character classes), multiple
  output streams (field spans, not just positions), comment/line-start
  context, ARM NEON backend, e-graph simplification of format graphs

## License

MIT OR Apache-2.0.
