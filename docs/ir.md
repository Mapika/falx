# The falx IR

falx compiles a format description to a SIMD parser in two halves. The front
half (`codegen::lower`) turns a dialect into a *module*; the back half
(`codegen::emit_module`) turns a module into Rust. This document specifies the
module and its textual form, which is the interchange point between them.

The contract: **anything that can write this text reaches every backend falx
has** — AVX-512, AVX-512+VBMI2, AVX2, NEON, runtime dispatch, the span API, the
typed columnar API, and the parallel entry points. No dialect builder, no spec
file, no patching the generator.

```
    spec.toml ──lower──▶ module ──emit──▶ parser.rs
                          ▲  │
              your tool ──┘  └──▶ falx opt ──▶ module
```

## Command line

```bash
falx build     spec.toml   -o parser.rs   # both halves
falx emit-ir   spec.toml   -o module.fxir # front half only
falx opt       module.fxir -o opt.fxir    # optimize IR you got from anywhere
falx build-ir  module.fxir -o parser.rs   # back half only
falx check     spec.toml | module.fxir    # validate, write nothing
```

`--no-opt` skips the graph optimizer, which is how you see what the optimizer
did:

```bash
falx emit-ir specs/json.toml --no-opt | grep -c '^%'   # 18
falx emit-ir specs/json.toml            | grep -c '^%'   # 16
```

## Model

A module is a DAG of operations over **bitstreams**: conceptually infinite bit
vectors with one bit per input byte, evaluated 64 bytes at a time. Every
operation maps to one or two machine instructions, which is what makes this a
credible codegen target rather than an abstract description.

Operations are either stateless per block (`class`, bitwise logic) or carry a
small fixed state across blocks (`shl1` carries one bit, `prefix-xor` a parity,
`add` a carry, `regions` a region state). That carried state is the entire
memory of a kernel — no lookback, no backtracking, no allocation, which is why
these parsers stream at GB/s.

Operands always refer to **earlier** nodes, so a module is in topological order
by construction and a backend can evaluate it in one forward pass per block.
The reader enforces this.

## Text format

Line-oriented. `;` starts a comment; blank lines are ignored. The first
directive must be the version header.

```
falx-ir 1
format csv_geo
structural 2c,0a
quote 22
escape none
column 5 latitude f64
column 6 longitude f64
%0 = class 22
%1 = prefix-xor %0
%2 = class 0a,2c
%3 = not %1
%4 = and %2 %3
%5 = class 0a
output %4
terminators %5
```

That is a complete, working CSV parser with two projected columns: 15 lines in,
~2,800 lines of dispatched SIMD Rust out.

### Header

| Directive | Meaning |
|---|---|
| `falx-ir <version>` | Required first. Refuses a version this build does not read. |
| `format <name>` | Format name; becomes the generated module's identity. |

### Dialect directives

These drive the generated span and columnar APIs — field cleaning, comment
skipping, record grouping — not the graph itself.

| Directive | Meaning |
|---|---|
| `structural <bytes>` | Bytes that delimit fields and records. Required. |
| `quote <byte>` | Quote convention. Omit for none. |
| `escape none` / `escape backslash <byte>` | In-field escaping. Doubled quotes need no declaration — they self-cancel under parity. |
| `comment <byte>` | Line-start comment byte; comment records are skipped by record walkers and contribute no fields or rows. |
| `nesting <o>:<c>,...` | Bracket pairs that nest, e.g. `7b:7d,5b:5d`. Enables the nested-tape API. |
| `lines-per-record <n>` | Group N newline-terminated lines per record (FASTQ = 4). Default 1. |
| `column <field> <name> <type> [info=<key>]` | Project field `<field>` as `<name>`. Types: `i64`, `f64`, `string`, `bytes`. `info=` extracts a key from within the field (VCF INFO-style). |

Bytes are two hex digits. Byte lists are comma-separated; classes also accept
`lo-hi` ranges, so a full class prints as `00-ff`.

### Nodes

`%k = <opcode> <operands...>`, numbered consecutively from `%0`.

| Opcode | Arity | Meaning |
|---|---|---|
| `class <bytes>` | — | Bit i set iff input byte i is in the set. `class 30-39,5f` |
| `const <u64>` | — | The same 64-bit pattern every block. Accepts `0x…` or decimal. |
| `not` / `and` / `or` / `xor` | 1 / 2 / 2 / 2 | Bitwise, over the block. |
| `shl1` | 1 | Bit i = operand bit i−1 ("previous byte matched"). Carries one bit. |
| `shl1-seeded` | 1 | Like `shl1`, but the stream behaves as if preceded by a matching byte — makes position 0 a line start. |
| `prefix-xor` | 1 | Bit i = XOR of operand bits 0..=i. Running parity: the quote-context primitive. Carries a parity bit. |
| `add` | 2 | 64-bit addition with carry into the next block. Makes a set bit ripple through a run — the primitive behind odd/even-run detection for backslash escapes. |
| `regions` | 3 | `regions <quotes> <comment-starts> <terminators>`. Three-state (normal/quote/comment) resolution: quote bits are inert inside comments and comment bits inert inside quotes, the interleaving no bit-parallel parity can express. Carries the region state. |

### Framing (optional)

The bitstream graph can only express framing determined by **byte identity** —
"this byte is a comma". A length-prefixed format is the opposite: where frame
N+1 starts is a value decoded out of frame N. That is a sequential dependency
chain, and no amount of SIMD removes it; it is a property of the format.

What *is* exploitable is the two-level structure such formats share. The chain
is cheap — bounded work per frame, touching only header bytes — while the
frames it produces are independent, so everything downstream is parallel. One
`frame` directive declares the outer container; the graph continues to describe
the payload grammar.

```
frame header=18 length-at=16 width=u16 endian=le counts=total adjust=1 trailer=8 magic=0:1f,8b
```

| Field | Meaning |
|---|---|
| `header` | Offset where the payload begins. With `width=varint` the payload begins right after the varint, and this is any extra header bytes after it (usually 0). |
| `length-at` | Offset of the length field within the frame. |
| `width` | `u8`, `u16`, `u32`, `u64`, or `varint` (unsigned LEB128, protobuf-style). |
| `endian` | `le` or `be`. Ignored for `varint`. |
| `counts` | `total` (the length counts the whole frame) or `payload` (frame = header + payload + trailer). |
| `adjust` | Signed value added to the decoded length. bgzf stores "total minus one", so it needs `adjust=1`. |
| `trailer` | Bytes at the end of the frame that are not payload (bgzf's CRC32 + ISIZE). |
| `magic` | `<offset>:<hex bytes>` that must match, else the frame is rejected. |
| `skip-empty` | `true` drops frames whose payload is empty. |
| `uncompressed` | `<offset>:<width>:<endian>` — where the frame records its *uncompressed* payload size. A non-negative offset counts from the frame start, negative from the frame end, so bgzf's trailing `ISIZE` is `-4:u32:le`. |

`uncompressed` is what makes parallel decompression possible: knowing every
block's inflated length up front lets the output be presized once and carved
into disjoint slots, so workers inflate concurrently with no locking and no
merge pass. Without it a decompressor has to grow buffers sequentially, and
falx refuses rather than guessing.

A module with framing additionally generates:

```rust
pub struct Frame { pub start: usize, pub len: usize, pub payload: Range<usize> }
pub fn frame_at(data: &[u8], pos: usize) -> Result<Frame, FrameError>;
pub fn scan_frames(data: &[u8]) -> Result<Vec<Frame>, FrameError>;
pub fn frames_par<S, Init, F>(data, frames, threads, init, process) -> Vec<S>;

// Whole records, including those straddling frames — see below.
pub const RECORD_TERMINATOR: u8;
pub fn parse_records_par<S, E, Init, MakeDecoder, Decoder, Process>(
    data, frames, threads, init, make_decoder, process,
) -> Result<Vec<S>, E>;
```

`parse_records_par` is the generated twin of
`falx::framing::parse_records_par`: same stitching, but self-contained, so a
standalone parser reassembles records across frames without linking falx.
Decoding is a caller parameter for exactly that reason — a generated file
cannot link a decompressor, so you pass one in (or an identity closure that
copies `data[frame.payload]` for an uncompressed container).

It is not emitted for grouped-line formats (`lines-per-record` > 1), where a
record ends every Nth terminator rather than every one; the generated file
explains the omission instead of shipping stitching that would cut records
mid-group.

`scan_frames` is the sequential pass; `frames_par` hands contiguous frame runs
to workers and returns their states in stream order. This is the shape
`falx::bgzf` reaches ~10 GiB/s with by hand, and the model is checked against
it: `tests/framing.rs` asserts the generalized scanner finds exactly the block
boundaries and payload ranges the hand-written bgzf scanner does.

**With a decompressor.** falx does not decode entropy-coded payloads, but the
`bgzf` feature wires the framing layer to its DEFLATE core, so a described
container can be decompressed and parsed without hand-written code:

```rust
// Locate blocks by description, inflate them in parallel.
let bytes = falx::bgzf::decompress_framed_par(&data, &framing, threads)?;

// Or fuse: inflate into a reusable per-worker buffer and parse each block,
// never materializing the whole stream.
let states = falx::bgzf::parse_framed_par(&data, &framing, threads,
    || MyState::default(),
    |state, _block_index, decompressed| state.absorb(parse_columns(decompressed)),
)?;
```

On 0.5 GiB of bgzf-compressed CSV at 24 threads, the fused form runs at
**~35 GiB/s** of uncompressed throughput against ~6.9 GiB/s for
decompress-then-parse — the gain is not the parsing, it is never writing a
whole-stream buffer. `decompress_framed_par` and the hand-written
`decompress_par` are equivalent in throughput and byte-identical in output.

`parse_framed_par` delivers block *payloads*, which is only enough when records
happen to align with blocks. When they don't — the normal case — use the
record-aware form, which delivers only whole records:

```rust
let states = falx::bgzf::parse_framed_records_par(&data, &framing, threads, b'\n',
    || MyState::default(),
    |state, records| state.absorb(parse_columns(records)),  // whole records only
)?;
```

Each worker decodes its own frames exactly once, streaming frame-by-frame with
a small carry so the working set stays cache-resident; the partial record at
each end is kept as a fragment and the seams are stitched serially afterwards —
at most one record per worker. A stitched record is attributed to the worker
whose region it *starts* in, the same record-ownership rule the parallel
structural parsers use, so the returned states stay in stream order and can be
concatenated directly. Records longer than a whole worker's run are handled;
so is a final record with no terminator.

The stitching itself is format-agnostic and lives in
`falx::framing::parse_records_par`, which takes the per-frame decoder as a
parameter. `bgzf::parse_framed_records_par` is the thin wrapper that supplies
an inflating one; with an identity decoder the same function parses records out
of a plain, uncompressed length-prefixed container, and any other decompressor
slots in the same way.

This measures the same as the block-aligned form (~35 GiB/s on the benchmark
above), so correctness across boundaries is not a tradeoff — prefer it for any
delimited payload.

**What this does not do.** Formats whose *payload* grammar is not a structural
byte stream (Parquet's encoded column chunks, say) are framed by this layer but
not parsed by the bitstream graph.

### Roles

Three directives give streams their meaning to the backend.

| Directive | Meaning |
|---|---|
| `output %k` | Set bits are structural positions. Required. |
| `terminators %k` | Set bits are record ends (ANDed with the output). Required. |
| `nest-open %k` / `nest-close %k` | Live bracket streams for nesting dialects. Both or neither. |

## Stability

The version header is the compatibility contract. Within a version, text that
parses keeps its meaning; any breaking change to a directive or opcode bumps
`ir_text::IR_VERSION`. Printing is deterministic and idempotent, so checked-in
IR diffs cleanly.

Both properties are enforced by `tests/ir_text.rs`, which for **every** in-tree
format lowers it, prints the IR, parses it back, and asserts the regenerated
code is byte-identical — plus round-trip coverage of every opcode, so an
operation cannot be added to the IR without a textual form.

## Worked example: a parser without a spec file

```bash
cat > pipe.fxir <<'EOF'
falx-ir 1
format pipe_delimited
structural 7c,0a          ; '|' and '\n'
escape none
column 0 id i64
column 1 label string
%0 = class 0a             ; record terminators
%1 = class 7c,0a          ; all structural bytes
output %1
terminators %0
EOF

falx build-ir pipe.fxir -o pipe_parser.rs
```

`pipe_parser.rs` is a self-contained, std-only module with runtime SIMD
dispatch, `parse`/`records`/`fields`, `parse_columns`, and the parallel entry
points — from ten lines of IR.

## Limits

The IR has two layers, and they have different reach.

The **bitstream graph** describes structural byte streams: which byte positions
matter, and what state carries between blocks. Everything in it is data-parallel
over bytes, which is why it reaches GB/s.

The **framing layer** describes length-prefixed containers, whose boundaries
cannot be data-parallel by construction. It makes them expressible and
parallel-*after*-scan, which is the most any implementation can do.

Still outside the model:

- **Entropy-coded payloads.** falx locates frames; it does not implement
  DEFLATE, dictionary decoding, or bit-packing. Compressed containers work by
  pairing the framing layer with a decompressor.
- **Payload grammars that are not structural byte streams.** Parquet's encoded
  column chunks are framed by the framing layer but not parsed by the graph.
- **Framing that depends on out-of-band state** — a schema, a footer read
  first, or a dictionary — since a frame's extent must be decidable from the
  frame itself.
