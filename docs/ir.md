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

The IR describes **structural byte streams**: which byte positions matter, and
what state carries between blocks. Formats whose framing is length-prefixed
(protobuf), block-compressed (Parquet), or otherwise not determined by byte
identity are outside what these operations can express. BGZF is handled as a
separate decompression stage feeding a delimited parser, not as IR.
