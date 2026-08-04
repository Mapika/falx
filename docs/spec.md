# The falx spec format

A spec is a TOML file describing a format's surface conventions. It is the
convenient frontend; [the IR](ir.md) is the contract underneath it. Use a spec
when your format is delimited in the ordinary way, and drop to IR when you need
a bitstream program a dialect cannot describe.

```bash
falx build   spec.toml -o parser.rs   # spec straight to parser
falx emit-ir spec.toml                # see what it lowered to
falx check   spec.toml                # validate, write nothing
```

## Fields

### Required

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Format name; becomes the generated module's identity. |
| `structural` | array of 1-byte strings | Bytes that delimit fields and records, e.g. `[",", "\n"]`. |

### Optional

| Field | Type | Default | Meaning |
|---|---|---|---|
| `quote` | 1-byte string | none | Quote convention. |
| `escape` | `"none"` \| `"doubled"` \| `"backslash"` | `"none"` | In-field escaping. `"doubled"` needs no machinery — doubled quotes self-cancel under parity — so it lowers identically to `"none"`. |
| `escape_char` | 1-byte string | `"\\"` | Escape byte when `escape = "backslash"`. |
| `comment` | 1-byte string | none | Line-start comment byte. Comment lines are skipped by record walkers, contribute no fields to field-byte stats, and produce no rows in columnar output. Must not be a line terminator or equal `quote`. |
| `nesting` | array of 2-char strings | none | Bracket pairs that nest, e.g. `["{}", "[]"]`. Non-empty enables `parse_nested`. Nesting bytes are added to `structural` automatically. |
| `lines_per_record` | integer ≥ 1 | 1 | Group N newline-terminated lines into one record (FASTQ = 4). Only valid for a pure newline-framed format — no separators, quoting, comments, escapes, or nesting. |

Byte strings accept the escapes `\n`, `\t`, `\r`, `\\`, `\"`.

### Columns

Each `[[columns]]` entry projects one field into a typed column, generating
Arrow-compatible value buffers and validity bitmaps.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `index` | integer ≥ 0 | yes | Zero-based field index within a record. |
| `type` | `"i64"` \| `"f64"` \| `"string"` \| `"bytes"` | yes | Column type. `bytes` is a zero-copy `(start, end)` span into the input. |
| `name` | string | no | Generated field name. Defaults to the `info_key`, else `c{index}`. |
| `info_key` | string | no | Extract this key from *within* field `index` rather than projecting the whole field (VCF INFO-style). Several `info_key` columns sharing an `index` are filled by one pass over that field. |

At most 32 columns; the projection sink's found-mask is a `u32`.

## Examples

Plain CSV:

```toml
name = "csv"
structural = [",", "\n"]
quote = '"'
escape = "doubled"
```

CSV with `#` comments, projecting two typed columns:

```toml
name = "csv_hash"
structural = [",", "\n"]
quote = '"'
comment = "#"

[[columns]]
index = 0
name = "key"
type = "string"

[[columns]]
index = 1
name = "amount"
type = "i64"
```

NDJSON (nesting enables the nested-tape API):

```toml
name = "ndjson"
structural = [",", ":", "\n"]
quote = '"'
escape = "backslash"
nesting = ["{}", "[]"]
```

FASTQ (four lines per record):

```toml
name = "fastq"
structural = ["\n"]
lines_per_record = 4
```

More live examples: [`specs/`](../specs) holds the spec for every in-tree
format, and each is checked against its generated kernel by the codegen drift
test.

## Build-script integration

The intended production shape is generating at compile time and shipping only
the generated file:

```toml
[build-dependencies]
falx = { git = "https://github.com/Mapika/falx", features = ["spec"] }
```

```rust
// build.rs
let spec = falx::spec::parse(&std::fs::read_to_string("spec.toml")?)?;
let code = falx::codegen::emit_parser_with_columns(&spec.dialect, &spec.name, &spec.columns)?;
std::fs::write(format!("{}/parser.rs", std::env::var("OUT_DIR")?), code)?;
```

A complete runnable version lives in [`examples/build-integration/`](../examples/build-integration).

To split the pipeline — say, to check the IR into version control and review
changes to it — use `falx::codegen::lower` to get a module,
`falx::ir_text::print` to write it, and `falx::codegen::emit_module` to
generate from it.

## When a spec is not enough

A spec describes dialects: separators, quoting, comments, nesting, record
grouping. If your format needs a bitstream program outside that shape, write
[the IR](ir.md) directly — `falx build-ir` accepts it, and you still get every
backend, the span API, the columnar API, and the parallel entry points.
