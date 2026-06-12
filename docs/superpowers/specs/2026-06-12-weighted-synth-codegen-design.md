# Weighted Synth Codegen Opt-In Design

## Goal

Make the weighted auto kernel discovery pipeline usable by code generation without changing the default generator yet.

The first implementation will add an opt-in path that synthesizes the parser graph with the weighted synthesizer, then passes that graph into the same native SIMD codegen backend used by the current generated kernels. This preserves the recent no-fallback native dispatch behavior while making the discovered graph path available for real kernel generation.

## Non-Goals

- Do not make weighted synthesis the default generator path in this change.
- Do not reintroduce generated scalar fallback modules.
- Do not require every existing target to be synth-backed before the opt-in flag can ship.
- Do not change parser public APIs or generated kernel call signatures unless the implementation uncovers a hard compatibility issue.

## Architecture

Codegen will support two graph sources:

- `Manual`: today's `formats::delimited_parts()` path.
- `SynthWeighted`: a new opt-in path that builds equivalent `DelimitedParts` through weighted synthesis using `Order::Cost` and the existing AVX cost model.

Both paths will converge before backend emission:

```text
Dialect + columns
  -> graph source
  -> DelimitedParts
  -> emit_with(...)
  -> generated AVX-512/AVX2 native dispatch
```

The generated Rust should continue to prefer AVX-512F/BW/VL plus PCLMULQDQ, then AVX2 plus PCLMULQDQ, and fail fast on unsupported CPUs. No `fallback` module should be emitted by either graph source.

## Components

### Codegen Options

Add a small options layer around parser emission, for example `CodegenOptions` with `GraphSource`.

Existing callers can keep using `emit_parser_with_columns(...)`, which will default to `Manual`. New opt-in callers can use an options-taking API to request `SynthWeighted`.

### Synth Format Builder

Add a focused module that converts a `Dialect` into multi-output synthesis specs and returns `DelimitedParts`.

The first supported outputs are:

- escaped-byte mask for backslash-escaped dialects
- real quote mask
- in-string mask
- structural-byte mask
- terminator mask
- open and close masks for nested formats such as JSON

The synth path should share the existing `Graph`, `Node`, and `DelimitedParts` types instead of introducing a second kernel representation.

### Generator Integration

Extend `examples/generate.rs` with an opt-in flag:

```text
cargo run --example generate -- --synth weighted
```

Default generation remains unchanged:

```text
cargo run --example generate
```

The generator should print a concise per-target report showing whether each target used `synth-weighted`, `manual`, or was skipped/failed.

## Supported Targets

The initial opt-in synth path should target dialects whose parsing behavior is expressible through the current bit-graph and synth/prover machinery:

- `csv`
- `tsv`
- `logfmt`
- `ndjson`
- `json`
- `multi`

The first implementation may leave comment-region dialects such as `csv_hash` on the manual path because current comments use `Regions`, which is not cleanly represented in the weighted synth pipeline yet.

## Error Handling

Synthesis failures should be explicit.

For direct opt-in codegen APIs, return an error that includes the format name and the failing synthesis stage.

For the generator flag, use a conservative mixed policy for the first iteration:

- use synthesized graphs for supported targets
- use manual graphs for unsupported targets
- fail the command if a target advertised as synth-supported cannot synthesize successfully

This gives us usable opt-in generation without silently hiding regressions in the supported synth set.

## Testing

Add focused tests before implementation:

- CSV or TSV synth graph equivalence against the manual graph on representative inputs.
- A JSON/nesting equivalence test if runtime is acceptable; otherwise mark a heavier JSON synth test ignored and keep a smaller structural contract test in CI.
- A codegen contract test that `SynthWeighted` generation emits native SIMD modules and no fallback module.
- A generator argument test or lightweight integration test covering `--synth weighted` option parsing and reporting.

Keep the tests small enough for normal `cargo test`. Expensive weighted searches can be protected with ignored tests or a fuller local verification command.

## Rollout

This change lands as opt-in only. After it is working and benchmarked, the next step is to switch the default generator to synth-weighted for the supported targets, then close the remaining gap for comment-region formats.
