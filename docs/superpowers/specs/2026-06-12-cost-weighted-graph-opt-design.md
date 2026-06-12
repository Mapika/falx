# Cost-Weighted Graph Optimization Design

## Goal

Add an end-to-end graph optimization pass that normalizes and simplifies parser IR graphs before native SIMD codegen, then measure whether the generated kernels get faster.

The first implementation is a local equality-style graph simplifier rather than a full external e-graph engine. It should preserve the public parser APIs and the current weighted-synthesis default while making the codegen boundary ready for a fuller e-graph implementation later.

## Non-Goals

- Do not add a new dependency for equality saturation in the first pass.
- Do not change generated parser public APIs or runtime dispatch behavior.
- Do not reintroduce scalar fallback modules.
- Do not optimize `Regions` internally; keep it as an opaque stateful op.
- Do not claim speedups unless local benchmark output proves them.

## Architecture

Codegen will use this flow:

```text
Dialect + columns
  -> graph source (auto weighted or manual fallback)
  -> DelimitedParts
  -> cost-weighted graph optimizer
  -> emit_with(...)
  -> generated AVX-512/AVX2 native dispatch
```

The optimizer will live in `src/graph_opt.rs`. It accepts a `DelimitedParts` plus a `CostModel` and returns an equivalent `DelimitedParts` with remapped node ids for the structural output, terminator stream, and optional nesting streams.

## Optimization Scope

The first pass will implement deterministic local rewrites and global rebuilding:

- prune nodes unreachable from required output roots
- merge structurally identical nodes with CSE
- canonicalize commutative operands for `And`, `Or`, and `Xor`
- simplify boolean identities involving constants
- simplify idempotent and inverse forms such as `x & x`, `x | x`, `x ^ x`, `!!x`, `x & !x`, and `x | !x`
- preserve all stateful operations as explicit nodes unless an operand rewrite makes them unreachable

This is smaller than a full e-graph, but it gives the codegen path the same extraction shape: build equivalent candidates, pick cheaper forms under the AVX cost model, and emit the chosen graph.

## Codegen Options

`CodegenOptions` will gain an optimizer mode. The default should enable optimization with `CostModel::avx2()`. A disabled mode is useful for tests and benchmarks that compare unoptimized and optimized codegen output.

## Testing

Tests should prove semantic preservation at the graph boundary, not just string changes in generated code:

- optimized CSV/TSV/logfmt/NDJSON/JSON graphs match unoptimized graphs through the interpreter
- terminator and nesting auxiliary nodes are preserved after remapping
- the default codegen output equals explicit optimized output
- disabled optimizer output can differ from optimized output but remains valid generated native SIMD source

The existing drift test will force checked-in kernels to be regenerated from the optimized default path.

## Benchmarking

Benchmark before and after the optimizer using `cargo run --release --example bench` with native CPU flags. Report raw GiB/s and percent deltas for each format. If the optimizer changes code shape but not throughput, report that plainly.
