# Cost-Weighted Graph Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-on cost-weighted graph optimizer before native codegen and benchmark the generated kernels.

**Architecture:** Add a focused `graph_opt` module that rewrites and rebuilds `DelimitedParts`, preserving all output roles by remapping node ids. Integrate it in `CodegenOptions` after graph-source selection and before `emit_with`, with an opt-out mode for comparison benchmarks.

**Tech Stack:** Rust 2024, existing `Graph`/`Op` IR, `DelimitedParts`, `CostModel::avx2()`, existing codegen and benchmark examples.

---

### Task 1: Tests for Semantic Preservation

**Files:**
- Create: `tests/graph_opt.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write tests that run optimized and unoptimized nodes**

Create `tests/graph_opt.rs` with helper functions that run a graph output or a specific node through `falx::interp::run`, then compare optimized and manual `DelimitedParts` for CSV, TSV, logfmt, NDJSON, and JSON.

- [ ] **Step 2: Expose the module**

Add `pub mod graph_opt;` to `src/lib.rs`.

- [ ] **Step 3: Run the new test before implementation**

Run: `cargo test --test graph_opt --quiet`

Expected: FAIL because `falx::graph_opt` does not exist yet.

### Task 2: Graph Optimizer

**Files:**
- Create: `src/graph_opt.rs`
- Test: `tests/graph_opt.rs`

- [ ] **Step 1: Implement root-preserving optimizer**

Create `GraphRoots`, `OptimizationStats`, `OptimizedParts`, and `optimize_parts`. Rebuild only nodes reachable from the structural output, terminator node, and optional nesting nodes.

- [ ] **Step 2: Implement deterministic simplification**

Implement CSE, commutative canonicalization, constant folding for boolean ops, double-not simplification, idempotent rules, and inverse boolean rules.

- [ ] **Step 3: Preserve opaque stateful ops**

Rebuild `ShiftLeft1`, `ShiftLeft1Seeded`, `PrefixXor`, `Add`, and `Regions` after their operands are optimized. Do not rewrite across their stateful semantics.

- [ ] **Step 4: Run optimizer tests**

Run: `cargo test --test graph_opt --quiet`

Expected: PASS.

### Task 3: Codegen Integration

**Files:**
- Modify: `src/codegen.rs`
- Modify: `tests/synth_codegen.rs`

- [ ] **Step 1: Add optimizer option**

Add `GraphOptimizer::{Disabled, CostWeighted(CostModel)}` or an equivalent option to `CodegenOptions`, defaulting to AVX2 cost-weighted optimization.

- [ ] **Step 2: Optimize parser parts before emission**

In `emit_parser_with_columns_options`, run `graph_opt::optimize_parts` on the selected `DelimitedParts` unless disabled.

- [ ] **Step 3: Add codegen behavior tests**

Add tests proving default codegen equals explicit optimized codegen, and that disabled optimizer still emits native SIMD without fallback.

- [ ] **Step 4: Run focused tests**

Run: `cargo test --test synth_codegen --quiet`

Expected: PASS.

### Task 4: Regenerate and Verify

**Files:**
- Modify: generated files under `src/kernels/*.rs` only when codegen output changes
- Modify: README or architecture docs only if behavior text needs updating

- [ ] **Step 1: Regenerate kernels**

Run: `cargo run --release --example generate`

Expected: supported targets use `[synth-weighted]`; unsupported `csv_hash` uses `[manual]`.

- [ ] **Step 2: Run drift and correctness tests**

Run:

```bash
cargo test --test codegen generated_kernels_match_codegen --quiet
cargo test --test codegen generated_kernels_differential --quiet
cargo test --test codegen generated_kernels_long_input --quiet
```

Expected: PASS.

- [ ] **Step 3: Run full verification**

Run:

```bash
cargo test --quiet
cargo clippy --all-targets -- -D warnings
git diff --check
```

Expected: PASS.

### Task 5: Benchmarks and Numbers

**Files:**
- No required source files; collect command output for final report.

- [ ] **Step 1: Benchmark optimized default**

Run: `RUSTFLAGS='-C target-cpu=native' cargo run --release --example bench`

Expected: benchmark table with GiB/s for each format.

- [ ] **Step 2: Benchmark unoptimized comparison**

Run the same benchmark against a temporary disabled-optimizer build or a saved baseline commit.

Expected: comparable benchmark table.

- [ ] **Step 3: Report deltas**

Calculate percent changes as `(optimized - baseline) / baseline * 100` and report raw throughput plus deltas.

### Task 6: Commit and Push

**Files:**
- All changed files.

- [ ] **Step 1: Inspect final diff**

Run: `git status --short` and `git diff --stat`.

- [ ] **Step 2: Commit**

Run:

```bash
git add docs/superpowers src tests README.md ARCHITECTURE.md
git commit -m "feat: optimize generated format graphs"
```

- [ ] **Step 3: Push**

Run: `git push`

Expected: branch `weighted-synth-codegen` updates on GitHub.

## Self-Review

- Spec coverage: the tasks cover optimizer implementation, codegen integration, regeneration, verification, benchmarking, and push.
- Placeholder scan: no task relies on unspecified future code or an unnamed command.
- Type consistency: the planned API is rooted in existing `DelimitedParts`, `Graph`, `NodeId`, and `CostModel` types.
