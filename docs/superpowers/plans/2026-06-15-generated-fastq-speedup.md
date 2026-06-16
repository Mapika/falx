# Generated FASTQ Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generated FASTQ parser API that fuses newline detection, FASTQ validation, and benchmark work accounting without handwritten FASTQ kernels.

**Architecture:** Add a `fastq` checked-in target that reuses the synthesized/egraph-optimized `lines` dialect graph. Extend codegen to append a FASTQ domain API only for the `fastq` target: generated `FastqStats`, `FastqError`, `parse_fastq`, and architecture-dispatched block loops that call the generated `step` function and feed a generated sink. Update the sustained benchmark to call `falx::kernels::fastq::parse_fastq` instead of local FASTQ helper logic.

**Tech Stack:** Rust 1.89, falx codegen, synthesized line structural graph, checked-in generated kernels, `seq_io`, `needletail`.

---

### Task 1: Add Red Tests For Generated FASTQ

**Files:**
- Modify: `tests/codegen.rs`
- Modify: `examples/bench_sustained.rs`

- [ ] **Step 1: Add codegen drift expectations**

Add a test that requires the registry to include `fastq`, generated from `formats::lines_dialect()`, and requires emitted code to expose the generated FASTQ API:

```rust
#[test]
fn generated_fastq_target_exposes_domain_api() {
    let target = falx::kernels::targets()
        .into_iter()
        .find(|(name, _, _)| *name == "fastq")
        .expect("fastq target is registered");
    let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
        .expect("fastq codegen should succeed");

    assert!(generated.contains("pub struct FastqStats"));
    assert!(generated.contains("pub enum FastqError"));
    assert!(generated.contains("pub fn parse_fastq("));
}
```

- [ ] **Step 2: Add benchmark equivalence tests**

Add tests that require the generated FASTQ API to match proper libraries:

```rust
#[test]
fn generated_fastq_matches_proper_libraries() {
    let data = b"@r0 desc\nACGT\n+\n!!!!\n@r1\nAC\n+\n!!\n";
    let falx = generated_fastq_work(data).unwrap();
    assert_eq!(falx, seq_io_fastq_work(data));
    assert_eq!(falx, needletail_fastq_work(data));
}

#[test]
fn generated_fastq_rejects_bad_separator() {
    let data = b"@r0\nACGT\n-\n!!!!\n";
    assert!(generated_fastq_work(data).is_err());
}
```

- [ ] **Step 3: Run red tests**

Run:

```bash
cargo test --test codegen generated_fastq_target_exposes_domain_api
cargo test --example bench_sustained generated_fastq
```

Expected: compile/test failures because `fastq` target and generated API do not exist.

### Task 2: Generate FASTQ Target And API

**Files:**
- Modify: `src/kernels/mod.rs`
- Modify: `src/codegen.rs`
- Create: `src/kernels/fastq.rs` by regeneration

- [ ] **Step 1: Register generated target**

Add `pub mod fastq;` and a target:

```rust
("fastq", formats::lines_dialect(), vec![])
```

- [ ] **Step 2: Extend codegen for `format_name == "fastq"`**

Add a `push_fastq_api(code, par_mode)` call from `emit_with` after parser APIs are emitted. The generated API must include:

```rust
pub struct FastqStats {
    pub records: u64,
    pub sequence_bytes: u64,
    pub quality_bytes: u64,
    pub checksum: u64,
}

pub enum FastqError {
    IncompleteRecord,
    BadHeader { offset: usize },
    BadSeparator { offset: usize },
    LengthMismatch { record: u64, sequence: usize, quality: usize },
    TrailingBytes { offset: usize },
}

pub fn parse_fastq(data: &[u8]) -> Result<FastqStats, FastqError>;
```

The generated architecture dispatch loops must call the generated `step` function to get newline masks and feed a generated sink. No handwritten `src/kernels/fastq.rs` code should remain after regeneration.

- [ ] **Step 3: Regenerate kernels**

Run:

```bash
cargo run --example generate
```

Expected: writes `src/kernels/fastq.rs` with generated FASTQ API.

### Task 3: Use Generated FASTQ In Benchmark

**Files:**
- Modify: `examples/bench_sustained.rs`

- [ ] **Step 1: Replace falx FASTQ helper**

Add:

```rust
fn generated_fastq_work(data: &[u8]) -> Result<Work, falx::kernels::fastq::FastqError> {
    falx::kernels::fastq::parse_fastq(data).map(|stats| {
        Work::new(
            stats.records,
            stats.sequence_bytes,
            stats.quality_bytes,
            stats.checksum,
        )
    })
}
```

- [ ] **Step 2: Update benchmark rows**

Replace falx FASTQ rows with:

```rust
Row {
    label: "falx generated FASTQ".into(),
    measurement: measure(options, || generated_fastq_work(data).expect("valid FASTQ for falx")),
}
```

Keep `seq_io fastq` and `needletail fastq` rows unchanged, and keep `assert_same_work`.

- [ ] **Step 3: Remove local falx FASTQ helper path**

Delete local `falx_fastq_work`, `falx_fastq_work_from_newlines`, and the release benchmark use of `lines::index_structurals_par` for FASTQ. Test-only scalar helpers can remain under `#[cfg(test)]` only if needed by existing tests.

### Task 4: Verify And Benchmark

**Files:**
- Modify: `docs/bench-datasets.md`

- [ ] **Step 1: Run tests and lint**

Run:

```bash
cargo fmt --check
cargo test --example bench_sustained
cargo test --test codegen generated_fastq_target_exposes_domain_api
cargo clippy --all-targets --all-features -- -D warnings
```

- [ ] **Step 2: Run full suite**

Run:

```bash
cargo test --all-targets --all-features
cargo build --release --features cli
```

- [ ] **Step 3: Run FASTQ benchmark**

Run:

```bash
cargo run --release --example bench_sustained -- --formats fastq --runs 5 --warmup 2
```

- [ ] **Step 4: Update docs**

Document that FASTQ is now a generated domain API backed by the synthesized line kernel, and record the best speed versus `seq_io` and `needletail`.
