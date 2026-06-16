# Fair Comparable Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace weak sustained benchmark baselines with proper parser libraries and make falx fill the missing FASTQ and VCF support needed for fair comparisons.

**Architecture:** Keep the benchmark as one example binary, but replace single-number work accounting with a checked multi-counter `Work` struct. Add small falx-facing helpers for FASTQ parsing and VCF typed projection, then compare those helpers with `seq_io`, `needletail`, `noodles-vcf`, and `logfmt-zerocopy` on equivalent outputs.

**Tech Stack:** Rust 1.89, falx generated kernels, `csv`, `arrow-csv`, `serde_json`, `simd-json`, `seq_io`, `needletail`, `noodles-vcf`, `logfmt-zerocopy`.

---

### Task 1: Benchmark Fairness Types And Tests

**Files:**
- Modify: `examples/bench_sustained.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Write failing tests**

Add tests that require a multi-counter work type, FASTQ validation, and proper library labels:

```rust
#[test]
fn work_equality_checks_all_counters() {
    let a = Work::new(2, 10, 20, 30);
    let b = Work::new(2, 10, 20, 31);
    assert_ne!(a, b);
}

#[test]
fn fastq_falx_reader_validates_sequence_quality_lengths() {
    let bad = b"@r0\nACGT\n+\n!!!\n";
    assert!(falx_fastq_work(bad).is_err());
}

#[test]
fn proper_library_labels_are_used() {
    let labels = comparable_labels();
    assert!(labels.contains(&"seq_io fastq"));
    assert!(labels.contains(&"needletail fastq"));
    assert!(labels.contains(&"noodles-vcf typed records"));
    assert!(labels.contains(&"logfmt-zerocopy pairs"));
}
```

- [ ] **Step 2: Verify the tests fail**

Run: `cargo test --example bench_sustained`

Expected: compile errors for missing `Work`, `falx_fastq_work`, and `comparable_labels`.

- [ ] **Step 3: Add benchmark dependencies**

Add dev dependencies:

```toml
seq_io = "0.3.4"
needletail = { version = "0.7.3", default-features = false }
noodles-vcf = "0.88.0"
logfmt-zerocopy = "0.1.3"
```

- [ ] **Step 4: Implement `Work` and label helpers**

Add a `Work` struct with records, primary bytes, secondary bytes, and checksum counters. Change `Measurement.work` from `u64` to `Work`, and keep formatting compact by printing the four counters.

- [ ] **Step 5: Verify unit tests pass**

Run: `cargo test --example bench_sustained`

Expected: all `bench_sustained` tests pass.

### Task 2: Proper FASTQ Support And Comparisons

**Files:**
- Modify: `examples/bench_sustained.rs`

- [ ] **Step 1: Write failing FASTQ equivalence test**

Add a test that compares falx FASTQ output to both proper libraries on a valid sample:

```rust
#[test]
fn fastq_falx_matches_proper_libraries() {
    let data = b"@r0 desc\nACGT\n+\n!!!!\n@r1\nAC\n+\n!!\n";
    let falx = falx_fastq_work(data).unwrap();
    assert_eq!(falx, seq_io_fastq_work(data));
    assert_eq!(falx, needletail_fastq_work(data));
}
```

- [ ] **Step 2: Verify the test fails**

Run: `cargo test --example bench_sustained fastq_falx_matches_proper_libraries`

Expected: compile errors until the proper-library helpers exist.

- [ ] **Step 3: Implement FASTQ helpers**

Implement:

```rust
fn falx_fastq_work(data: &[u8]) -> Result<Work, &'static str>;
fn seq_io_fastq_work(data: &[u8]) -> Work;
fn needletail_fastq_work(data: &[u8]) -> Work;
```

Each helper counts records, sequence bytes, quality bytes, and a wrapping checksum over sequence and quality bytes.

- [ ] **Step 4: Replace FASTQ benchmark rows**

Use rows for `falx FASTQ records xN`, `seq_io fastq`, and `needletail fastq`. Enforce `assert_same_work("fastq", "records", &rows)`.

- [ ] **Step 5: Verify FASTQ tests pass**

Run: `cargo test --example bench_sustained fastq`

Expected: FASTQ tests pass.

### Task 3: Proper VCF Typed Projection

**Files:**
- Modify: `src/kernels/mod.rs`
- Create: `src/kernels/vcf_typed.rs` by regeneration
- Modify: `examples/bench_sustained.rs`

- [ ] **Step 1: Write failing VCF equivalence test**

Add:

```rust
#[test]
fn vcf_falx_typed_matches_noodles() {
    let data = b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t7\trs1\tA\tC\t42.5\tPASS\tDP=10\n";
    assert_eq!(falx_vcf_typed_work(data), noodles_vcf_work(data));
}
```

- [ ] **Step 2: Verify the test fails**

Run: `cargo test --example bench_sustained vcf_falx_typed_matches_noodles`

Expected: compile error for missing typed VCF helper or module.

- [ ] **Step 3: Add checked-in VCF typed kernel target**

Add a `vcf_typed` target using `formats::vcf_dialect()` with typed columns:

```rust
Column { index: 1, name: Some("pos".into()), ty: ColumnType::I64 }
Column { index: 3, name: Some("reference".into()), ty: ColumnType::Bytes }
Column { index: 4, name: Some("alternate".into()), ty: ColumnType::Bytes }
Column { index: 5, name: Some("quality".into()), ty: ColumnType::F64 }
```

Regenerate with `cargo run --example generate`.

- [ ] **Step 4: Implement VCF work helpers**

Implement:

```rust
fn falx_vcf_typed_work(data: &[u8]) -> Work;
fn noodles_vcf_work(data: &[u8]) -> Work;
```

Both count records, REF+ALT bytes, valid POS count, and a checksum from POS plus QUAL bits.

- [ ] **Step 5: Replace VCF benchmark rows**

Use rows for `falx vcf_typed projection`, `falx vcf_typed projection xN`, and `noodles-vcf typed records`. Enforce matching work.

### Task 4: Proper Logfmt And JSON Labels

**Files:**
- Modify: `examples/bench_sustained.rs`
- Modify: `docs/bench-datasets.md`

- [ ] **Step 1: Write failing logfmt equivalence test**

Add:

```rust
#[test]
fn logfmt_falx_matches_logfmt_zerocopy() {
    let data = b"level=info msg=\"hello world\" count=42\npath=/tmp ok=true\n";
    assert_eq!(falx_logfmt_work(data), logfmt_zerocopy_work(data));
}
```

- [ ] **Step 2: Verify the test fails**

Run: `cargo test --example bench_sustained logfmt_falx_matches_logfmt_zerocopy`

Expected: failure until falx and library helpers normalize the same key/value semantics.

- [ ] **Step 3: Implement logfmt helpers**

Implement `falx_logfmt_work` and `logfmt_zerocopy_work` so both count key/value pairs, key bytes, value bytes, and checksum. Use `record.fields()` on falx and pair adjacent fields as key/value.

- [ ] **Step 4: Rename JSON rows**

Rename NDJSON rows to make the scope explicit:

```text
falx NDJSON line framing
serde_json full DOM parse
simd-json full tape parse
```

Keep falx framing in a separate report section and avoid presenting it as equivalent to full JSON parsing.

### Task 5: Full Verification And Sustained Run

**Files:**
- Modify: `docs/bench-datasets.md`

- [ ] **Step 1: Run formatting and tests**

Run:

```bash
cargo fmt --check
cargo test --example bench_sustained
cargo test --all-targets --all-features
```

- [ ] **Step 2: Run lint and release build**

Run:

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --features cli
```

- [ ] **Step 3: Run sustained comparison**

Run:

```bash
cargo run --release --example bench_sustained -- --formats all --runs 3 --warmup 1
```

- [ ] **Step 4: Update docs**

Document that fair scoreboard rows require matching `Work` counters, and list the proper comparison crates.
