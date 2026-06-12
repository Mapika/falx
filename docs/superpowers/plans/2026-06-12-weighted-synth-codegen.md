# Weighted Synth Codegen Opt-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in weighted synthesis graph source for generated kernels while keeping the default generator on the current manual graph path.

**Architecture:** Codegen gets a small options API that chooses between manual `formats::delimited_parts()` and a new `synth_formats` graph builder. The synth builder converts each supported `Dialect` into weighted multi-output synthesis specs and returns the existing `DelimitedParts` shape, so the native AVX-512/AVX2 backend remains unchanged.

**Tech Stack:** Rust 2024, existing `falx::ir`, `falx::synth`, `falx::formats`, `falx::codegen`, cargo examples, cargo tests.

---

## File Structure

- Create `src/synth_formats.rs`: owns weighted synthesis of `DelimitedParts` from a `Dialect`, including synth profiles, references, corpus generation, support checks, and error formatting.
- Modify `src/lib.rs`: export `synth_formats`.
- Modify `src/codegen.rs`: add `GraphSource`, `CodegenOptions`, and an options-taking parser emission function.
- Modify `examples/generate.rs`: parse `--synth weighted`, use the synth graph source for supported targets, and report `synth-weighted` or `manual` per generated kernel.
- Create `tests/synth_codegen.rs`: focused opt-in tests for graph equivalence, native no-fallback emission, and unsupported comment dialect behavior.
- Modify `README.md` and `ARCHITECTURE.md`: document the opt-in generator command and the default/manual vs opt-in/synth split.

---

### Task 1: Add Failing Opt-In Tests

**Files:**
- Create: `tests/synth_codegen.rs`

- [ ] **Step 1: Add the test file**

Create `tests/synth_codegen.rs` with this content:

```rust
use falx::codegen::{self, CodegenOptions, GraphSource};
use falx::formats;
use falx::interp;
use falx::ir::{Graph, NodeId};
use falx::synth_formats::{self, SynthProfile};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn run_graph(graph: &Graph, data: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    interp::run(graph, data, &mut out);
    out
}

fn run_node(graph: &Graph, node: NodeId, data: &[u8]) -> Vec<u32> {
    let mut graph = graph.clone();
    graph.set_output(node);
    run_graph(&graph, data)
}

fn cases(alphabet: &[u8]) -> Vec<Vec<u8>> {
    let mut rng = Rng(0xA17E_5A11_D15C_0DED);
    let mut cases = vec![
        Vec::new(),
        b"a,b,c\n".to_vec(),
        b"\"a,b\",c\n".to_vec(),
        b"\"multi\nline\",x\n".to_vec(),
        b"\t\t\nx\ty\n".to_vec(),
    ];
    for _ in 0..64 {
        let len = (rng.next() % 256) as usize;
        cases.push(
            (0..len)
                .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                .collect(),
        );
    }
    cases
}

#[test]
fn weighted_synth_csv_and_tsv_match_manual_graphs() {
    let formats = [
        (formats::csv_dialect(), b"\",\nxy".as_slice()),
        (formats::tsv_dialect(), b"\t\nxy".as_slice()),
    ];

    for (dialect, alphabet) in formats {
        let manual = formats::delimited_parts(&dialect);
        let synthesized =
            synth_formats::synthesize_delimited_parts_with_profile(&dialect, SynthProfile::Fast)
                .expect("fast weighted synthesis should solve this dialect");

        for data in cases(alphabet) {
            assert_eq!(
                run_graph(&synthesized.graph, &data),
                run_graph(&manual.graph, &data),
                "structural stream diverged for input {data:?}"
            );
            assert_eq!(
                run_node(&synthesized.graph, synthesized.terminators, &data),
                run_node(&manual.graph, manual.terminators, &data),
                "raw terminator stream diverged for input {data:?}"
            );
        }
    }
}

#[test]
fn weighted_synth_codegen_emits_native_simd_without_fallback() {
    let code = codegen::emit_parser_with_columns_options(
        &formats::csv_dialect(),
        "csv_synth_test",
        &[],
        CodegenOptions { graph_source: GraphSource::SynthWeighted(SynthProfile::Fast) },
    )
    .expect("synth-weighted codegen should succeed");

    assert!(!code.contains("pub mod fallback"));
    assert!(!code.contains("fallback::"));
    assert!(code.contains("mod avx512"));
    assert!(
        code.find("avx512::").expect("AVX-512 dispatch present")
            < code.find("avx2::").expect("AVX2 dispatch present")
    );
}

#[test]
fn weighted_synth_rejects_comment_region_dialects() {
    let dialect = formats::csv_hash_dialect();
    assert!(!synth_formats::supports_weighted(&dialect));

    let err = match synth_formats::synthesize_delimited_parts_with_profile(
        &dialect,
        SynthProfile::Fast,
    ) {
        Ok(_) => panic!("comment regions are not synth-supported yet"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("comment"));
}
```

- [ ] **Step 2: Run the new tests and verify they fail for missing APIs**

Run:

```bash
cargo test --test synth_codegen --quiet
```

Expected: compilation fails because `falx::synth_formats`, `CodegenOptions`, `GraphSource`, and `emit_parser_with_columns_options` do not exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/synth_codegen.rs
git commit -m "test: cover weighted synth codegen opt-in"
```

---

### Task 2: Add Codegen Option Surface

**Files:**
- Modify: `src/codegen.rs`

- [ ] **Step 1: Add graph source types near the existing public codegen types**

Insert this after `Column` and before `RESERVED_FIELDS`:

```rust
/// Source used to build the parser graph before native backend emission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphSource {
    /// Use the handwritten graph builder in `formats`.
    Manual,
    /// Use weighted synthesis to build the graph, then emit the same native
    /// SIMD backend as manual graph generation.
    SynthWeighted(crate::synth_formats::SynthProfile),
}

/// Options for parser code generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodegenOptions {
    pub graph_source: GraphSource,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        Self { graph_source: GraphSource::Manual }
    }
}
```

- [ ] **Step 2: Add an options-taking parser emission API**

Replace the body of `emit_parser_with_columns` and add the new function immediately after it:

```rust
pub fn emit_parser_with_columns(
    dialect: &crate::formats::Dialect,
    format_name: &str,
    columns: &[Column],
) -> Result<String, CodegenError> {
    emit_parser_with_columns_options(dialect, format_name, columns, CodegenOptions::default())
}

pub fn emit_parser_with_columns_options(
    dialect: &crate::formats::Dialect,
    format_name: &str,
    columns: &[Column],
    options: CodegenOptions,
) -> Result<String, CodegenError> {
    validate_columns(columns)?;
    validate_nesting(dialect)?;
    let parts = match options.graph_source {
        GraphSource::Manual => crate::formats::delimited_parts(dialect),
        GraphSource::SynthWeighted(profile) => {
            crate::synth_formats::synthesize_delimited_parts_with_profile(dialect, profile)
                .map_err(|err| CodegenError(format!("synth-weighted {format_name}: {err}")))?
        }
    };
    emit_with(
        &parts.graph,
        format_name,
        Some((dialect, parts.terminators, parts.nest)),
        columns,
    )
}
```

- [ ] **Step 3: Run the focused test and confirm only `synth_formats` remains missing**

Run:

```bash
cargo test --test synth_codegen --quiet
```

Expected: compilation still fails because `crate::synth_formats` is not defined.

- [ ] **Step 4: Commit the API surface**

```bash
git add src/codegen.rs
git commit -m "feat: add codegen graph source options"
```

---

### Task 3: Implement Weighted Synth Format Builder

**Files:**
- Create: `src/synth_formats.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Export the new module**

In `src/lib.rs`, add:

```rust
pub mod synth_formats;
```

Place it next to the existing `pub mod synth;` export.

- [ ] **Step 2: Create the synth builder module**

Create `src/synth_formats.rs` with these public items and helpers:

```rust
//! Weighted synthesis of format graphs from `Dialect` descriptions.

use crate::formats::{DelimitedParts, Dialect, Escape};
use crate::synth::{
    Budget, CostModel, Leaf, MultiOutcome, MultiSpec, Order, Spec, Stats, synthesize_multi,
};

const EVEN: u64 = 0x5555_5555_5555_5555;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthProfile {
    /// Small search profile for CI and smoke tests.
    Fast,
    /// Full weighted profile for opt-in kernel generation.
    Weighted,
}

#[derive(Debug)]
pub enum SynthFormatError {
    Unsupported(&'static str),
    NotFound { stage: &'static str, stats: Stats },
}

impl std::fmt::Display for SynthFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(f, "unsupported dialect: {reason}"),
            Self::NotFound { stage, stats } => write!(
                f,
                "no solution for {stage}: completed level {}, {} candidates, {} bank terms",
                stats.completed_level, stats.candidates, stats.bank_unique
            ),
        }
    }
}

impl std::error::Error for SynthFormatError {}

pub fn supports_weighted(dialect: &Dialect) -> bool {
    dialect.comment.is_none()
}

pub fn synthesize_delimited_parts_with_profile(
    dialect: &Dialect,
    profile: SynthProfile,
) -> Result<DelimitedParts, SynthFormatError> {
    if dialect.comment.is_some() {
        return Err(SynthFormatError::Unsupported(
            "comment regions currently use the sequential Regions op",
        ));
    }

    let corpus = corpus_for(dialect);
    let budget = budget(profile);
    let structural_bytes = dialect.structural.clone();
    let terminator_bytes = [b'\n'];
    let quote_byte = dialect.quote;
    let escape_byte = match dialect.escape {
        Escape::None => None,
        Escape::Backslash(byte) => Some(byte),
    };
    let opens: Vec<u8> = dialect.nesting.iter().map(|&(open, _)| open).collect();
    let closes: Vec<u8> = dialect.nesting.iter().map(|&(_, close)| close).collect();
    let opens_for_ref = opens.clone();
    let closes_for_ref = closes.clone();

    let escaped_ref = move |data: &[u8]| escaped_reference(data, escape_byte.unwrap_or(b'\\'));
    let escaped_care = move |data: &[u8]| match quote_byte {
        Some(quote) => mask_ref(data, |b| b == quote),
        None => vec![0; data.len().div_ceil(64)],
    };
    let real_quotes_ref = move |data: &[u8]| real_quotes_reference(data, quote_byte, escape_byte);
    let in_string_ref = move |data: &[u8]| in_string_reference(data, quote_byte, escape_byte);
    let structural_ref =
        move |data: &[u8]| live_class_reference(data, &structural_bytes, quote_byte, escape_byte);
    let terminator_ref = move |data: &[u8]| mask_ref(data, |b| b == b'\n');
    let opens_ref =
        move |data: &[u8]| live_class_reference(data, &opens_for_ref, quote_byte, escape_byte);
    let closes_ref =
        move |data: &[u8]| live_class_reference(data, &closes_for_ref, quote_byte, escape_byte);

    let escape_leaves = vec![
        Leaf::class("B", &[escape_byte.unwrap_or(b'\\')]),
        Leaf::constant("EVEN", EVEN),
    ];
    let quote_leaves = quote_byte.map(|quote| vec![Leaf::class("Q", &[quote])]);
    let structural_leaves = vec![Leaf::class("Struct", &dialect.structural)];
    let terminator_leaves = vec![Leaf::class("N", &terminator_bytes)];
    let open_leaves = (!dialect.nesting.is_empty()).then(|| Leaf::class("Open", &opens));
    let close_leaves = (!dialect.nesting.is_empty()).then(|| Leaf::class("Close", &closes));

    let mut specs = Vec::new();
    let mut stage_names = Vec::new();
    let structural_idx;
    let terminator_idx;
    let mut opens_idx = None;
    let mut closes_idx = None;

    if escape_byte.is_some() {
        stage_names.push("escaped positions");
        specs.push(MultiSpec {
            leaves: &escape_leaves,
            spec: Spec::with_care(&escaped_ref, &escaped_care),
        });
    }
    if let Some(leaves) = quote_leaves.as_ref() {
        stage_names.push("real quotes");
        specs.push(MultiSpec { leaves, spec: Spec::exact(&real_quotes_ref) });

        stage_names.push("in-string mask");
        specs.push(MultiSpec { leaves: &[], spec: Spec::exact(&in_string_ref) });
    }

    structural_idx = specs.len();
    stage_names.push("structural mask");
    specs.push(MultiSpec { leaves: &structural_leaves, spec: Spec::exact(&structural_ref) });

    terminator_idx = specs.len();
    stage_names.push("terminator mask");
    specs.push(MultiSpec { leaves: &terminator_leaves, spec: Spec::exact(&terminator_ref) });

    if let (Some(open_leaf), Some(close_leaf)) = (open_leaves.as_ref(), close_leaves.as_ref()) {
        opens_idx = Some(specs.len());
        stage_names.push("live open brackets");
        specs.push(MultiSpec {
            leaves: std::slice::from_ref(open_leaf),
            spec: Spec::exact(&opens_ref),
        });

        closes_idx = Some(specs.len());
        stage_names.push("live close brackets");
        specs.push(MultiSpec {
            leaves: std::slice::from_ref(close_leaf),
            spec: Spec::exact(&closes_ref),
        });
    }

    let multi = match synthesize_multi(&corpus, &specs, &budget) {
        MultiOutcome::Found(multi) => multi,
        MultiOutcome::NotFound { failed_spec, stats } => {
            return Err(SynthFormatError::NotFound {
                stage: stage_names[failed_spec],
                stats,
            });
        }
    };

    let mut graph = multi.graph.clone();
    graph.set_output(multi.outputs[structural_idx]);
    let terminators = multi.outputs[terminator_idx];
    let nest = opens_idx.zip(closes_idx).map(|(open, close)| (multi.outputs[open], multi.outputs[close]));

    Ok(DelimitedParts { graph, terminators, nest })
}
```

- [ ] **Step 3: Add the helper functions in the same file**

Add these helpers below the public function:

```rust
fn budget(profile: SynthProfile) -> Budget {
    match profile {
        SynthProfile::Fast => Budget {
            max_level: 18,
            max_candidates: 5_000_000,
            max_bank: 500_000,
            settle_levels: 0,
            cost: CostModel::avx2(),
            order: Order::Cost,
            progress: false,
        },
        SynthProfile::Weighted => Budget {
            max_level: 28,
            max_candidates: 60_000_000,
            max_bank: 2_000_000,
            settle_levels: 2,
            cost: CostModel::avx2(),
            order: Order::Cost,
            progress: false,
        },
    }
}

fn corpus_for(dialect: &Dialect) -> Vec<Vec<u8>> {
    let mut alphabet = dialect.structural.clone();
    if let Some(quote) = dialect.quote {
        alphabet.extend([quote, quote]);
    }
    if let Escape::Backslash(escape) = dialect.escape {
        alphabet.extend([escape, escape, escape]);
    }
    alphabet.extend_from_slice(b"xy \t\r");
    alphabet.sort_unstable();
    alphabet.dedup();

    let mut rng = Rng(0x51A7_EC0D_EC0D_0001);
    vec![
        pad_to_block(Vec::new(), &alphabet),
        pad_to_block(alphabet.iter().copied().cycle().take(128).collect(), &alphabet),
        uniform(&alphabet, 2, &mut rng),
        runs(&alphabet, 2, &mut rng),
        quote_boundary_case(dialect, &alphabet),
    ]
}

fn pad_to_block(mut data: Vec<u8>, alphabet: &[u8]) -> Vec<u8> {
    let pad = alphabet.iter().copied().find(|&b| b != b'"' && b != b'\\').unwrap_or(b'x');
    let len = data.len().next_multiple_of(64).max(64);
    data.resize(len, pad);
    data
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn uniform(alphabet: &[u8], blocks: usize, rng: &mut Rng) -> Vec<u8> {
    (0..blocks * 64)
        .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
        .collect()
}

fn runs(alphabet: &[u8], blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let len = blocks * 64;
    let mut input = Vec::with_capacity(len);
    while input.len() < len {
        let byte = alphabet[(rng.next() % alphabet.len() as u64) as usize];
        for _ in 0..(1 + rng.next() % 10).min((len - input.len()) as u64) {
            input.push(byte);
        }
    }
    input
}

fn quote_boundary_case(dialect: &Dialect, alphabet: &[u8]) -> Vec<u8> {
    let mut data = vec![alphabet[0]; 192];
    if let Some(quote) = dialect.quote {
        for pos in [0usize, 63, 64, 65, 127, 128] {
            data[pos] = quote;
        }
    }
    if let Escape::Backslash(escape) = dialect.escape {
        for pos in [10usize, 11, 62, 63, 64, 126, 127, 128] {
            data[pos] = escape;
        }
    }
    for (i, byte) in dialect.structural.iter().copied().enumerate() {
        data[20 + i * 7] = byte;
    }
    data
}

fn mask_ref(data: &[u8], mut f: impl FnMut(u8) -> bool) -> Vec<u64> {
    let mut masks = vec![0u64; data.len().div_ceil(64)];
    for (i, &byte) in data.iter().enumerate() {
        if f(byte) {
            masks[i / 64] |= 1 << (i % 64);
        }
    }
    masks
}

fn escaped_reference(data: &[u8], escape: u8) -> Vec<u64> {
    let mut run_odd = false;
    mask_ref(data, |byte| {
        if byte == escape {
            run_odd = !run_odd;
            false
        } else {
            let out = run_odd;
            run_odd = false;
            out
        }
    })
}

fn real_quotes_reference(data: &[u8], quote: Option<u8>, escape: Option<u8>) -> Vec<u64> {
    let Some(quote) = quote else {
        return vec![0; data.len().div_ceil(64)];
    };
    let mut run_odd = false;
    mask_ref(data, |byte| {
        let escaped = match escape {
            Some(escape) if byte == escape => {
                run_odd = !run_odd;
                return false;
            }
            Some(_) => {
                let escaped = run_odd;
                run_odd = false;
                escaped
            }
            None => false,
        };
        byte == quote && !escaped
    })
}

fn in_string_reference(data: &[u8], quote: Option<u8>, escape: Option<u8>) -> Vec<u64> {
    let Some(quote) = quote else {
        return vec![0; data.len().div_ceil(64)];
    };
    let mut run_odd = false;
    let mut in_string = false;
    mask_ref(data, |byte| {
        let escaped = match escape {
            Some(escape) if byte == escape => {
                run_odd = !run_odd;
                return in_string;
            }
            Some(_) => {
                let escaped = run_odd;
                run_odd = false;
                escaped
            }
            None => false,
        };
        if byte == quote && !escaped {
            in_string = !in_string;
        }
        in_string
    })
}

fn live_class_reference(
    data: &[u8],
    class: &[u8],
    quote: Option<u8>,
    escape: Option<u8>,
) -> Vec<u64> {
    if quote.is_none() {
        return mask_ref(data, |byte| class.contains(&byte));
    }
    let inside = in_string_reference(data, quote, escape);
    let raw = mask_ref(data, |byte| class.contains(&byte));
    raw.into_iter().zip(inside).map(|(raw, inside)| raw & !inside).collect()
}
```

- [ ] **Step 4: Fix borrow or formatting issues from the concrete insertion**

If Rust rejects `std::slice::from_ref(open_leaf)` because of temporary lifetimes, replace the dynamic nesting leaves with stable vectors:

```rust
let open_leaves_vec = open_leaves.into_iter().collect::<Vec<_>>();
let close_leaves_vec = close_leaves.into_iter().collect::<Vec<_>>();
```

Then pass `&open_leaves_vec` and `&close_leaves_vec` to the two nesting specs.

- [ ] **Step 5: Run focused tests**

Run:

```bash
cargo test --test synth_codegen --quiet
```

Expected: tests compile and pass. If `SynthProfile::Fast` is too small for CSV, increase only `max_level` to `20` and `max_candidates` to `10_000_000`, then rerun the same command.

- [ ] **Step 6: Commit synth builder**

```bash
git add src/lib.rs src/synth_formats.rs tests/synth_codegen.rs
git commit -m "feat: synthesize delimited graph parts"
```

---

### Task 4: Wire Opt-In Generator Flag

**Files:**
- Modify: `examples/generate.rs`

- [ ] **Step 1: Replace the generator with CLI-aware code**

Replace `examples/generate.rs` with:

```rust
//! Regenerate the checked-in kernels in src/kernels/ from their dialects.
//! Run after changing the IR, codegen, or a format definition:
//! `cargo run --example generate`
//!
//! To opt into weighted synthesized graphs for supported targets:
//! `cargo run --example generate -- --synth weighted`

use falx::codegen::{self, CodegenOptions, GraphSource};
use falx::synth_formats::{self, SynthProfile};
use falx::kernels;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerateMode {
    Manual,
    SynthWeighted,
}

fn parse_mode<I>(args: I) -> Result<GenerateMode, String>
where
    I: IntoIterator,
    I::Item: Into<String>,
{
    let mut mode = GenerateMode::Manual;
    let mut args = args.into_iter().map(Into::into).skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--synth" => {
                let Some(value) = args.next() else {
                    return Err("--synth requires a value".into());
                };
                match value.as_str() {
                    "weighted" => mode = GenerateMode::SynthWeighted,
                    other => return Err(format!("unknown --synth value '{other}'")),
                }
            }
            "--help" | "-h" => {
                return Err("usage: generate [--synth weighted]".into());
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(mode)
}

fn main() {
    let mode = match parse_mode(std::env::args()) {
        Ok(mode) => mode,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    for (name, dialect, columns) in kernels::targets() {
        let (options, source) =
            if mode == GenerateMode::SynthWeighted && synth_formats::supports_weighted(&dialect) {
                (
                    CodegenOptions {
                        graph_source: GraphSource::SynthWeighted(SynthProfile::Weighted),
                    },
                    "synth-weighted",
                )
            } else {
                (CodegenOptions::default(), "manual")
            };

        let code = codegen::emit_parser_with_columns_options(&dialect, name, &columns, options)
            .expect("dialect should be emittable");
        let path = format!("{}/src/kernels/{name}.rs", env!("CARGO_MANIFEST_DIR"));
        std::fs::write(&path, code).expect("write generated kernel");
        println!("wrote {path} [{source}]");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_to_manual() {
        assert_eq!(parse_mode(["generate"]).unwrap(), GenerateMode::Manual);
    }

    #[test]
    fn parse_weighted_synth_mode() {
        assert_eq!(
            parse_mode(["generate", "--synth", "weighted"]).unwrap(),
            GenerateMode::SynthWeighted
        );
    }

    #[test]
    fn parse_rejects_unknown_synth_mode() {
        let err = parse_mode(["generate", "--synth", "tree"]).unwrap_err();
        assert!(err.contains("unknown --synth value"));
    }
}
```

- [ ] **Step 2: Run example compile checks**

Run:

```bash
cargo test --all-targets --quiet
```

Expected: all targets compile and tests pass. If Cargo does not run example unit tests, rely on compile coverage for `parse_mode` and the integration behavior covered by `tests/synth_codegen.rs`.

- [ ] **Step 3: Run default generation and confirm drift test remains clean**

Run:

```bash
cargo run --example generate
cargo test --test codegen generated_kernels_match_codegen --quiet
```

Expected: default generator writes `manual` entries and the drift test passes.

- [ ] **Step 4: Commit generator flag**

```bash
git add examples/generate.rs src/kernels
git commit -m "feat: add weighted synth generator option"
```

---

### Task 5: Document Opt-In Weighted Generation

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Update README generation docs**

Add this sentence near the kernel generation instructions:

```markdown
Kernel generation defaults to the handwritten format graph path. To try the weighted auto-discovery path for supported dialects, run `cargo run --example generate -- --synth weighted`; unsupported dialects such as comment-region CSV stay on the manual graph path in this opt-in mode.
```

- [ ] **Step 2: Update architecture docs**

Add this paragraph in the codegen or synthesis section:

```markdown
The generator now has two graph sources. The default source is the manual `formats::delimited_parts()` builder. The opt-in `--synth weighted` source runs weighted multi-output synthesis with the AVX cost model and feeds the resulting `DelimitedParts` into the same native AVX-512/AVX2 backend. Comment-region formats remain manual until the synth/prover path grows support for the `Regions` behavior.
```

- [ ] **Step 3: Run doc diff check**

Run:

```bash
git diff --check README.md ARCHITECTURE.md
```

Expected: no whitespace errors.

- [ ] **Step 4: Commit docs**

```bash
git add README.md ARCHITECTURE.md
git commit -m "docs: describe weighted synth generation"
```

---

### Task 6: Full Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt
```

Expected: command exits successfully.

- [ ] **Step 2: Run tests**

Run:

```bash
cargo test --quiet
```

Expected: all non-ignored tests pass.

- [ ] **Step 3: Run clippy**

Run:

```bash
cargo clippy --all-targets -- -D warnings
```

Expected: no warnings.

- [ ] **Step 4: Verify default generation still matches checked-in kernels**

Run:

```bash
cargo run --example generate
cargo test --test codegen generated_kernels_match_codegen --quiet
```

Expected: drift test passes.

- [ ] **Step 5: Smoke the opt-in generator and restore default artifacts**

Run:

```bash
cargo run --release --example generate -- --synth weighted
cargo run --example generate
cargo test --test codegen generated_kernels_match_codegen --quiet
```

Expected: the first command reports `synth-weighted` for supported targets and `manual` for `csv_hash`; the second command restores default manual generated artifacts; the drift test passes after restoration.

- [ ] **Step 6: Check emitted source contract**

Run:

```bash
rg -n "pub mod fallback|fallback::|__m512i|_mm512|avx512vl.*avx512vl" src/kernels src/codegen.rs tests examples README.md ARCHITECTURE.md
```

Expected: only deliberate negative-test string checks mention fallback; no generated fallback module or full-width `_mm512` code appears.

- [ ] **Step 7: Final status**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: only intentional files are modified or committed. The latest commits correspond to the plan tasks.
