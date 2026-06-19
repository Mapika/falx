# Contributing to Falx

## Development Setup

1. Install stable Rust: `rustup default stable`.
2. Clone the repository and navigate to the root.
3. Run tests: `cargo test` (all three executors differential-tested).
4. Benchmark: `cargo run --release --example bench` (throughput per format).

## Adding a Format Preset

New delimited formats take ~20 lines of code.

**Example: pipe-separated format** (hypothetical `|` delimiter, CSV-style quoting).

### 1. Define the dialect in `src/formats.rs`:

```rust
pub fn pipe_dialect() -> Dialect {
    Dialect {
        structural: vec![b'|', b'\n'],
        quote: Some(b'"'),
        escape: Escape::None,  // Doubled-quote (RFC 4180 style)
        comment: None,
        nesting: vec![],
    }
}
```

### 2. Add a spec file `specs/pipe.toml`:

```toml
name = "pipe"
structural = ["|", "\n"]
quote = "\""
escape = "doubled"
```

### 3. Wire it into the kernel registry `src/kernels/targets()`:

```rust
// src/kernels/mod.rs
pub mod pipe;

pub fn targets() -> Vec<(&'static str, Dialect, Vec<Column>)> {
    vec![
        // ...existing formats...
        ("pipe", formats::pipe_dialect(), vec![]),
        // The Vec<Column> declares typed columns, if any — see csv_typed.
    ]
}
```

Both `examples/generate.rs` and the drift test read this registry, so a
`cargo run --example generate` regenerates your kernel and the drift test
keeps it honest.

### 4. Add a differential test in `tests/codegen.rs`:

```rust
#[test]
fn pipe_format_differential() {
    let dialect = formats::pipe_dialect();
    let mut rng = Rng(0x1234_5678);
    
    for _ in 0..800 {
        let len = (rng.next() % 300) as usize;
        let data: Vec<u8> = (0..len)
            .map(|_| {
                let alphabet = b"|xyz\n\"";
                alphabet[(rng.next() % alphabet.len() as u64) as usize]
            })
            .collect();
        
        // Compare against scalar reference
        let mut expected = Vec::new();
        scalar::index_structurals_spec(&data, &[b'|', b'\n'], Some(b'"'), None, &mut expected);
        
        // Compare against generated kernel
        let mut actual = Vec::new();
        // ... (dispatch to falx::kernels::pipe once generated)
    }
}
```

### 5. Regenerate and test:

```bash
cargo run --example generate
cargo test
```

## Adding an IR Operation

Implement support for a new bitstream operation in all three executors.

**Checklist**:

1. **ir.rs**: Add variant to `enum Op` with doc comment explaining the bitstream semantics.
2. **ir.rs**: Add builder method on `Graph` (e.g., `pub fn my_op(&mut self, a: NodeId) -> NodeId`).
3. **interp.rs**: Add match arm in `Machine::step()` evaluating the operation over one block and threading any carry state.
4. **codegen.rs**: 
   - Allocate a carry slot if stateful (in the initial carry-slot loop).
   - Emit code in both `Flavor::Avx2` and `Flavor::Fallback` branches of `emit_step_body()`.
   - Scalar flavor should use portable logic; AVX2 flavor may use target-specific intrinsics.
5. **tests/ir.rs**: Add test covering block boundaries if the op is stateful (carries can hide bugs at seams).
6. **tests/codegen.rs**: Add differential test using randomized inputs.

Example: if adding a `PopCount(a)` operation to count set bits:

```rust
// ir.rs
PopCount(NodeId),

// interp.rs
Op::PopCount(a) => self.values[a.0 as usize].count_ones() as u64,

// codegen.rs (Fallback)
Op::PopCount(a) => {
    let v = self.values[a.0 as usize];
    format!("    let v{} = v{}.count_ones() as u64;", self.next_id, a.0)
}

// codegen.rs (Avx2)
Op::PopCount(a) => {
    // Intrinsics: _popcnt64 on x86_64
    format!("    let v{} = std::arch::x86_64::_popcnt64(v{} as i64) as u64;", self.next_id, a.0)
}
```

## Adding a Backend

Mirror the AVX2 template in `codegen.rs` to target a new architecture (e.g., ARM NEON).

**Key steps**:

1. Each ISA's kernel is emitted by `codegen.rs` as a self-contained `mod <arch>`
   submodule (no hand-written backend files); ARM NEON, for example, derives
   from the AVX2 template via `neon_driver` / `add_neon_dispatch`.
2. In `codegen.rs`, add a `Flavor` variant for the ISA and emit its code.
3. Implement the three arch-specific primitives:
   - **eq_mask(block, byte) -> u64**: Character-class membership (SIMD compare, reduce to bits).
   - **prefix_xor(x) -> u64**: Running parity. For NEON without PMUL, use scalar fallback or table-based approach.
   - **flatten(mask) -> Vec<u32>**: Extract set-bit positions (trailing-zero iteration).
4. Add `#[target_feature]` attributes and ensure `unsafe` blocks are justified.
5. Differential-test against scalar and existing backends.

## Code Style

- **Zero warnings**: `cargo clippy` and `rustfmt` must pass.
- **Regenerate kernels before committing**: `cargo run --example generate`.
- **Differential tests over unit tests**: Prefer randomized cross-backend validation to hand-coded asserts.
- **Carry-state clarity**: Comment carry semantics in IR ops and codegen—these are easy bugs.
- **Inline hot functions**: Step bodies and mask extraction should be `#[inline]`.

## Debugging Tips

- **Differential testing**: Run `cargo test` with `RUST_LOG=debug` to see mismatches.
- **IR visualization**: Print `graph.nodes()` and trace `Machine::step()` on a small input.
- **Codegen verification**: Check `target/debug/examples/generate` output against a known-good kernel.
- **Bench regression**: `cargo run --release --example bench` before and after changes.
