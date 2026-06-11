use falx::codegen;
use falx::formats::{self, Escape};
use falx::ir::{CharClass, Graph};
use falx::scalar;

/// xorshift64* RNG; avoids a dev-dependency for test data generation.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Test 1: Generated kernels match codegen output.
/// Verifies that the checked-in kernel files match what the code generator
/// currently emits, for every entry in the kernel registry (typed-column
/// kernels included).
#[test]
fn generated_kernels_match_codegen() {
    for (name, dialect, columns) in falx::kernels::targets() {
        let generated = codegen::emit_parser_with_columns(&dialect, name, &columns)
            .expect("codegen should succeed");

        let path = format!("{}/src/kernels/{}.rs", env!("CARGO_MANIFEST_DIR"), name);
        let expected = std::fs::read_to_string(&path)
            .expect("failed to read checked-in kernel file");

        assert_eq!(
            generated, expected,
            "Generated code for {} does not match checked-in kernel at {}.\n\
             This is a drift test: run `cargo run --example generate` to regenerate.",
            name, path
        );
    }
}

/// Test 2: Generated kernels differential test.
/// Runs 800 randomized inputs through generated kernels and compares against scalar reference.
#[test]
fn generated_kernels_differential() {
    let formats = [
        ("csv", formats::csv_dialect()),
        ("tsv", formats::tsv_dialect()),
        ("logfmt", formats::logfmt_dialect()),
        ("ndjson", formats::ndjson_dialect()),
        ("multi", formats::multi_dialect()),
        ("csv_hash", formats::csv_hash_dialect()),
    ];

    let mut rng = Rng(0xDEAD_BEEF_CAFE_BABE);

    for (name, dialect) in &formats {
        // Build alphabet: all structural bytes, quote byte (twice), escape byte (thrice), plus filler.
        let mut alphabet = Vec::new();
        alphabet.extend_from_slice(&dialect.structural);
        if let Some(q) = dialect.quote {
            alphabet.push(q);
            alphabet.push(q);
        }
        if let Escape::Backslash(esc) = dialect.escape {
            alphabet.push(esc);
            alphabet.push(esc);
            alphabet.push(esc);
        }
        if let Some(c) = dialect.comment {
            alphabet.push(c);
            alphabet.push(c);
        }
        alphabet.extend_from_slice(b"xy");

        // Convert Escape enum to Option for scalar reference.
        let escape_byte = match dialect.escape {
            Escape::None => None,
            Escape::Backslash(b) => Some(b),
        };

        for test_num in 0..800 {
            let len = (rng.next() % 300) as usize;
            let data: Vec<u8> = (0..len)
                .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                .collect();

            // Get expected output from scalar reference.
            let mut expected = Vec::new();
            scalar::index_structurals_spec(
                &data,
                &dialect.structural,
                dialect.quote,
                escape_byte,
                dialect.comment,
                &mut expected,
            );

            // Test dispatched kernel.
            let mut dispatched = Vec::new();
            match *name {
                "csv" => falx::kernels::csv::index_structurals(&data, &mut dispatched),
                "tsv" => falx::kernels::tsv::index_structurals(&data, &mut dispatched),
                "logfmt" => falx::kernels::logfmt::index_structurals(&data, &mut dispatched),
                "ndjson" => falx::kernels::ndjson::index_structurals(&data, &mut dispatched),
                "multi" => falx::kernels::multi::index_structurals(&data, &mut dispatched),
                "csv_hash" => falx::kernels::csv_hash::index_structurals(&data, &mut dispatched),
                _ => panic!("unknown format: {}", name),
            }

            // Test fallback kernel.
            let mut fallback = Vec::new();
            match *name {
                "csv" => falx::kernels::csv::fallback::index_structurals(&data, &mut fallback),
                "tsv" => falx::kernels::tsv::fallback::index_structurals(&data, &mut fallback),
                "logfmt" => falx::kernels::logfmt::fallback::index_structurals(&data, &mut fallback),
                "ndjson" => falx::kernels::ndjson::fallback::index_structurals(&data, &mut fallback),
                "multi" => falx::kernels::multi::fallback::index_structurals(&data, &mut fallback),
                "csv_hash" => falx::kernels::csv_hash::fallback::index_structurals(&data, &mut fallback),
                _ => panic!("unknown format: {}", name),
            }

            assert_eq!(
                dispatched, expected,
                "dispatched kernel mismatch for {} test {}: dispatched={:?}, expected={:?}",
                name, test_num, dispatched, expected
            );
            assert_eq!(
                fallback, expected,
                "fallback kernel mismatch for {} test {}: fallback={:?}, expected={:?}",
                name, test_num, fallback, expected
            );
        }
    }
}

/// Test 3: Generated kernels with long input.
/// Each format tested with a 100_000 byte randomized input; all three paths (dispatched, fallback, scalar) must agree.
#[test]
fn generated_kernels_long_input() {
    let formats = [
        ("csv", formats::csv_dialect()),
        ("tsv", formats::tsv_dialect()),
        ("logfmt", formats::logfmt_dialect()),
        ("ndjson", formats::ndjson_dialect()),
        ("multi", formats::multi_dialect()),
        ("csv_hash", formats::csv_hash_dialect()),
    ];

    let mut rng = Rng(0xCAFE_BABE_DEAD_BEEF);

    for (name, dialect) in &formats {
        // Build alphabet.
        let mut alphabet = Vec::new();
        alphabet.extend_from_slice(&dialect.structural);
        if let Some(q) = dialect.quote {
            alphabet.push(q);
            alphabet.push(q);
        }
        if let Escape::Backslash(esc) = dialect.escape {
            alphabet.push(esc);
            alphabet.push(esc);
            alphabet.push(esc);
        }
        if let Some(c) = dialect.comment {
            alphabet.push(c);
            alphabet.push(c);
        }
        alphabet.extend_from_slice(b"xy");

        let escape_byte = match dialect.escape {
            Escape::None => None,
            Escape::Backslash(b) => Some(b),
        };

        // Generate 100_000 byte input.
        let data: Vec<u8> = (0..100_000)
            .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
            .collect();

        // Get expected output from scalar reference.
        let mut expected = Vec::new();
        scalar::index_structurals_spec(
            &data,
            &dialect.structural,
            dialect.quote,
            escape_byte,
            dialect.comment,
            &mut expected,
        );

        // Test dispatched kernel.
        let mut dispatched = Vec::new();
        match *name {
            "csv" => falx::kernels::csv::index_structurals(&data, &mut dispatched),
            "tsv" => falx::kernels::tsv::index_structurals(&data, &mut dispatched),
            "logfmt" => falx::kernels::logfmt::index_structurals(&data, &mut dispatched),
            "ndjson" => falx::kernels::ndjson::index_structurals(&data, &mut dispatched),
            "multi" => falx::kernels::multi::index_structurals(&data, &mut dispatched),
            "csv_hash" => falx::kernels::csv_hash::index_structurals(&data, &mut dispatched),
            _ => panic!("unknown format: {}", name),
        }

        // Test fallback kernel.
        let mut fallback = Vec::new();
        match *name {
            "csv" => falx::kernels::csv::fallback::index_structurals(&data, &mut fallback),
            "tsv" => falx::kernels::tsv::fallback::index_structurals(&data, &mut fallback),
            "logfmt" => falx::kernels::logfmt::fallback::index_structurals(&data, &mut fallback),
            "ndjson" => falx::kernels::ndjson::fallback::index_structurals(&data, &mut fallback),
            "multi" => falx::kernels::multi::fallback::index_structurals(&data, &mut fallback),
            "csv_hash" => falx::kernels::csv_hash::fallback::index_structurals(&data, &mut fallback),
            _ => panic!("unknown format: {}", name),
        }

        assert_eq!(dispatched, expected, "dispatched kernel mismatch for {} long input", name);
        assert_eq!(fallback, expected, "fallback kernel mismatch for {} long input", name);
    }
}

/// Test 4: Large classes go through the shuffle classifier; only classes
/// that cannot be decomposed into PSHUFB nibble tables are rejected.
#[test]
fn codegen_classifies_large_classes() {
    // 10 digits: trivially decomposable (one hi-nibble row), must emit.
    let mut g = Graph::new();
    let digits = g.class(CharClass::from_bytes(b"0123456789"));
    g.set_output(digits);
    let code = codegen::emit(&g, "digit_class_test").expect("digit class should emit");
    assert!(
        code.contains("table_mask"),
        "a 10-byte class should use the shuffle/table classifier"
    );

    // Pathological class: 9 hi-nibble rows with 9 *distinct* lo-nibble
    // sets (row h holds lo nibbles 0..=h) — needs 9 table bits, which
    // PSHUFB's 8-bit lanes cannot encode.
    let mut bytes = Vec::new();
    for h in 0u8..9 {
        for l in 0..=h {
            bytes.push(h * 16 + l);
        }
    }
    let mut g = Graph::new();
    let undecomposable = g.class(CharClass::from_bytes(&bytes));
    g.set_output(undecomposable);
    let err = codegen::emit(&g, "undecomposable_test")
        .expect_err("9 distinct row patterns should be rejected");
    assert!(
        err.to_string().contains("distinct"),
        "error should explain the row-pattern limit, got: {}",
        err
    );
}

/// Parallel indexing must be byte-identical to serial for every thread
/// count, including quoted regions that span chunk boundaries.
#[test]
fn parallel_index_matches_serial() {
    let mut rng = Rng(0x0DDB_A115_0DDB_A115);
    let alphabet = b"\",\n\rxy";
    for _ in 0..40 {
        let len = 4096 + (rng.next() % 200_000) as usize;
        let data: Vec<u8> = (0..len)
            .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
            .collect();
        let mut serial = Vec::new();
        falx::kernels::csv::index_structurals(&data, &mut serial);
        for threads in [1, 2, 3, 7, 16] {
            let mut par = Vec::new();
            falx::kernels::csv::index_structurals_par(&data, threads, &mut par);
            assert_eq!(par, serial, "csv par mismatch at {threads} threads, len {len}");
            let mut par_tsv = Vec::new();
            falx::kernels::tsv::index_structurals_par(&data, threads, &mut par_tsv);
            let mut serial_tsv = Vec::new();
            falx::kernels::tsv::index_structurals(&data, &mut serial_tsv);
            assert_eq!(par_tsv, serial_tsv, "tsv par mismatch at {threads} threads");
        }
    }
}
