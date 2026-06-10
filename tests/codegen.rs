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
/// Verifies that the checked-in kernel files match what the code generator currently emits.
#[test]
fn generated_kernels_match_codegen() {
    let formats = [
        ("csv", formats::csv_dialect()),
        ("tsv", formats::tsv_dialect()),
        ("logfmt", formats::logfmt_dialect()),
        ("ndjson", formats::ndjson_dialect()),
    ];

    for (name, dialect) in &formats {
        let generated = codegen::emit_parser(dialect, name).expect("codegen should succeed");

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
                &mut expected,
            );

            // Test dispatched kernel.
            let mut dispatched = Vec::new();
            match name {
                &"csv" => falx::kernels::csv::index_structurals(&data, &mut dispatched),
                &"tsv" => falx::kernels::tsv::index_structurals(&data, &mut dispatched),
                &"logfmt" => falx::kernels::logfmt::index_structurals(&data, &mut dispatched),
                &"ndjson" => falx::kernels::ndjson::index_structurals(&data, &mut dispatched),
                _ => panic!("unknown format: {}", name),
            }

            // Test fallback kernel.
            let mut fallback = Vec::new();
            match name {
                &"csv" => falx::kernels::csv::fallback::index_structurals(&data, &mut fallback),
                &"tsv" => falx::kernels::tsv::fallback::index_structurals(&data, &mut fallback),
                &"logfmt" => falx::kernels::logfmt::fallback::index_structurals(&data, &mut fallback),
                &"ndjson" => falx::kernels::ndjson::fallback::index_structurals(&data, &mut fallback),
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
            &mut expected,
        );

        // Test dispatched kernel.
        let mut dispatched = Vec::new();
        match name {
            &"csv" => falx::kernels::csv::index_structurals(&data, &mut dispatched),
            &"tsv" => falx::kernels::tsv::index_structurals(&data, &mut dispatched),
            &"logfmt" => falx::kernels::logfmt::index_structurals(&data, &mut dispatched),
            &"ndjson" => falx::kernels::ndjson::index_structurals(&data, &mut dispatched),
            _ => panic!("unknown format: {}", name),
        }

        // Test fallback kernel.
        let mut fallback = Vec::new();
        match name {
            &"csv" => falx::kernels::csv::fallback::index_structurals(&data, &mut fallback),
            &"tsv" => falx::kernels::tsv::fallback::index_structurals(&data, &mut fallback),
            &"logfmt" => falx::kernels::logfmt::fallback::index_structurals(&data, &mut fallback),
            &"ndjson" => falx::kernels::ndjson::fallback::index_structurals(&data, &mut fallback),
            _ => panic!("unknown format: {}", name),
        }

        assert_eq!(dispatched, expected, "dispatched kernel mismatch for {} long input", name);
        assert_eq!(fallback, expected, "fallback kernel mismatch for {} long input", name);
    }
}

/// Test 4: Codegen rejects oversized character classes.
/// Verifies that emit() returns Err when a character class exceeds MAX_CLASS_BYTES.
#[test]
fn codegen_rejects_oversized_class() {
    // Create a graph with a character class larger than 8 bytes.
    // MAX_CLASS_BYTES is 8, so we use 9 bytes.
    let mut g = Graph::new();
    let large_class = g.class(CharClass::from_bytes(b"0123456789")); // 10 bytes
    g.set_output(large_class);

    let result = codegen::emit(&g, "oversized_test");

    assert!(
        result.is_err(),
        "codegen should reject a character class with 10 bytes (exceeds MAX_CLASS_BYTES of 8)"
    );

    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("exceeds the compare-based limit"),
        "error message should mention exceeding the compare-based limit, got: {}",
        err
    );
}
