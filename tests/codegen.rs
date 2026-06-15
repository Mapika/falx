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
        let expected =
            std::fs::read_to_string(&path).expect("failed to read checked-in kernel file");

        assert_eq!(
            generated, expected,
            "Generated code for {} does not match checked-in kernel at {}.\n\
             This is a drift test: run `cargo run --example generate` to regenerate.",
            name, path
        );
    }
}

/// Generated kernels are native SIMD artifacts: the emitted source should
/// contain only runtime-selected x86 SIMD backends, not a portable fallback
/// module.
#[test]
fn generated_kernels_emit_native_simd_without_fallback() {
    for (name, dialect, columns) in falx::kernels::targets() {
        let generated = codegen::emit_parser_with_columns(&dialect, name, &columns)
            .expect("codegen should succeed");

        assert!(
            !generated.contains("pub mod fallback"),
            "{name} still emits the portable fallback module"
        );
        assert!(
            !generated.contains("fallback::"),
            "{name} still dispatches to the portable fallback path"
        );
        assert!(
            generated.contains("mod avx512"),
            "{name} does not emit the native AVX-512 backend"
        );
        assert!(
            generated
                .find("avx512::")
                .expect("AVX-512 dispatch present")
                < generated.find("avx2::").expect("AVX2 dispatch present"),
            "{name} should prefer the AVX-512 backend before AVX2"
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

            assert_eq!(
                dispatched, expected,
                "dispatched kernel mismatch for {} test {}: dispatched={:?}, expected={:?}",
                name, test_num, dispatched, expected
            );
        }
    }
}

/// Test 3: Generated kernels with long input.
/// Each format tested with a 100_000 byte randomized input; dispatched
/// generated SIMD must agree with the independent scalar reference.
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

        assert_eq!(
            dispatched, expected,
            "dispatched kernel mismatch for {} long input",
            name
        );
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
            assert_eq!(
                par, serial,
                "csv par mismatch at {threads} threads, len {len}"
            );
            let mut par_tsv = Vec::new();
            falx::kernels::tsv::index_structurals_par(&data, threads, &mut par_tsv);
            let mut serial_tsv = Vec::new();
            falx::kernels::tsv::index_structurals(&data, &mut serial_tsv);
            assert_eq!(par_tsv, serial_tsv, "tsv par mismatch at {threads} threads");
        }
    }
}

/// Comment dialects parallelize by line ownership; their parallel output must
/// match serial for any thread count, including when a (possibly long) comment
/// line straddles a chunk boundary. Uses the VCF dialect (tab + `#`, no quote).
#[test]
fn comment_parallel_matches_serial() {
    let mut rng = Rng(0x7A57_7A57_C0DE_F00D);
    for _ in 0..40 {
        let target = 4096 + (rng.next() % 200_000) as usize;
        let mut data: Vec<u8> = Vec::with_capacity(target + 512);
        while data.len() < target {
            // ~1 in 5 lines is a `#` comment; lines vary in length, and ~1 in
            // 16 is long enough to straddle chunk boundaries at high thread
            // counts. Comment lines carry tabs and `#` (inert) on purpose.
            let long = rng.next().is_multiple_of(16);
            let cells = if long { 40 + rng.next() % 60 } else { 1 + rng.next() % 6 };
            if rng.next().is_multiple_of(5) {
                data.push(b'#');
                for _ in 0..cells {
                    data.push(*b"abc\t#9".get((rng.next() % 6) as usize).unwrap());
                }
            } else {
                for f in 0..cells {
                    if f > 0 {
                        data.push(b'\t');
                    }
                    for _ in 0..(rng.next() % 8) {
                        data.push(*b"abcd".get((rng.next() % 4) as usize).unwrap());
                    }
                }
            }
            data.push(b'\n');
        }
        let mut serial = Vec::new();
        falx::kernels::vcf::index_structurals(&data, &mut serial);
        let serial_fields: Vec<Vec<Vec<u8>>> = falx::kernels::vcf::parse(&data)
            .records()
            .map(|r| r.fields().map(|f| f.into_owned()).collect())
            .collect();
        for threads in [1usize, 2, 3, 7, 16, 32] {
            let mut par = Vec::new();
            falx::kernels::vcf::index_structurals_par(&data, threads, &mut par);
            assert_eq!(par, serial, "vcf index_par mismatch at {threads} threads");
            let par_fields: Vec<Vec<Vec<u8>>> = falx::kernels::vcf::parse_par(&data, threads)
                .records()
                .map(|r| r.fields().map(|f| f.into_owned()).collect())
                .collect();
            assert_eq!(par_fields, serial_fields, "vcf parse_par mismatch at {threads} threads");
        }
    }
}
