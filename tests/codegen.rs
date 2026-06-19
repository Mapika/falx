use falx::codegen;
use falx::formats::{self, Escape};
use falx::ir::{CharClass, Graph};
use falx::scalar;

mod common;
use common::Rng;

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
    assert!(generated.contains("pub fn parse_fastq_par("));
}

#[test]
fn generated_fastq_parallel_streams_without_newline_vector() {
    let target = falx::kernels::targets()
        .into_iter()
        .find(|(name, _, _)| *name == "fastq")
        .expect("fastq target is registered");
    let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
        .expect("fastq codegen should succeed");

    assert!(generated.contains("fn fastq_record_bounds("));
    assert!(generated.contains("fn fastq_find_record_boundary("));
    assert!(
        !generated.contains("fastq_count_newlines_dispatch("),
        "parallel FASTQ should find local record boundaries instead of scanning the whole input in a newline-count prepass"
    );
    assert!(generated.contains("checksum_lanes: [u64; 8]"));
    assert!(!generated.contains("checksum_adjust: u64"));
    assert!(!generated.contains("fn subtract_block_newlines(&mut self, mask: u64)"));
    assert!(!generated.contains("fn subtract_ignored_line_checksum(&mut self, bytes: &[u8])"));
    assert!(
        generated.contains("self.stats.checksum = finish_fastq_checksum(&self.checksum_lanes);")
    );
    assert!(!generated.contains("const FASTQ_BACKEND_AVX512: u8 = 1;"));
    assert!(!generated.contains("const FASTQ_BACKEND_AVX2: u8 = 2;"));
    assert!(!generated.contains("fn drive_backend<const BACKEND: u8>"));
    assert!(!generated.contains("fn drive_pair_backend<const BACKEND: u8>"));
    assert!(!generated.contains("fn line_backend<const BACKEND: u8>"));
    assert!(!generated.contains("fn record_backend<const BACKEND: u8>"));
    assert!(generated.contains("fn drive_pair_avx512("));
    assert!(generated.contains("fn drive_pair_avx2("));
    assert!(generated.contains("fn record_avx512("));
    assert!(generated.contains("fn record_avx2("));
    assert!(generated.contains("sink.drive_pair_avx512(data, m0, m1, offset, &mut checksum_acc)"));
    assert!(generated.contains("sink.drive_pair_avx2(data, m0, m1, offset, &mut checksum_acc)"));
    assert!(
        !generated.contains("unsafe fn fastq_step("),
        "FASTQ should not duplicate the generated step body for checksum work"
    );
    assert!(
        !generated.contains("let error = sink.drive_pair(data, m0, m1, offset);"),
        "FASTQ backend drivers should avoid per-record checksum feature dispatch"
    );
    assert!(generated.contains("fn finish(mut self) -> Result<FastqStats, FastqError>"));
    assert!(generated.contains("error_quality_len: usize"));
    assert!(generated.contains("fn take_error(&self, code: u8) -> FastqError"));
    assert!(generated.contains("unsafe { *data.get_unchecked(self.line_start) } != b'@'"));
    assert!(generated.contains("unsafe { *data.get_unchecked(self.line_start) } != b'+'"));
    assert!(
        !generated.contains("self.line_start >= data.len() || data[self.line_start]"),
        "FASTQ tag checks run only on newline-delimited lines and should avoid hot bounds checks"
    );
    assert!(generated.contains(
        "let error = unsafe { sink.drive_pair_avx512(data, m0, m1, offset, &mut checksum_acc) };\n            if error != 0 {\n                return Err(sink.take_error(error));\n            }"
    ));
    assert!(!generated.contains("fn checksum_fastq_record("));
    assert!(!generated.contains("fn checksum_fastq_record_avx512("));
    assert!(!generated.contains("fn checksum_fastq_record_avx2("));
    assert!(generated.contains("fn checksum_fastq_record_avx512_acc("));
    assert!(generated.contains("fn checksum_fastq_record_avx2_acc("));
    assert!(generated.contains(
        "data.get_unchecked(self.sequence_start..self.sequence_start + self.sequence_len)"
    ));
    assert!(generated.contains("data.get_unchecked(self.line_start..end)"));
    assert!(generated.contains("_mm512_sad_epu8("));
    assert!(generated.contains("_mm512_maskz_loadu_epi8("));
    assert!(generated.contains("_mm256_sad_epu8("));
    assert!(
        !generated.contains("fastq_stats_from_newlines_par"),
        "parallel FASTQ should stream generated masks into chunk sinks instead of validating from a newline-position vector"
    );
    assert!(
        !generated.contains("let mut newlines = Vec::"),
        "parallel FASTQ should not allocate a full newline-position vector"
    );
}

#[test]
fn generated_logfmt_target_exposes_fused_pair_api() {
    let target = falx::kernels::targets()
        .into_iter()
        .find(|(name, _, _)| *name == "logfmt")
        .expect("logfmt target is registered");
    let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
        .expect("logfmt codegen should succeed");

    assert!(generated.contains("pub struct LogfmtStats"));
    assert!(generated.contains("pub fn parse_logfmt_pairs("));
    assert!(generated.contains("pub fn parse_logfmt_pairs_par("));
    assert!(generated.contains("pub fn logfmt_blocks("));
    assert!(generated.contains("stats.key_bytes += key.len() as u64;"));
    assert!(generated.contains("fn add_logfmt_value("));
    assert!(generated.contains("value.first() != Some(&Q)"));
    assert!(generated.contains("has_key: bool"));
    assert!(generated.contains("#[inline(always)]\n    fn drive("));
    assert!(generated.contains("#[inline(always)]\n    fn field("));
    assert!(generated.contains("data.get_unchecked(self.key_start..self.key_end)"));
    assert!(generated.contains("data.get_unchecked(self.field_start..end)"));
    assert!(generated.contains("#[inline(always)]\nfn add_logfmt_pair("));
    assert!(
        !generated.contains("pending_key: Option"),
        "fused logfmt pairs should keep key state as plain fields in the hot sink"
    );
    assert!(
        !generated.contains("let key = clean(key);"),
        "fused logfmt pairs should not scan bare keys for quote/escape cleaning"
    );
    assert!(
        !generated.contains("clean(value)"),
        "fused logfmt pairs should count/checksum cleaned values without materializing them"
    );
}

#[test]
fn generated_vcf_typed_target_exposes_fused_stats_api() {
    let target = falx::kernels::targets()
        .into_iter()
        .find(|(name, _, _)| *name == "vcf_typed")
        .expect("vcf_typed target is registered");
    let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
        .expect("vcf_typed codegen should succeed");

    assert!(generated.contains("pub struct VcfStats"));
    assert!(generated.contains("pub fn parse_vcf_stats("));
    assert!(generated.contains("pub fn parse_vcf_stats_par("));
    assert!(generated.contains("pub(crate) fn index_vcf_stats("));
    assert!(generated.contains("fn vcf_checksum_primary_bytes("));
    assert!(generated.contains("fn parse_vcf_pos_cell("));
    assert!(
        generated.contains(
            "self.stats.checksum = vcf_checksum_primary_bytes(self.stats.checksum, bytes);"
        )
    );
    assert!(generated.contains("parse_vcf_pos_cell(&data[from as usize..to as usize])"));
}

#[test]
fn generated_csv_geo_targets_expose_fused_stats_api() {
    for (name, stats, serial, parallel, index) in [
        (
            "csv_geo",
            "pub struct CsvGeoStats",
            "pub fn parse_csv_geo_stats(",
            "pub fn parse_csv_geo_stats_par(",
            "pub(crate) fn index_csv_geo_stats(",
        ),
        (
            "csv_geo_text",
            "pub struct CsvGeoTextStats",
            "pub fn parse_csv_geo_text_stats(",
            "pub fn parse_csv_geo_text_stats_par(",
            "pub(crate) fn index_csv_geo_text_stats(",
        ),
    ] {
        let target = falx::kernels::targets()
            .into_iter()
            .find(|(target_name, _, _)| *target_name == name)
            .unwrap_or_else(|| panic!("{name} target is registered"));
        let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
            .unwrap_or_else(|_| panic!("{name} codegen should succeed"));

        assert!(generated.contains(stats), "{name} missing stats struct");
        assert!(
            generated.contains(serial),
            "{name} missing serial stats API"
        );
        assert!(
            generated.contains(parallel),
            "{name} missing parallel stats API"
        );
        assert!(
            generated.contains(index),
            "{name} missing generated stats indexer"
        );
        assert!(
            !generated.contains("fn parse_f64_field_bits("),
            "{name} should keep the proven f64 field parser path"
        );
        assert!(
            generated.contains("wrapping_add(value.to_bits())"),
            "{name} should keep the proven checksum path"
        );
    }
}

#[test]
fn generated_float_columns_emit_fixed_six_decimal_fast_path() {
    let target = falx::kernels::targets()
        .into_iter()
        .find(|(name, _, _)| *name == "csv_geo")
        .expect("csv_geo target is registered");
    let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
        .expect("csv_geo codegen should succeed");

    assert!(generated.contains("#[inline(always)]\nfn parse_f64_cell("));
    assert!(generated.contains("fn parse_fixed_6_decimal_signed("));
    assert!(generated.contains("if let Some((neg, mantissa)) = parse_fixed_6_decimal_signed(s)"));
}

#[test]
fn generated_csv_tsv_targets_expose_fused_field_byte_api() {
    for name in ["csv", "tsv"] {
        let target = falx::kernels::targets()
            .into_iter()
            .find(|(target_name, _, _)| *target_name == name)
            .unwrap_or_else(|| panic!("{name} target is registered"));
        let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
            .unwrap_or_else(|_| panic!("{name} codegen should succeed"));

        assert!(generated.contains("pub struct FieldByteStats"));
        assert!(generated.contains("pub fn parse_field_bytes("));
        assert!(generated.contains("pub fn parse_field_bytes_par("));
        assert!(generated.contains("pub(crate) fn index_field_bytes("));
    }
}

#[test]
fn generated_csv_stats_paths_use_simd_quote_parity() {
    let target = falx::kernels::targets()
        .into_iter()
        .find(|(name, _, _)| *name == "csv")
        .expect("csv target is registered");
    let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
        .expect("csv codegen should succeed");

    assert!(generated.contains("fn quote_parity_dispatch("));
    assert!(generated.contains("quote_parity_dispatch(&data[start..end], 0)"));
    assert!(
        !generated.contains("iter().filter(|&&b| b == 34u8).count()"),
        "parallel stats paths should not run a scalar quote-count prepass"
    );
}

#[test]
fn generated_csv_geo_stats_paths_use_simd_quote_parity() {
    for name in ["csv_geo", "csv_geo_text"] {
        let target = falx::kernels::targets()
            .into_iter()
            .find(|(target_name, _, _)| *target_name == name)
            .unwrap_or_else(|| panic!("{name} target is registered"));
        let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
            .unwrap_or_else(|_| panic!("{name} codegen should succeed"));

        assert!(
            generated.contains("fn quote_parity_dispatch("),
            "{name} missing generated quote-parity dispatch"
        );
        assert!(
            generated.contains("quote_parity_dispatch(&data[start..end], 0)"),
            "{name} should seed stats workers from generated SIMD quote parity"
        );
        assert!(
            !generated.contains("iter().filter(|&&b| b == 34u8).count()"),
            "{name} stats paths should not run a scalar quote-count prepass"
        );
    }
}

#[test]
fn generated_csv_geo_text_stats_specializes_short_city_checksums() {
    let target = falx::kernels::targets()
        .into_iter()
        .find(|(name, _, _)| *name == "csv_geo_text")
        .expect("csv_geo_text target is registered");
    let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
        .expect("csv_geo_text codegen should succeed");

    assert!(generated.contains("fn csv_geo_checksum_first_8("));
    assert!(generated.contains("match bytes.len()"));
}

#[test]
fn generated_ndjson_target_exposes_fused_line_stats_api() {
    let target = falx::kernels::targets()
        .into_iter()
        .find(|(name, _, _)| *name == "ndjson")
        .expect("ndjson target is registered");
    let generated = codegen::emit_parser_with_columns(&target.1, target.0, &target.2)
        .expect("ndjson codegen should succeed");

    assert!(generated.contains("pub struct NdjsonLineStats"));
    assert!(generated.contains("pub fn parse_ndjson_lines("));
    assert!(generated.contains("pub fn parse_ndjson_lines_par("));
    assert!(generated.contains("pub fn index_ndjson_lines("));
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
            let cells = if long {
                40 + rng.next() % 60
            } else {
                1 + rng.next() % 6
            };
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
            assert_eq!(
                par_fields, serial_fields,
                "vcf parse_par mismatch at {threads} threads"
            );
        }
    }
}
