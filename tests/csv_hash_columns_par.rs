//! Differential tests for the csv_hash (comment + quote) parallel column path.
//!
//! csv_hash is the one dialect whose chunk entry state is a region
//! (NORMAL/QUOTE/COMMENT) rather than a quote parity, so `parse_columns_par`
//! resolves it with the three-phase transfer-function scheme instead of a
//! parity prefix. The hazards that scheme has to survive are exactly the ones
//! generated here: a quoted field holding `\n` or `#`, a comment line holding
//! an unbalanced `"`, and either straddling a 64-byte chunk boundary.
//!
//! Serial `parse_columns` is the reference — it is covered independently by the
//! codegen drift and interpreter-parity suites.

use falx::kernels::csv_hash;

mod common;
use common::Rng;

/// Compare every column of the serial and parallel paths for one input at one
/// thread count. Panics with the offending row on the first divergence.
fn assert_par_matches_serial(data: &[u8], threads: usize, label: &str) {
    let want = csv_hash::parse_columns(data);
    let got = csv_hash::parse_columns_par(data, threads);

    assert_eq!(
        want.rows, got.rows,
        "{label}: row count differs at {threads} threads"
    );
    for row in 0..want.rows {
        let want_key_valid = csv_hash::bitmap_get(&want.key_valid, row);
        let got_key_valid = csv_hash::bitmap_get(&got.key_valid, row);
        assert_eq!(
            want_key_valid, got_key_valid,
            "{label}: key validity differs at row {row}, {threads} threads"
        );
        if want_key_valid {
            assert_eq!(
                csv_hash::string_at(&want.key_offsets, &want.key_data, row),
                csv_hash::string_at(&got.key_offsets, &got.key_data, row),
                "{label}: key bytes differ at row {row}, {threads} threads"
            );
        }
        assert_eq!(
            csv_hash::bitmap_get(&want.amount_valid, row),
            csv_hash::bitmap_get(&got.amount_valid, row),
            "{label}: amount validity differs at row {row}, {threads} threads"
        );
        assert_eq!(
            want.amount[row], got.amount[row],
            "{label}: amount differs at row {row}, {threads} threads"
        );
    }
}

/// Hand-written cases covering each way region state can cross a boundary.
#[test]
fn par_matches_serial_on_region_hazards() {
    let cases: &[(&str, &[u8])] = &[
        ("plain", b"a,1\nb,2\nc,3\n"),
        ("leading comment", b"# header\na,1\nb,2\n"),
        ("trailing comment", b"a,1\n# tail\n"),
        ("comment only", b"# just a comment\n"),
        ("quoted newline", b"\"a\nb\",1\nc,2\n"),
        // A `#` inside a quoted field must NOT open a comment.
        ("hash inside quotes", b"\"a#b\",1\n\"#\",2\n"),
        // A `"` inside a comment must NOT open a quoted region.
        ("quote inside comment", b"# a \" b\na,1\n"),
        // Comment line whose unbalanced quote would flip parity if the region
        // machine ignored comment context.
        ("unbalanced quote in comment", b"# \"\na,1\nb,2\n"),
        ("quoted comma", b"\"a,b\",1\n"),
        ("doubled quotes", b"\"a\"\"b\",1\n"),
        ("empty cells", b",\n,\n"),
        ("no trailing newline", b"a,1\nb,2"),
        ("crlf", b"a,1\r\nb,2\r\n"),
        ("comment mid stream", b"a,1\n# mid\nb,2\n"),
    ];
    for (label, data) in cases {
        for &threads in &[1usize, 2, 3, 4, 8, 16] {
            assert_par_matches_serial(data, threads, label);
        }
    }
}

/// The same hazards, but padded so the hazard itself lands at every offset
/// around a 64-byte block and chunk boundary.
#[test]
fn par_matches_serial_across_chunk_boundaries() {
    let hazards: &[&[u8]] = &[
        b"\"x\ny\",7\n",
        b"# c \" c\n",
        b"\"#\",7\n",
        b"\"a\"\"b\",7\n",
    ];
    for hazard in hazards {
        for pad_rows in 0..40usize {
            let mut data = Vec::new();
            for i in 0..pad_rows {
                data.extend_from_slice(format!("k{i},{i}\n").as_bytes());
            }
            data.extend_from_slice(hazard);
            data.extend_from_slice(b"tail,99\n");
            for &threads in &[2usize, 4, 8] {
                assert_par_matches_serial(
                    &data,
                    threads,
                    &format!(
                        "pad {pad_rows} hazard {:?}",
                        String::from_utf8_lossy(hazard)
                    ),
                );
            }
        }
    }
}

/// Randomized documents mixing comments, quoted fields, embedded newlines and
/// hashes, checked at several thread counts.
#[test]
fn par_matches_serial_on_random_documents() {
    let mut rng = Rng(0x5EED_C5A5_1234_ABCD);
    for doc in 0..200 {
        let rows = 1 + (rng.next() % 60) as usize;
        let mut data = Vec::new();
        for _ in 0..rows {
            match rng.next() % 8 {
                0 => data.extend_from_slice(b"# comment with \" quote\n"),
                1 => data.extend_from_slice(b"# plain comment\n"),
                2 => data.extend_from_slice(b"\"quoted\nnewline\",42\n"),
                3 => data.extend_from_slice(b"\"has#hash\",7\n"),
                4 => data.extend_from_slice(b"\"a\"\"b\",13\n"),
                5 => data.extend_from_slice(b",\n"),
                6 => data.extend_from_slice(b"\"a,b\",5\n"),
                _ => {
                    let k = rng.next() % 1000;
                    let v = rng.next() % 100000;
                    data.extend_from_slice(format!("key{k},{v}\n").as_bytes());
                }
            }
        }
        if rng.next() % 4 == 0 {
            data.pop();
        }
        for &threads in &[2usize, 3, 5, 8, 17] {
            assert_par_matches_serial(&data, threads, &format!("random doc {doc}"));
        }
    }
}

/// Chunked output concatenates to exactly the flattened output.
#[test]
fn chunks_par_concatenates_to_parse_columns_par() {
    let mut data = Vec::new();
    for i in 0..5000 {
        match i % 7 {
            0 => data.extend_from_slice(b"# comment \"\n"),
            1 => data.extend_from_slice(b"\"multi\nline\",1\n"),
            2 => data.extend_from_slice(b"\"h#sh\",2\n"),
            _ => data.extend_from_slice(format!("k{i},{i}\n").as_bytes()),
        }
    }
    for &threads in &[2usize, 4, 16] {
        let flat = csv_hash::parse_columns_par(&data, threads);
        let chunks = csv_hash::parse_columns_chunks_par(&data, threads);
        assert_eq!(
            chunks.iter().map(|c| c.rows).sum::<usize>(),
            flat.rows,
            "chunk rows sum to the flattened row count at {threads} threads"
        );
        let mut row = 0;
        for chunk in &chunks {
            for r in 0..chunk.rows {
                assert_eq!(
                    csv_hash::bitmap_get(&chunk.amount_valid, r),
                    csv_hash::bitmap_get(&flat.amount_valid, row),
                    "amount validity at flattened row {row}, {threads} threads"
                );
                assert_eq!(
                    chunk.amount[r], flat.amount[row],
                    "amount at flattened row {row}, {threads} threads"
                );
                row += 1;
            }
        }
    }
}
