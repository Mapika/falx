use std::borrow::Cow;

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

#[test]
fn csv_fields_match_csv_crate() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);

    // Generate 300 random well-formed RFC 4180 CSV records.
    let mut csv_data = Vec::new();
    for _ in 0..300 {
        let num_fields = (rng.next() % 8 + 1) as usize; // 1..8 fields

        for field_idx in 0..num_fields {
            if field_idx > 0 {
                csv_data.push(b',');
            }

            let field_type = rng.next() % 3;
            match field_type {
                0 => {
                    // Plain alnum field (0..10 chars)
                    let len = (rng.next() % 11) as usize;
                    for _ in 0..len {
                        let idx = rng.next() % 26;
                        csv_data.push(b'a' + idx as u8);
                    }
                }
                1 => {
                    // Integer field
                    let val = rng.next() % 1000000;
                    for ch in val.to_string().bytes() {
                        csv_data.push(ch);
                    }
                }
                2 => {
                    // Quoted field with possible escapes, commas, newlines
                    csv_data.push(b'"');
                    let inner_len = (rng.next() % 20) as usize;
                    let mut inner = Vec::new();
                    for _ in 0..inner_len {
                        let ch_type = rng.next() % 5;
                        match ch_type {
                            0 => inner.push(b','),
                            1 => inner.push(b'"'),
                            2 => inner.push(b'\n'),
                            _ => {
                                let idx = rng.next() % 26;
                                inner.push(b'a' + idx as u8);
                            }
                        }
                    }
                    // Escape quotes by doubling them
                    for &byte in &inner {
                        csv_data.push(byte);
                        if byte == b'"' {
                            csv_data.push(b'"');
                        }
                    }
                    csv_data.push(b'"');
                }
                _ => unreachable!(),
            }
        }

        csv_data.push(b'\n');
    }

    // Parse with falx
    let parsed = falx::kernels::csv::parse(&csv_data);
    let falx_records: Vec<Vec<Vec<u8>>> = parsed
        .records()
        .map(|record| {
            (0..record.field_count())
                .map(|i| record.field(i).unwrap().to_vec())
                .collect()
        })
        .collect();

    // Parse with csv crate
    let csv_records: Vec<Vec<Vec<u8>>> = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(&csv_data[..])
        .records()
        .map(|r| {
            r.expect("csv record")
                .iter()
                .map(|f| f.as_bytes().to_vec())
                .collect()
        })
        .collect();

    assert_eq!(
        falx_records.len(),
        csv_records.len(),
        "record count mismatch: falx {} vs csv {}",
        falx_records.len(),
        csv_records.len()
    );

    for (i, (falx_rec, csv_rec)) in falx_records.iter().zip(csv_records.iter()).enumerate() {
        assert_eq!(
            falx_rec.len(),
            csv_rec.len(),
            "record {}: field count mismatch: falx {} vs csv {}",
            i,
            falx_rec.len(),
            csv_rec.len()
        );

        for (j, (falx_field, csv_field)) in falx_rec.iter().zip(csv_rec.iter()).enumerate() {
            assert_eq!(
                falx_field, csv_field,
                "record {} field {}: mismatch: falx {:?} vs csv {:?}",
                i, j, falx_field, csv_field
            );
        }
    }
}

#[test]
fn csv_hand_cases() {
    let input = b"a,\"b,c\",\"d\"\"e\"\r\nlast,row";
    let parsed = falx::kernels::csv::parse(input);
    let records: Vec<_> = parsed.records().collect();

    assert_eq!(records.len(), 2, "expected 2 records");

    // Record 0: a, "b,c", "d""e"
    let record0 = records[0];
    assert_eq!(record0.field_count(), 3, "record 0: field_count");

    let f0 = record0.field(0).expect("record 0 field 0");
    assert_eq!(f0.as_ref(), b"a");

    let f1 = record0.field(1).expect("record 0 field 1");
    assert_eq!(f1.as_ref(), b"b,c");

    let f2 = record0.field(2).expect("record 0 field 2");
    assert_eq!(f2.as_ref(), b"d\"e");

    // Check Cow::Borrowed for field(1) - no escape
    match f1 {
        Cow::Borrowed(_) => {
            // Good: field 1 had no escapes, so it should be borrowed
        }
        Cow::Owned(_) => {
            panic!("record 0 field 1 should be Borrowed (no escape collapsed)")
        }
    }

    // Check Cow::Owned for field(2) - escape collapsed
    match f2 {
        Cow::Owned(_) => {
            // Good: field 2 had escaped quote, should be owned
        }
        Cow::Borrowed(_) => {
            panic!("record 0 field 2 should be Owned (escape collapsed)")
        }
    }

    assert!(
        record0.field(3).is_none(),
        "record 0: field(3) should be None"
    );

    // Record 1: last, row
    let record1 = records[1];
    assert_eq!(record1.field_count(), 2, "record 1: field_count");
    assert_eq!(record1.field(0).expect("record 1 field 0").as_ref(), b"last");
    assert_eq!(record1.field(1).expect("record 1 field 1").as_ref(), b"row");
}

#[test]
fn backslash_hand_cases() {
    // logfmt: a="x \" y" b=plain\n
    // The backslash escapes the quote in the quoted field.
    let input = b"a=\"x \\\" y\" b=plain\n";
    let parsed = falx::kernels::logfmt::parse(input);
    let records: Vec<_> = parsed.records().collect();

    assert_eq!(records.len(), 1, "expected 1 record");

    let record = records[0];
    // logfmt fields: a, "x \" y", b, plain
    // But field count depends on separators (space, =). Let's check the fields.
    // The record content is: a="x \" y" b=plain
    // Separators are at unquoted spaces and '=' positions.

    let fields: Vec<_> = record.fields().map(|f| f.to_vec()).collect();

    // In logfmt, the fields are key-value pairs separated by spaces and =
    // a="x \" y" b=plain
    // Separators (unquoted): = at position 1, space at position 12, = at position 14
    // But we need to check what the actual fields are.

    assert_eq!(fields.len(), 4, "expected 4 fields: a, x \" y, b, plain");
    assert_eq!(fields[0], b"a");
    assert_eq!(fields[1], b"x \" y", "field 1: backslash should unescape \" to \"");
    assert_eq!(fields[2], b"b");
    assert_eq!(fields[3], b"plain");
}

#[test]
fn ndjson_records_match_line_split() {
    let mut rng = Rng(0xFEED_FACE_CAFE_BEEF);

    // Generate 200 random NDJSON lines.
    let mut ndjson_data = Vec::new();
    let mut expected_lines = Vec::new();

    for _ in 0..200 {
        let line_start = ndjson_data.len();
        ndjson_data.push(b'{');
        ndjson_data.extend_from_slice(b"\"k\":\"");

        let content_len = (rng.next() % 30) as usize;
        let mut content = Vec::new();
        for _ in 0..content_len {
            let ch_type = rng.next() % 5;
            match ch_type {
                0 => content.push(b'\\'),
                1 => content.push(b'"'),
                _ => {
                    let idx = rng.next() % 26;
                    content.push(b'a' + idx as u8);
                }
            }
        }

        // Escape backslashes and quotes for JSON
        for &byte in &content {
            if byte == b'\\' || byte == b'"' {
                ndjson_data.push(b'\\');
            }
            ndjson_data.push(byte);
        }

        ndjson_data.extend_from_slice(b"\"}\n");

        let line_end = ndjson_data.len() - 1; // Exclude \n
        expected_lines.push(ndjson_data[line_start..line_end].to_vec());
    }

    // Parse with falx
    let parsed = falx::kernels::ndjson::parse(&ndjson_data);
    let falx_records: Vec<_> = parsed
        .records()
        .map(|r| r.as_bytes().to_vec())
        .collect();

    // Parse with line split
    let split_records: Vec<_> = ndjson_data
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| line.to_vec())
        .collect();

    assert_eq!(
        falx_records.len(),
        split_records.len(),
        "record count mismatch"
    );

    for (i, (falx_rec, split_rec)) in falx_records.iter().zip(split_records.iter()).enumerate() {
        assert_eq!(
            falx_rec, split_rec,
            "record {}: byte mismatch: falx {:?} vs split {:?}",
            i, falx_rec, split_rec
        );

        // Each ndjson record should have field_count()==1
        let parsed = falx::kernels::ndjson::parse(&ndjson_data);
        let record = parsed.records().nth(i).expect("record exists");
        assert_eq!(
            record.field_count(),
            1,
            "record {}: field_count should be 1",
            i
        );
    }
}

#[test]
fn record_edge_cases() {
    // Empty input → 0 records
    {
        let input = b"";
        let parsed = falx::kernels::csv::parse(input);
        let records: Vec<_> = parsed.records().collect();
        assert_eq!(records.len(), 0, "empty input: expected 0 records");
    }

    // b"\n" → 1 record, field_count 1, field(0) == empty
    {
        let input = b"\n";
        let parsed = falx::kernels::csv::parse(input);
        let records: Vec<_> = parsed.records().collect();
        assert_eq!(records.len(), 1, "newline input: expected 1 record");

        let record = records[0];
        assert_eq!(record.field_count(), 1, "newline record: field_count");
        assert_eq!(
            record.field(0).expect("field 0").as_ref(),
            b"",
            "newline record: field(0) should be empty"
        );
    }

    // b"a,b" (no trailing newline) → 1 record with 2 fields
    {
        let input = b"a,b";
        let parsed = falx::kernels::csv::parse(input);
        let records: Vec<_> = parsed.records().collect();
        assert_eq!(
            records.len(),
            1,
            "no-newline input: expected 1 record"
        );

        let record = records[0];
        assert_eq!(record.field_count(), 2, "no-newline record: field_count");
        assert_eq!(record.field(0).expect("field 0").as_ref(), b"a");
        assert_eq!(record.field(1).expect("field 1").as_ref(), b"b");
    }

    // b"a\r\nb\r\n" → records [b"a"], [b"b"] (\r trimmed)
    {
        let input = b"a\r\nb\r\n";
        let parsed = falx::kernels::csv::parse(input);
        let records: Vec<_> = parsed.records().collect();
        assert_eq!(records.len(), 2, "crlf input: expected 2 records");

        let record0 = records[0];
        assert_eq!(
            record0.as_bytes(),
            b"a",
            "record 0: \\r should be trimmed"
        );

        let record1 = records[1];
        assert_eq!(
            record1.as_bytes(),
            b"b",
            "record 1: \\r should be trimmed"
        );
    }
}
