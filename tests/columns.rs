/// Differential test suite for the typed columnar API of falx::kernels::csv_typed.
///
/// Reference implementation: a dumb scalar parser that independently decodes CSV
/// records and columns without calling any falx functions. Tests compare the
/// library output against this reference via the exact API contract.

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

/// Reference implementation of CSV record splitting and field extraction.
/// Mirrors the library's behavior exactly.
fn ref_parse_records(data: &[u8]) -> Vec<Vec<&[u8]>> {
    let mut records = Vec::new();
    let mut current_record = Vec::new();
    let mut field_start = 0;
    let mut in_quotes = false;

    let mut pos = 0;
    while pos < data.len() {
        let byte = data[pos];
        match byte {
            b'"' => {
                in_quotes = !in_quotes;
            }
            b',' if !in_quotes => {
                // Field separator
                current_record.push(&data[field_start..pos]);
                field_start = pos + 1;
            }
            b'\n' if !in_quotes => {
                // Record terminator
                let mut rec_end = pos;

                // Trim trailing \r if present
                if rec_end > field_start && data[rec_end - 1] == b'\r' {
                    rec_end -= 1;
                }

                current_record.push(&data[field_start..rec_end]);
                records.push(current_record);
                current_record = Vec::new();
                field_start = pos + 1;
            }
            _ => {}
        }
        pos += 1;
    }

    // Handle trailing unterminated record
    if field_start < data.len() {
        let mut rec_end = data.len();

        // Trim trailing \r if present
        if rec_end > field_start && data[rec_end - 1] == b'\r' {
            rec_end -= 1;
        }

        // Only add if there's content
        if rec_end > field_start || !current_record.is_empty() {
            current_record.push(&data[field_start..rec_end]);
            if !current_record.is_empty() {
                records.push(current_record);
            }
        }
    } else if !current_record.is_empty() {
        // Final record with trailing comma and no more content: add empty final field
        current_record.push(&data[0..0]); // Empty field
        records.push(current_record);
    }

    records
}

/// Strip outer quotes and unescape doubled quotes.
fn ref_clean_cell(raw: &[u8]) -> Cow<'_, [u8]> {
    const Q: u8 = b'"';
    if raw.len() >= 2 && raw[0] == Q && raw[raw.len() - 1] == Q {
        let inner = &raw[1..raw.len() - 1];
        if !inner.windows(2).any(|w| w[0] == Q && w[1] == Q) {
            return Cow::Borrowed(inner);
        }
        let mut out = Vec::with_capacity(inner.len());
        let mut i = 0;
        while i < inner.len() {
            out.push(inner[i]);
            if inner[i] == Q && i + 1 < inner.len() && inner[i + 1] == Q {
                i += 2;
            } else {
                i += 1;
            }
        }
        return Cow::Owned(out);
    }
    Cow::Borrowed(raw)
}

/// Parse i64 from a cell with str::parse semantics.
fn ref_parse_i64(s: &[u8]) -> Option<i64> {
    std::str::from_utf8(s).ok()?.parse::<i64>().ok()
}

/// Parse f64 from a cell with str::parse semantics.
fn ref_parse_f64(s: &[u8]) -> Option<f64> {
    std::str::from_utf8(s).ok()?.parse::<f64>().ok()
}

/// Reference columnar parser: builds expected columns without calling falx.
fn ref_parse_columns(data: &[u8]) -> RefColumns {
    let records = ref_parse_records(data);
    let rows = records.len();

    let mut id = Vec::with_capacity(rows);
    let mut id_valid: Vec<u64> = Vec::new();
    let mut title_offsets = vec![0i32];
    let mut title_data = Vec::new();
    let mut title_valid: Vec<u64> = Vec::new();
    let mut value = Vec::with_capacity(rows);
    let mut value_valid: Vec<u64> = Vec::new();
    let mut label = Vec::with_capacity(rows);
    let mut label_valid: Vec<u64> = Vec::new();

    for (row, record) in records.iter().enumerate() {
        // Allocate new validity bitmap word when starting a new 64-row block
        if row & 63 == 0 {
            id_valid.push(0);
            title_valid.push(0);
            value_valid.push(0);
            label_valid.push(0);
        }

        // Field 0 (id): i64
        if record.len() > 0 {
            let raw = record[0];
            let cleaned = ref_clean_cell(raw);
            if let Some(v) = ref_parse_i64(&cleaned) {
                id.push(v);
                id_valid[row >> 6] |= 1u64 << (row & 63);
            } else {
                id.push(0i64);
            }
        } else {
            id.push(0i64);
        }

        // Field 1 (title): string (Arrow varbinary layout)
        let title_ok = record.len() > 1;
        if title_ok {
            let raw = record[1];
            let cleaned = ref_clean_cell(raw);
            title_data.extend_from_slice(&cleaned);
            title_valid[row >> 6] |= 1u64 << (row & 63);
        }
        title_offsets.push(title_data.len() as i32);

        // Field 2 (value): f64
        if record.len() > 2 {
            let raw = record[2];
            let cleaned = ref_clean_cell(raw);
            if let Some(v) = ref_parse_f64(&cleaned) {
                value.push(v);
                value_valid[row >> 6] |= 1u64 << (row & 63);
            } else {
                value.push(0.0f64);
            }
        } else {
            value.push(0.0f64);
        }

        // Field 4 (label): bytes (raw, non-empty)
        if record.len() > 4 && !record[4].is_empty() {
            let start = record[4].as_ptr() as usize - data.as_ptr() as usize;
            let end = start + record[4].len();
            label.push((start as u32, end as u32));
            label_valid[row >> 6] |= 1u64 << (row & 63);
        } else {
            label.push((0u32, 0u32));
        }
    }

    RefColumns {
        rows,
        id,
        id_valid,
        title_offsets,
        title_data,
        title_valid,
        value,
        value_valid,
        label,
        label_valid,
    }
}

#[derive(Debug)]
struct RefColumns {
    rows: usize,
    id: Vec<i64>,
    id_valid: Vec<u64>,
    title_offsets: Vec<i32>,
    title_data: Vec<u8>,
    title_valid: Vec<u64>,
    value: Vec<f64>,
    value_valid: Vec<u64>,
    label: Vec<(u32, u32)>,
    label_valid: Vec<u64>,
}

/// Compare library columns against reference; show input if mismatch.
fn assert_columns_match(
    data: &[u8],
    actual: &falx::kernels::csv_typed::Columns,
    expected: &RefColumns,
) {
    let data_str = String::from_utf8_lossy(data);

    assert_eq!(
        actual.rows, expected.rows,
        "rows mismatch for input:\n{}",
        data_str
    );
    assert_eq!(
        actual.id, expected.id,
        "id values mismatch for input:\n{}",
        data_str
    );
    assert_eq!(
        actual.id_valid, expected.id_valid,
        "id_valid bitmap mismatch for input:\n{}",
        data_str
    );

    // Compare title columns
    assert_eq!(
        actual.title_offsets, expected.title_offsets,
        "title_offsets mismatch for input:\n{}",
        data_str
    );
    assert_eq!(
        actual.title_data, expected.title_data,
        "title_data mismatch for input:\n{}",
        data_str
    );
    assert_eq!(
        actual.title_valid, expected.title_valid,
        "title_valid bitmap mismatch for input:\n{}",
        data_str
    );

    // For every valid title, check string_at contents
    for row in 0..expected.rows {
        if falx::kernels::csv_typed::bitmap_get(&expected.title_valid, row) {
            let exp_bytes = &expected.title_data[expected.title_offsets[row] as usize
                ..expected.title_offsets[row + 1] as usize];
            let act_bytes =
                falx::kernels::csv_typed::string_at(&actual.title_offsets, &actual.title_data, row);
            assert_eq!(
                act_bytes, exp_bytes,
                "title string_at[{}] contents mismatch for input:\n{}",
                row,
                data_str
            );
        }
    }

    // Compare f64 by bit pattern to handle NaN and -0.0
    assert_eq!(
        actual.value.len(),
        expected.value.len(),
        "value len mismatch for input:\n{}",
        data_str
    );
    for (i, (a, e)) in actual.value.iter().zip(expected.value.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "value[{}] bit mismatch ({} vs {}) for input:\n{}",
            i,
            a,
            e,
            data_str
        );
    }

    assert_eq!(
        actual.value_valid, expected.value_valid,
        "value_valid bitmap mismatch for input:\n{}",
        data_str
    );
    assert_eq!(
        actual.label, expected.label,
        "label span mismatch for input:\n{}",
        data_str
    );
    assert_eq!(
        actual.label_valid, expected.label_valid,
        "label_valid bitmap mismatch for input:\n{}",
        data_str
    );

    // For every valid label, check span contents
    for row in 0..expected.rows {
        if falx::kernels::csv_typed::bitmap_get(&expected.label_valid, row) {
            let exp_span = expected.label[row];
            let act_span = actual.label[row];
            assert_eq!(
                actual.span(act_span),
                &data[exp_span.0 as usize..exp_span.1 as usize],
                "label span[{}] contents mismatch for input:\n{}",
                row,
                data_str
            );
        }
    }
}

#[test]
fn hand_picked_cells() {
    // Build a CSV with edge cases for i64, f64, label, and title parsing
    let mut csv = String::new();

    // Row 0: title = plain word "hello"
    csv.push_str("9223372036854775807,hello,1.5,x,label0\n");

    // Row 1: title = empty cell (valid empty string)
    csv.push_str("-9223372036854775808,,-0.0,x,\"lab,el1\"\n");

    // Row 2: title = quoted cell with comma and doubled quote "a,""b"
    csv.push_str("9223372036854775808,\"a,\"\"b\"\",0.1,x,\"\"\n");

    // Row 3: title = quoted cell with embedded newline
    csv.push_str("+5,\"quo\nted\",0.0,x,lab3\n");

    // Row 4: title = quoted empty cell ""
    csv.push_str("-0,\"\",1_000,x,lab4\n");

    // Row 5: title = plain word
    csv.push_str("0007,world,1.5e308,x,\"123\"\n");

    // Row 6: title = empty cell
    csv.push_str(",,,5e-324,x,\"abc\"\n");

    // Row 7: only one field (no title field - invalid)
    csv.push_str("17\n");

    // Row 8: title = plain word with label that is NaN in value
    csv.push_str("18,test,NaN,x,label8\n");

    // Row 9: title = quoted with doubled quote in different positions
    csv.push_str("19,\"a\"\"b\",inf,x,label9\n");

    // Row 10: title = another plain word
    csv.push_str("20,simple,-inf,x,label10\n");

    let data = csv.as_bytes();
    let actual = falx::kernels::csv_typed::parse_columns(data);
    let expected = ref_parse_columns(data);

    assert_columns_match(data, &actual, &expected);
}

#[test]
fn randomized_differential() {
    let mut rng = Rng(0xDEAD_BEEF_CAFE_BABE);

    for _iter in 0..3000 {
        let num_records = rng.next() as usize % 40 + 1;
        let mut csv = Vec::new();

        for _record_idx in 0..num_records {
            let num_fields = rng.next() as usize % 8;
            let has_crlf = rng.next() % 10 < 1;

            for field_idx in 0..num_fields {
                if field_idx > 0 {
                    csv.push(b',');
                }

                let cell_type = rng.next() % 7;
                match cell_type {
                    0 => {
                        // Random i64-ish: optional sign, 1..22 digits
                        let sign = if rng.next() % 2 == 0 { b'-' } else { b'+' };
                        if rng.next() % 3 == 0 {
                            csv.push(sign);
                        }
                        let len = rng.next() as usize % 22 + 1;
                        for _ in 0..len {
                            csv.push(b'0' + (rng.next() % 10) as u8);
                        }
                    }
                    1 => {
                        // Random float: "<int>.<frac>" or "<int>e<exp>"
                        let sign = rng.next() % 2 == 0;
                        if sign {
                            csv.push(b'-');
                        }
                        let float_style = rng.next() % 5;
                        match float_style {
                            0 => {
                                // "int.frac"
                                let int_len = rng.next() as usize % 18 + 1;
                                for _ in 0..int_len {
                                    csv.push(b'0' + (rng.next() % 10) as u8);
                                }
                                csv.push(b'.');
                                let frac_len = rng.next() as usize % 18;
                                for _ in 0..frac_len {
                                    csv.push(b'0' + (rng.next() % 10) as u8);
                                }
                            }
                            1 => {
                                // "int"
                                let int_len = rng.next() as usize % 15 + 1;
                                for _ in 0..int_len {
                                    csv.push(b'0' + (rng.next() % 10) as u8);
                                }
                            }
                            2 => {
                                // "int e exp"
                                let int_len = rng.next() as usize % 8 + 1;
                                for _ in 0..int_len {
                                    csv.push(b'0' + (rng.next() % 10) as u8);
                                }
                                csv.push(b'e');
                                let exp_sign = rng.next() % 2 == 0;
                                if exp_sign {
                                    csv.push(b'-');
                                }
                                let exp = rng.next() as i32 % 700 - 350;
                                for ch in exp.abs().to_string().bytes() {
                                    csv.push(ch);
                                }
                            }
                            3 => {
                                // ".frac" (starts with .)
                                csv.push(b'.');
                                let frac_len = rng.next() as usize % 15 + 1;
                                for _ in 0..frac_len {
                                    csv.push(b'0' + (rng.next() % 10) as u8);
                                }
                            }
                            4 => {
                                // "int." (ends with .)
                                let int_len = rng.next() as usize % 15 + 1;
                                for _ in 0..int_len {
                                    csv.push(b'0' + (rng.next() % 10) as u8);
                                }
                                csv.push(b'.');
                            }
                            _ => unreachable!(),
                        }
                    }
                    2 => {
                        // Garbage letters
                        let len = rng.next() as usize % 10;
                        for _ in 0..len {
                            csv.push(b'a' + (rng.next() % 26) as u8);
                        }
                    }
                    3 => {
                        // Empty field
                    }
                    4 => {
                        // Literal "inf", "nan", "NaN"
                        let lit = rng.next() % 3;
                        match lit {
                            0 => csv.extend_from_slice(b"inf"),
                            1 => csv.extend_from_slice(b"nan"),
                            2 => csv.extend_from_slice(b"NaN"),
                            _ => unreachable!(),
                        }
                    }
                    5 | 6 => {
                        // Quoted cell with mixed content
                        csv.push(b'"');
                        let inner_len = rng.next() as usize % 50;
                        for _ in 0..inner_len {
                            let inner_type = rng.next() % 5;
                            match inner_type {
                                0 => csv.push(b','),
                                1 => {
                                    csv.push(b'"');
                                    csv.push(b'"'); // doubled quote
                                }
                                2 => csv.push(b'\n'),
                                3 => csv.push(b' '),
                                4 => csv.push(b'a' + (rng.next() % 26) as u8),
                                _ => unreachable!(),
                            }
                        }
                        csv.push(b'"');
                    }
                    _ => unreachable!(),
                }
            }

            if has_crlf {
                csv.push(b'\r');
                csv.push(b'\n');
            } else {
                csv.push(b'\n');
            }
        }

        // ~1/3 chance of unterminated final record
        if rng.next() % 3 == 0 {
            let num_fields = rng.next() as usize % 8;
            for field_idx in 0..num_fields {
                if field_idx > 0 {
                    csv.push(b',');
                }
                let len = rng.next() as usize % 20;
                for _ in 0..len {
                    csv.push(b'0' + (rng.next() % 10) as u8);
                }
            }
        }

        let actual = falx::kernels::csv_typed::parse_columns(&csv);
        let expected = ref_parse_columns(&csv);

        assert_columns_match(&csv, &actual, &expected);
    }
}

#[test]
fn block_boundary_stress() {
    // Test with records positioned at 64-byte boundaries
    for pad_len in 55..=70 {
        let mut csv = Vec::new();
        let record_template = format!("{}x,10,20.5,x,label\n", "a".repeat(pad_len));

        for _ in 0..5 {
            csv.extend_from_slice(record_template.as_bytes());
        }

        let actual = falx::kernels::csv_typed::parse_columns(&csv);
        let expected = ref_parse_columns(&csv);
        assert_columns_match(&csv, &actual, &expected);
    }

    // Large random input crossing many block boundaries
    let mut rng = Rng(0x5678_1234);
    let mut csv = Vec::new();

    while csv.len() < 100_000 {
        let num_fields = rng.next() as usize % 8;

        for field_idx in 0..num_fields {
            if field_idx > 0 {
                csv.push(b',');
            }

            let field_type = rng.next() % 3;
            match field_type {
                0 => {
                    let len = rng.next() as usize % 30;
                    for _ in 0..len {
                        csv.push(b'a' + (rng.next() % 26) as u8);
                    }
                }
                1 => {
                    let len = rng.next() as usize % 20;
                    for _ in 0..len {
                        csv.push(b'0' + (rng.next() % 10) as u8);
                    }
                }
                2 => {
                    csv.push(b'"');
                    let inner_len = rng.next() as usize % 40;
                    for _ in 0..inner_len {
                        let ch_type = rng.next() % 4;
                        match ch_type {
                            0 => csv.push(b','),
                            1 => {
                                csv.push(b'"');
                                csv.push(b'"');
                            }
                            2 => csv.push(b'\n'),
                            3 => csv.push(b'x'),
                            _ => unreachable!(),
                        }
                    }
                    csv.push(b'"');
                }
                _ => unreachable!(),
            }
        }

        csv.push(b'\n');
    }

    let actual = falx::kernels::csv_typed::parse_columns(&csv);
    let expected = ref_parse_columns(&csv);
    assert_columns_match(&csv, &actual, &expected);
}

#[test]
fn parallel_matches_serial() {
    let mut rng = Rng(0x9ABC_DEF0_1234_5678);

    for _iter in 0..10 {
        // Generate 50 KiB .. 300 KiB random input
        let target_size = 50_000 + rng.next() as usize % 250_000;
        let mut csv = Vec::new();

        while csv.len() < target_size {
            let num_fields = rng.next() as usize % 8;

            for field_idx in 0..num_fields {
                if field_idx > 0 {
                    csv.push(b',');
                }

                let field_type = rng.next() % 3;
                match field_type {
                    0 => {
                        let len = rng.next() as usize % 30;
                        for _ in 0..len {
                            csv.push(b'a' + (rng.next() % 26) as u8);
                        }
                    }
                    1 => {
                        let sign = rng.next() % 2 == 0;
                        if sign {
                            csv.push(b'-');
                        }
                        let len = rng.next() as usize % 20;
                        for _ in 0..len {
                            csv.push(b'0' + (rng.next() % 10) as u8);
                        }
                    }
                    2 => {
                        csv.push(b'"');
                        let inner_len = rng.next() as usize % 40;
                        for _ in 0..inner_len {
                            let ch_type = rng.next() % 4;
                            match ch_type {
                                0 => csv.push(b','),
                                1 => {
                                    csv.push(b'"');
                                    csv.push(b'"');
                                }
                                2 => csv.push(b'\n'),
                                3 => csv.push(b'x'),
                                _ => unreachable!(),
                            }
                        }
                        csv.push(b'"');
                    }
                    _ => unreachable!(),
                }
            }

            csv.push(b'\n');
        }

        let serial = falx::kernels::csv_typed::parse_columns(&csv);

        for threads in &[1, 2, 3, 7, 16] {
            let parallel = falx::kernels::csv_typed::parse_columns_par(&csv, *threads);

            assert_eq!(
                serial.rows, parallel.rows,
                "rows mismatch with {} threads",
                threads
            );
            assert_eq!(serial.id, parallel.id, "id mismatch with {} threads", threads);
            assert_eq!(
                serial.id_valid, parallel.id_valid,
                "id_valid mismatch with {} threads",
                threads
            );

            assert_eq!(
                serial.title_offsets, parallel.title_offsets,
                "title_offsets mismatch with {} threads",
                threads
            );
            assert_eq!(
                serial.title_data, parallel.title_data,
                "title_data mismatch with {} threads",
                threads
            );
            assert_eq!(
                serial.title_valid, parallel.title_valid,
                "title_valid mismatch with {} threads",
                threads
            );

            assert_eq!(
                serial.value.len(),
                parallel.value.len(),
                "value len mismatch with {} threads",
                threads
            );
            for (i, (s, p)) in serial.value.iter().zip(parallel.value.iter()).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    p.to_bits(),
                    "value[{}] mismatch with {} threads",
                    i,
                    threads
                );
            }

            assert_eq!(
                serial.value_valid, parallel.value_valid,
                "value_valid mismatch with {} threads",
                threads
            );
            assert_eq!(
                serial.label, parallel.label,
                "label mismatch with {} threads",
                threads
            );
            assert_eq!(
                serial.label_valid, parallel.label_valid,
                "label_valid mismatch with {} threads",
                threads
            );
        }
    }
}
