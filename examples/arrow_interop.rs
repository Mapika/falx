//! falx columnar output → Arrow arrays, zero-copy.
//!
//! Generated kernels deliberately emit Arrow's own layouts:
//!   - numeric columns: values `Vec<T>` + LSB-first validity bitmap
//!     (= Arrow primitive array),
//!   - string columns: `offsets: Vec<i32>` + contiguous `data: Vec<u8>` +
//!     validity bitmap (= Arrow varbinary array).
//!
//!     Converting a column is therefore a buffer *wrap*, not a conversion —
//!     every Vec moves into Arrow untouched.
//!
//! The generated kernels themselves stay std-only; arrow is a
//! dev-dependency of this example alone.

use arrow_array::{Array, BinaryArray, Float64Array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};

/// Wrap a falx validity bitmap as an Arrow NullBuffer. falx bitmaps are
/// LSB-first like Arrow's, and on little-endian targets (everywhere
/// arrow-rs runs in practice) the u64 words already lie in memory in
/// Arrow's byte order — so this reuses the allocation.
fn null_buffer(bitmap: Vec<u64>, rows: usize) -> NullBuffer {
    NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bitmap), 0, rows))
}

fn main() {
    // csv_typed projects: id (i64 @0), title (string @1), value (f64 @2),
    // label (bytes @4).
    let csv_data = b"id,title,value,notes,label
1,hello,2.5,skip,x
2,\"quoted, with \"\"q\"\"\",3.25,skip,y
3,,4.5,skip,z
4,unparsed-value,not-a-number,skip,w
5
";

    let cols = falx::kernels::csv_typed::parse_columns(csv_data);
    let rows = cols.rows;
    println!("Parsed {rows} rows");

    // Capture expectations before the buffers move into Arrow.
    let expected_titles: Vec<(bool, Vec<u8>)> = (0..rows)
        .map(|r| {
            (
                falx::kernels::csv_typed::bitmap_get(&cols.title_valid, r),
                falx::kernels::csv_typed::string_at(&cols.title_offsets, &cols.title_data, r)
                    .to_vec(),
            )
        })
        .collect();
    let expected_values: Vec<(bool, f64)> = (0..rows)
        .map(|r| {
            (
                falx::kernels::csv_typed::bitmap_get(&cols.value_valid, r),
                cols.value[r],
            )
        })
        .collect();

    // f64 column -> Float64Array: both buffers move, nothing is copied.
    let value = Float64Array::new(
        ScalarBuffer::from(cols.value),
        Some(null_buffer(cols.value_valid, rows)),
    );

    // string column -> BinaryArray: offsets, data, and validity all move.
    // (BinaryArray skips UTF-8 validation; for data known to be UTF-8,
    // `StringArray::try_new` over the same buffers validates once.)
    let title = BinaryArray::new(
        OffsetBuffer::new(ScalarBuffer::from(cols.title_offsets)),
        Buffer::from_vec(cols.title_data),
        Some(null_buffer(cols.title_valid, rows)),
    );

    for r in 0..rows {
        let (tv, ref tbytes) = expected_titles[r];
        assert_eq!(title.is_valid(r), tv, "title validity mismatch at row {r}");
        if tv {
            assert_eq!(title.value(r), tbytes.as_slice(), "title bytes mismatch at row {r}");
        }
        let (vv, vval) = expected_values[r];
        assert_eq!(value.is_valid(r), vv, "value validity mismatch at row {r}");
        if vv {
            assert_eq!(value.value(r), vval, "value mismatch at row {r}");
        }
        println!(
            "  row {r}: title={:>22} value={}",
            if title.is_valid(r) {
                format!("{:?}", String::from_utf8_lossy(title.value(r)))
            } else {
                "null".to_string()
            },
            if value.is_valid(r) { value.value(r).to_string() } else { "null".to_string() },
        );
    }

    // Row 0 is the header (title "title" is a fine string; value "value"
    // is not a number), row 3 has an empty-but-valid title, row 5 has one
    // field so both title and value are null there.
    assert_eq!(value.null_count(), 3, "header + not-a-number + one-field rows");
    assert_eq!(title.null_count(), 1, "the one-field row");
    println!("\nOK: falx columns became Float64Array + BinaryArray without copying any buffer.");
}
