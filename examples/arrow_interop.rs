//! falx columnar output → Arrow primitive arrays, zero-copy.
//!
//! Generated kernels deliberately emit the Arrow primitive-array layout:
//! a values `Vec<T>` plus an LSB-first validity bitmap. Converting a
//! column to Arrow is therefore a buffer *wrap*, not a conversion — both
//! the values Vec and the bitmap allocation move into Arrow untouched.
//!
//! The generated kernels themselves stay std-only; arrow is a
//! dev-dependency of this example alone.

use arrow_array::{Array, Float64Array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};

fn main() {
    let csv_data = b"Country,City,AccentCity,Region,Population,Latitude,Longitude
US,NewYork,New York,NY,8000000,40.712776,-74.005974
GB,London,London,EN,9000000,51.507351,-0.127758
JP,Tokyo,Tokyo,TY,,35.652832,139.839478
FR,Paris,Paris,IL,2000000,not-a-number,2.352222
DE,Berlin,Berlin,BE,3500000,52.517037,13.388860
";

    let cols = falx::kernels::csv_geo::parse_columns(csv_data);
    println!("Parsed {} rows", cols.rows);

    // Capture the expected per-row state before the buffers move into
    // Arrow (the conversion consumes them — that is the point).
    let expected: Vec<(bool, f64)> = (0..cols.rows)
        .map(|row| {
            (
                falx::kernels::csv_geo::bitmap_get(&cols.latitude_valid, row),
                cols.latitude[row],
            )
        })
        .collect();

    // Zero-copy on both buffers:
    //  - ScalarBuffer takes ownership of the values Vec<f64> as-is.
    //  - Buffer::from_vec reuses the bitmap's Vec<u64> allocation; falx
    //    bitmaps are LSB-first like Arrow's, and on little-endian targets
    //    (everywhere arrow-rs runs in practice) the u64 words already lie
    //    in memory in Arrow's byte order.
    let rows = cols.rows;
    let latitude = Float64Array::new(
        ScalarBuffer::from(cols.latitude),
        Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from_vec(cols.latitude_valid),
            0,
            rows,
        ))),
    );

    for (row, &(valid, value)) in expected.iter().enumerate() {
        assert_eq!(latitude.is_valid(row), valid, "validity mismatch at row {row}");
        if valid {
            assert_eq!(latitude.value(row), value, "value mismatch at row {row}");
        }
        println!(
            "  row {row}: {}",
            if valid {
                format!("latitude {}", latitude.value(row))
            } else {
                "null (header or malformed cell)".to_string()
            }
        );
    }

    assert_eq!(latitude.null_count(), 2, "header row + the not-a-number row");
    println!("\nOK: falx column buffers became an Arrow Float64Array without copying.");
}
