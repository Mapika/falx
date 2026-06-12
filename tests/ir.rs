use falx::ir::{CharClass, Graph};
use falx::interp;
use falx::scalar;
use falx::formats;

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

/// Compute parity of quotes to verify prefix_xor semantics.
fn scalar_parity(data: &[u8], quote: u8) -> Vec<u32> {
    let mut out = Vec::new();
    let mut parity = false;
    for (i, &byte) in data.iter().enumerate() {
        if byte == quote {
            parity = !parity;
        }
        if parity {
            out.push(i as u32);
        }
    }
    out
}

#[test]
fn prefix_xor_carry_across_blocks() {
    // Test at several quote positions to stress carry logic across 64-byte boundaries.
    let positions = [10usize, 63, 64, 127, 128];

    for &quote_pos in &positions {
        let mut data = vec![b'x'; 300];
        data[quote_pos] = b'"';

        // Build graph: Class(b'"') -> PrefixXor -> output
        let mut g = Graph::new();
        let quotes = g.class_byte(b'"');
        let parity = g.prefix_xor(quotes);
        g.set_output(parity);

        // Get expected output from scalar parity computation.
        let expected = scalar_parity(&data, b'"');

        // Run the graph.
        let mut output = Vec::new();
        interp::run(&g, &data, &mut output);

        assert_eq!(
            output, expected,
            "prefix_xor mismatch at quote position {}: graph output {:?}, expected {:?}",
            quote_pos, output, expected
        );
    }
}

#[test]
fn shift_left1_carry_across_blocks() {
    let mut data = vec![b'x'; 200];

    // Place backslashes at strategic positions across block boundaries.
    let backslash_positions = [62usize, 63, 64, 127];
    for &pos in &backslash_positions {
        data[pos] = b'\\';
    }

    // Build graph: Class(b'\\') -> ShiftLeft1 -> output
    let mut g = Graph::new();
    let backslashes = g.class_byte(b'\\');
    let shifted = g.shift_left1(backslashes);
    g.set_output(shifted);

    // ShiftLeft1 shifts bits left by 1, so backslash at position i appears as
    // a set bit at position i+1 in the output.
    let mut expected = Vec::new();
    for &pos in &backslash_positions {
        // Shifted position is pos + 1
        expected.push((pos + 1) as u32);
    }
    expected.sort();

    let mut output = Vec::new();
    interp::run(&g, &data, &mut output);

    assert_eq!(
        output, expected,
        "shift_left1 mismatch: graph output {:?}, expected {:?}",
        output, expected
    );

    // Also verify no index >= input length (pad bits are masked in interp::run).
    for &idx in &output {
        assert!(
            (idx as usize) < data.len(),
            "output index {} >= input length {}, pad bits leaked",
            idx,
            data.len()
        );
    }
}

#[test]
fn shift_left1_at_end_of_input() {
    // A mark on the final byte shifts to a position one past the input; that
    // bit lands in the zero padding and must not be reported.
    let mut g = Graph::new();
    let backslashes = g.class_byte(b'\\');
    let shifted = g.shift_left1(backslashes);
    g.set_output(shifted);

    for len in [1usize, 63, 64, 65, 200] {
        let mut data = vec![b'x'; len];
        data[len - 1] = b'\\';
        let mut output = Vec::new();
        interp::run(&g, &data, &mut output);
        assert!(
            output.is_empty(),
            "len {}: shifted bit past end of input leaked: {:?}",
            len,
            output
        );
    }
}

#[test]
fn alternative_dialects_differential() {
    let dialects = [
        (b';', b'\''),
        (b'\t', b'"'),
        (b'|', b'q'),
    ];

    let mut rng = Rng(0x1234_5678_9ABC_DEF0);

    for (delimiter, quote) in &dialects {
        // Build the graph for this dialect.
        let g = formats::delimited(&formats::Dialect {
            structural: vec![*delimiter, b'\n'],
            record_terminator: b'\n',
            quote: Some(*quote),
            escape: formats::Escape::None,
            comment: None,
            nesting: vec![],
        });

        // Run 500 randomized tests.
        for _ in 0..500 {
            let len = (rng.next() % 300) as usize;

            // Generate input weighted toward the dialect's special bytes.
            let alphabet = [*delimiter, *quote, b'\n', b'\r', b'x', b'y'];
            let data: Vec<u8> = (0..len)
                .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                .collect();

            // Get expected output from scalar dialect version.
            let mut expected = Vec::new();
            scalar::index_structurals_dialect(&data, *delimiter, *quote, &mut expected);

            // Get output from IR interpreter.
            let mut output = Vec::new();
            interp::run(&g, &data, &mut output);

            assert_eq!(
                output, expected,
                "dialect (delimiter={:?}, quote={:?}) mismatch on input: {:?}",
                *delimiter as char, *quote as char, data
            );
        }
    }
}

/// The backslash-escape machinery (ShiftLeft1 + Add odd-run detection) must
/// agree with the escape-aware scalar reference, including escape runs that
/// span 64-byte block boundaries.
#[test]
fn backslash_escape_differential() {
    let mut rng = Rng(0xFEED_FACE_CAFE_BEEF);
    for dialect in [formats::logfmt_dialect(), formats::ndjson_dialect()] {
        let g = formats::delimited(&dialect);
        let escape_byte = match dialect.escape {
            formats::Escape::Backslash(b) => b,
            formats::Escape::None => unreachable!("presets under test use backslash"),
        };

        // Alphabet heavy on backslashes so long runs (incl. across block
        // seams) occur often.
        let mut alphabet = vec![escape_byte, escape_byte, escape_byte];
        alphabet.extend_from_slice(&dialect.structural);
        alphabet.push(dialect.quote.unwrap());
        alphabet.extend_from_slice(b"xy");

        for _ in 0..2000 {
            let len = (rng.next() % 300) as usize;
            let data: Vec<u8> = (0..len)
                .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
                .collect();

            let mut expected = Vec::new();
            scalar::index_structurals_spec(
                &data,
                &dialect.structural,
                dialect.quote,
                Some(escape_byte),
                None,
                &mut expected,
            );
            let mut output = Vec::new();
            interp::run(&g, &data, &mut output);
            assert_eq!(
                output, expected,
                "escape dialect {:?} mismatch on input: {:?}",
                dialect,
                String::from_utf8_lossy(&data)
            );
        }
    }
}

#[test]
fn charclass_membership() {
    // Test from_bytes and from_byte.
    let bytes = [0u8, 63, 64, 127, 128, 255];
    let class = CharClass::from_bytes(&bytes);

    // Verify exactly those 6 bytes are in the class.
    let mut found = Vec::new();
    for byte in 0..=255u8 {
        if class.contains(byte) {
            found.push(byte);
        }
    }

    assert_eq!(
        found, bytes,
        "CharClass::from_bytes contains unexpected bytes: {:?}",
        found
    );

    // Test from_byte.
    let class_single = CharClass::from_byte(42);
    assert!(class_single.contains(42));
    for byte in 0..=255u8 {
        if byte != 42 {
            assert!(!class_single.contains(byte), "from_byte(42) unexpectedly contains {}", byte);
        }
    }

    // Test empty().
    let empty = CharClass::empty();
    for byte in 0..=255u8 {
        assert!(!empty.contains(byte), "empty() unexpectedly contains {}", byte);
    }
}

#[test]
fn multi_byte_class_in_graph() {
    // Build a graph with a single Class node for [b',', b';', b'\n'].
    let mut g = Graph::new();
    let class = g.class(CharClass::from_bytes(b",;\n"));
    g.set_output(class);

    // Input: b"a,b;c\nd"
    let data = b"a,b;c\nd".as_slice();
    let mut output = Vec::new();
    interp::run(&g, data, &mut output);

    // Expected structural positions: 1 (,), 3 (;), 5 (\n)
    let expected = vec![1u32, 3, 5];

    assert_eq!(
        output, expected,
        "multi_byte_class output {:?}, expected {:?}",
        output, expected
    );
}
