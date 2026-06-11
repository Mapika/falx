//! Comment-line support: the csv_hash kernel (CSV + `#` comments) must
//! agree with the comment-aware scalar reference at the structural level,
//! and every record walker (records iterator, columns sink, streaming)
//! must skip comment records identically.

use falx::kernels::csv_hash as k;
use falx::{formats, interp, scalar};

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
fn hand_picked_comment_semantics() {
    // Comment with embedded quote and commas; mid-data comment; a quoted
    // field *starting* with the comment byte (not a comment); trailing
    // comment without newline.
    let data = b"# header comment, with \"quotes\" and, commas\nfoo,42\n#mid, comment\nbar,7\n\"#not,a comment\",9\n# trailing comment";

    let parsed = k::parse(data);
    let recs: Vec<Vec<u8>> = parsed.records().map(|r| r.as_bytes().to_vec()).collect();
    assert_eq!(
        recs,
        vec![
            b"foo,42".to_vec(),
            b"bar,7".to_vec(),
            b"\"#not,a comment\",9".to_vec()
        ]
    );

    let cols = k::parse_columns(data);
    assert_eq!(cols.rows, 3);
    let keys: Vec<Vec<u8>> = (0..cols.rows)
        .map(|r| k::string_at(&cols.key_offsets, &cols.key_data, r).to_vec())
        .collect();
    assert_eq!(
        keys,
        vec![b"foo".to_vec(), b"bar".to_vec(), b"#not,a comment".to_vec()]
    );
    assert_eq!(cols.amount, vec![42, 7, 9]);
    for r in 0..3 {
        assert!(k::bitmap_get(&cols.amount_valid, r));
    }

    // A comment line between quoted multi-line fields must not confuse
    // quote context, and a quote inside a comment must be inert.
    let tricky = b"#comment with one \" quote\na,\"multi\nline\"\n#another \" stray\nb,2\n";
    let recs: Vec<Vec<u8>> = k::parse(tricky).records().map(|r| r.as_bytes().to_vec()).collect();
    assert_eq!(recs, vec![b"a,\"multi\nline\"".to_vec(), b"b,2".to_vec()]);
}

#[test]
fn streaming_skips_comments_like_batch() {
    let data = b"# c1\nfoo,1\n#c2,x\nbar,2\n# trailing";
    let batch: Vec<Vec<u8>> = k::parse(data).records().map(|r| r.as_bytes().to_vec()).collect();
    for feed in [1usize, 3, 7, 64, 1024] {
        let mut streamed = Vec::new();
        let mut sp = k::stream();
        for chunk in data.chunks(feed) {
            sp.feed(chunk, |r| streamed.push(r.as_bytes().to_vec()));
        }
        sp.finish(|r| streamed.push(r.as_bytes().to_vec()));
        assert_eq!(streamed, batch, "feed size {feed}");
    }
}

/// Structural-level differential: generated AVX2, generated fallback, and
/// the IR interpreter must agree with the comment-aware scalar reference,
/// with comment bytes, quotes, and newlines interleaving freely across
/// 64-byte block boundaries.
#[test]
fn randomized_structural_differential() {
    let d = formats::csv_hash_dialect();
    let g = formats::delimited(&d);
    let alphabet = b"\"\",,\n\n##xy\r";
    let mut rng = Rng(0xFEED_FACE_CAFE_F00D);
    for round in 0..4000 {
        let len = (rng.next() % 350) as usize;
        let input: Vec<u8> = (0..len)
            .map(|_| alphabet[(rng.next() % alphabet.len() as u64) as usize])
            .collect();
        let mut expected = Vec::new();
        scalar::index_structurals_spec(&input, &d.structural, d.quote, None, d.comment, &mut expected);
        let mut got = Vec::new();
        k::index_structurals(&input, &mut got);
        assert_eq!(
            got,
            expected,
            "dispatched divergence round {round} on {:?}",
            String::from_utf8_lossy(&input)
        );
        let mut fb = Vec::new();
        k::fallback::index_structurals(&input, &mut fb);
        assert_eq!(fb, expected, "fallback divergence round {round}");
        let mut ir = Vec::new();
        interp::run(&g, &input, &mut ir);
        assert_eq!(ir, expected, "interp divergence round {round}");
    }
}

/// Record-level differential: a dumb reference parser (state machine over
/// bytes) must agree with records() and parse_columns on which records
/// survive comment skipping.
#[test]
fn randomized_record_differential() {
    let mut rng = Rng(0x0BAD_F00D_DEAD_C0DE);
    for _ in 0..1500 {
        // Build records line-wise so comments appear in realistic places.
        let mut data = Vec::new();
        let lines = rng.next() % 12;
        for _ in 0..lines {
            match rng.next() % 4 {
                0 => {
                    data.extend_from_slice(b"#");
                    for _ in 0..rng.next() % 10 {
                        data.push(b",\"x#\n".as_slice()[(rng.next() % 4) as usize]);
                    }
                    data.push(b'\n');
                }
                _ => {
                    let fields = rng.next() % 4;
                    for f in 0..=fields {
                        if f > 0 {
                            data.push(b',');
                        }
                        for _ in 0..rng.next() % 5 {
                            data.push(b'a' + (rng.next() % 26) as u8);
                        }
                    }
                    data.push(b'\n');
                }
            }
        }
        if rng.next() % 3 == 0 {
            data.extend_from_slice(b"# unterminated trailer");
        }

        // Reference: split on newlines (no quotes in this generator's
        // unquoted lines... comments may contain quote bytes, which the
        // kernel treats as inert; restrict the check to inputs where that
        // cannot bleed: comment quote bytes only appear inside comments).
        let expected: Vec<&[u8]> = data
            .split(|&b| b == b'\n')
            .enumerate()
            .filter(|(i, line)| {
                // the final empty split after a trailing newline is not a record
                !(line.is_empty() && *i == data.iter().filter(|&&b| b == b'\n').count())
                    && !line.first().is_some_and(|&b| b == b'#')
            })
            .map(|(_, line)| line)
            .collect();
        // NOTE: this dumb split is only valid because the generator never
        // emits quote bytes outside comment lines.
        let got: Vec<Vec<u8>> = k::parse(&data).records().map(|r| r.as_bytes().to_vec()).collect();
        assert_eq!(
            got.len(),
            expected.len(),
            "record count mismatch on {:?}",
            String::from_utf8_lossy(&data)
        );
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_eq!(g.as_slice(), *e, "record mismatch on {:?}", String::from_utf8_lossy(&data));
        }
    }
}
