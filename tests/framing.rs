//! Length-prefixed framing: the IR's outer container layer.
//!
//! The load-bearing test here is bgzf parity. `falx::bgzf` has a hand-written,
//! independently-tested block scanner that predates the framing model; if the
//! generalized model describes real block-compressed framing correctly, it must
//! find exactly the same block boundaries. That is a much stronger check than
//! any self-consistency property the model could assert about itself.

use falx::framing::{self, Counts, Endian, Framing, Width};

/// Canonical bgzf: header 18, BSIZE (u16 le) at 16 counting total-minus-one,
/// 8-byte gzip trailer, `1f 8b` magic.
fn bgzf() -> Framing {
    framing::bgzf_framing()
}

/// Build a bgzf stream the same way `falx::bgzf`'s own tests do: stored-mode
/// DEFLATE blocks so no compressor is needed, with a valid BC subfield.
fn build_bgzf(chunks: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in chunks {
        // Stored (uncompressed) DEFLATE: final block, type 00, then LEN/NLEN.
        let mut deflated = Vec::new();
        deflated.push(0x01);
        deflated.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
        deflated.extend_from_slice(&(!(chunk.len() as u16)).to_le_bytes());
        deflated.extend_from_slice(chunk);

        let block_len = 12 + 6 + deflated.len() + 8;
        let mut hdr = [0u8; 18];
        hdr[0] = 0x1f;
        hdr[1] = 0x8b;
        hdr[2] = 8; // CM = deflate
        hdr[3] = 0x04; // FLG = FEXTRA
        hdr[9] = 0xff; // OS unknown
        hdr[10] = 6; // XLEN
        hdr[12] = b'B';
        hdr[13] = b'C';
        hdr[14] = 2; // SLEN
        let bsize = (block_len - 1) as u16;
        hdr[16] = bsize as u8;
        hdr[17] = (bsize >> 8) as u8;
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&deflated);
        out.extend_from_slice(&0u32.to_le_bytes()); // CRC32 (unchecked here)
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes()); // ISIZE
    }
    out
}

/// The generalized scanner must agree with the hand-written bgzf scanner on
/// every block boundary and payload range.
#[cfg(feature = "bgzf")]
#[test]
fn matches_the_handwritten_bgzf_scanner() {
    let payloads: Vec<Vec<u8>> = (0..64)
        .map(|i| format!("block {i}: {}\n", "x".repeat(i * 7)).into_bytes())
        .collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    let data = build_bgzf(&refs);

    let blocks = falx::bgzf::scan(&data).expect("hand-written scanner should accept this stream");
    let frames = framing::scan(&bgzf(), &data).expect("framing scanner should accept it too");

    assert_eq!(
        frames.len(),
        blocks.len(),
        "frame count differs from the hand-written scanner's block count"
    );
    for (i, (frame, block)) in frames.iter().zip(&blocks).enumerate() {
        assert_eq!(
            frame.payload, block.payload,
            "payload range differs from the hand-written scanner at block {i}"
        );
    }
    // The frames must tile the input exactly — no gap, no overlap.
    let covered: usize = frames.iter().map(|f| f.len).sum();
    assert_eq!(covered, data.len(), "frames do not tile the input");
}

/// An empty final member (the bgzf EOF marker) is still a well-formed frame,
/// and `skip_empty` is about empty *payloads*, which the EOF marker does not
/// have — its payload is a 2-byte empty DEFLATE stream. Pinning this down
/// keeps the model honest about what it does and does not know.
#[test]
fn eof_marker_is_a_valid_frame() {
    let data = build_bgzf(&[b"hello", b""]);
    let frames = framing::scan(&bgzf(), &data).expect("scan");
    assert_eq!(frames.len(), 2);
    assert!(
        !frames[1].payload.is_empty(),
        "the trailing member still carries a DEFLATE payload"
    );
}

/// Varint framing: protobuf-style length-delimited streams, where the length
/// field's own size shifts the payload start.
#[test]
fn varint_framing_walks_a_length_delimited_stream() {
    let f = Framing {
        header: 0,
        length_at: 0,
        width: Width::Varint,
        endian: Endian::Le,
        counts: Counts::Payload,
        adjust: 0,
        trailer: 0,
        magic: None,
        skip_empty: false,
        uncompressed: None,
    };
    // Payload lengths spanning the 1-byte/2-byte varint boundary (128).
    let lengths = [0usize, 1, 5, 127, 128, 300];
    let mut data = Vec::new();
    for &len in &lengths {
        let mut n = len as u64;
        loop {
            let byte = (n & 0x7f) as u8;
            n >>= 7;
            if n == 0 {
                data.push(byte);
                break;
            }
            data.push(byte | 0x80);
        }
        data.extend(std::iter::repeat_n(b'z', len));
    }

    let frames = framing::scan(&f, &data).expect("varint scan");
    assert_eq!(frames.len(), lengths.len());
    for (frame, &len) in frames.iter().zip(&lengths) {
        assert_eq!(frame.payload.len(), len, "decoded payload length");
        assert!(data[frame.payload.clone()].iter().all(|&b| b == b'z'));
    }
    assert_eq!(
        frames.iter().map(|f| f.len).sum::<usize>(),
        data.len(),
        "frames do not tile the input"
    );
}

/// Big-endian, payload-counting, trailer-carrying framing — the axes that are
/// independent of bgzf's particular choices.
#[test]
fn big_endian_payload_counted_framing() {
    let f = Framing {
        header: 4,
        length_at: 0,
        width: Width::U32,
        endian: Endian::Be,
        counts: Counts::Payload,
        adjust: 0,
        trailer: 2,
        magic: None,
        skip_empty: false,
        uncompressed: None,
    };
    let mut data = Vec::new();
    for payload in [b"abc".as_slice(), b"".as_slice(), b"defgh".as_slice()] {
        data.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        data.extend_from_slice(payload);
        data.extend_from_slice(b"\xff\xff"); // trailer
    }
    let frames = framing::scan(&f, &data).expect("scan");
    assert_eq!(frames.len(), 3);
    assert_eq!(&data[frames[0].payload.clone()], b"abc");
    assert!(frames[1].payload.is_empty());
    assert_eq!(&data[frames[2].payload.clone()], b"defgh");
}

/// `skip_empty` drops empty-payload frames.
#[test]
fn skip_empty_drops_empty_payloads() {
    let base = Framing {
        header: 4,
        length_at: 0,
        width: Width::U32,
        endian: Endian::Be,
        counts: Counts::Payload,
        adjust: 0,
        trailer: 0,
        magic: None,
        skip_empty: false,
        uncompressed: None,
    };
    let mut data = Vec::new();
    for payload in [b"ab".as_slice(), b"".as_slice(), b"cd".as_slice()] {
        data.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        data.extend_from_slice(payload);
    }
    assert_eq!(framing::scan(&base, &data).expect("scan").len(), 3);
    let skipping = Framing {
        skip_empty: true,
        ..base
    };
    assert_eq!(framing::scan(&skipping, &data).expect("scan").len(), 2);
}

/// Malformed streams are rejected rather than looping, over-reading, or
/// silently truncating. A frame that does not advance the scan is the
/// dangerous case — it would hang.
#[test]
fn malformed_framing_is_rejected() {
    let f = bgzf();

    // Wrong magic.
    let mut bad = build_bgzf(&[b"hello"]);
    bad[0] = 0x00;
    assert!(
        framing::scan(&f, &bad).is_err(),
        "bad magic should be rejected"
    );

    // Declared length runs past the end of the input.
    let mut truncated = build_bgzf(&[b"hello world"]);
    truncated.truncate(truncated.len() - 4);
    assert!(
        framing::scan(&f, &truncated).is_err(),
        "truncated stream should be rejected"
    );

    // A length too small to contain the header and trailer would not advance
    // the scan; it must be an error, not an infinite loop.
    let mut degenerate = build_bgzf(&[b"hello"]);
    degenerate[16] = 0;
    degenerate[17] = 0;
    assert!(
        framing::scan(&f, &degenerate).is_err(),
        "a non-advancing frame length should be rejected"
    );

    // Header cut off before the length field.
    assert!(
        framing::scan(&f, &[0x1f, 0x8b, 8, 4]).is_err(),
        "a truncated header should be rejected"
    );

    // Varint that never terminates.
    let runaway = Framing {
        header: 0,
        length_at: 0,
        width: Width::Varint,
        endian: Endian::Le,
        counts: Counts::Payload,
        adjust: 0,
        trailer: 0,
        magic: None,
        skip_empty: false,
        uncompressed: None,
    };
    assert!(
        framing::scan(&runaway, &[0x80, 0x80, 0x80]).is_err(),
        "an unterminated varint should be rejected"
    );
}

/// Framing survives the IR round trip, and a framed module generates the
/// scanner and parallel driver.
#[test]
fn framing_round_trips_through_textual_ir_and_generates_code() {
    let text = "\
falx-ir 1
format bgzf_container
structural 0a
frame header=18 length-at=16 width=u16 endian=le counts=total adjust=1 trailer=8 magic=0:1f,8b uncompressed=-4:u32:le
%0 = class 0a
output %0
terminators %0
";
    let module = falx::ir_text::parse(text).expect("framed IR should parse");
    let declared = module.framing.clone().expect("framing should be present");
    assert_eq!(
        declared,
        bgzf(),
        "parsed framing differs from the reference"
    );

    let printed = falx::ir_text::print(&module);
    let reparsed = falx::ir_text::parse(&printed).expect("reprint should parse");
    assert_eq!(reparsed.framing, module.framing);
    assert_eq!(
        printed,
        falx::ir_text::print(&reparsed),
        "printing is not idempotent"
    );

    let code = falx::codegen::emit_module(&module).expect("emit should succeed");
    for item in [
        "pub fn scan_frames",
        "pub fn frame_at",
        "pub fn frames_par",
        "pub struct Frame",
    ] {
        assert!(code.contains(item), "generated code is missing `{item}`");
    }
    // The unframed path must not grow a frame API.
    let unframed = falx::ir_text::parse(
        "falx-ir 1\nformat plain\nstructural 0a\n%0 = class 0a\noutput %0\nterminators %0\n",
    )
    .expect("parse");
    let plain = falx::codegen::emit_module(&unframed).expect("emit");
    assert!(
        !plain.contains("scan_frames"),
        "a module without framing should not emit a frame scanner"
    );
}

/// A length field that does not fit inside the declared header is a spec
/// error, caught at generation time rather than producing a scanner that
/// reads past its own header.
#[test]
fn length_field_outside_the_header_is_rejected() {
    let text = "\
falx-ir 1
format bad_frame
structural 0a
frame header=4 length-at=16 width=u16 endian=le counts=total adjust=0 trailer=0
%0 = class 0a
output %0
terminators %0
";
    let module = falx::ir_text::parse(text).expect("parse");
    assert!(
        falx::codegen::emit_module(&module).is_err(),
        "a length field outside the header should be rejected at emit time"
    );
}

/// Decompression driven by the framing descriptor must agree byte-for-byte
/// with the hand-written bgzf path. This is the end-to-end check that the
/// generalized container model can actually drive a real decompressor.
#[cfg(feature = "bgzf")]
#[test]
fn framing_driven_decompression_matches_handwritten_bgzf() {
    let payloads: Vec<Vec<u8>> = (0..97)
        .map(|i| format!("row {i},{},{}\n", i * 3, "payload".repeat(i % 11)).into_bytes())
        .collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    let data = build_bgzf(&refs);
    let expected: Vec<u8> = payloads.concat();

    for threads in [1usize, 2, 4, 16] {
        let handwritten = falx::bgzf::decompress_par(&data, threads).expect("hand-written path");
        let framed = falx::bgzf::decompress_framed_par(&data, &bgzf(), threads)
            .expect("framing-driven path");
        assert_eq!(
            handwritten, expected,
            "hand-written decompression lost data at {threads} threads"
        );
        assert_eq!(
            framed, handwritten,
            "framing-driven decompression differs from the hand-written path at {threads} threads"
        );
    }
}

/// The full compressed-container pipeline: framing locates blocks, bgzf
/// inflates them in parallel, and a generated payload parser runs over each
/// decompressed block. This is the path the framing layer exists to enable.
///
/// The result is checked against decompress-then-parse of the whole stream —
/// two independent routes to the same answer. `latitude_checksum` is a
/// wrapping sum of f64 bit patterns, so it is additive across blocks and the
/// comparison is exact rather than approximate.
#[cfg(feature = "bgzf")]
#[test]
fn compressed_container_feeds_a_generated_payload_parser() {
    // CSV payload split on record boundaries, so no record spans a block
    // (block-spanning records are the caller's concern, as documented).
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut expected_rows = 0usize;
    for block in 0..40 {
        let mut chunk = Vec::new();
        for row in 0..25 {
            let lat = (block * 25 + row) as f64 / 8.0;
            chunk.extend_from_slice(
                format!("cc,city{row},accent,region,999,{lat:.6},-1.500000\n").as_bytes(),
            );
            expected_rows += 1;
        }
        chunks.push(chunk);
    }
    let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
    let data = build_bgzf(&refs);

    // Route A: inflate + parse each block in parallel.
    let states = falx::bgzf::parse_framed_par(
        &data,
        &bgzf(),
        8,
        || (0u64, 0u64),
        |state, _index, block| {
            let stats = falx::kernels::csv_geo::parse_csv_geo_stats(block);
            state.0 += stats.records;
            state.1 = state.1.wrapping_add(stats.latitude_checksum);
        },
    )
    .expect("framed parse");
    let rows: u64 = states.iter().map(|s| s.0).sum();
    let checksum: u64 = states.iter().fold(0u64, |acc, s| acc.wrapping_add(s.1));

    // Route B: decompress the whole stream, then parse it once.
    let whole = falx::bgzf::decompress_framed_par(&data, &bgzf(), 8).expect("decompress");
    let direct = falx::kernels::csv_geo::parse_csv_geo_stats(&whole);

    assert_eq!(
        rows as usize, expected_rows,
        "row count through the compressed pipeline"
    );
    assert_eq!(
        direct.records as usize, expected_rows,
        "row count via decompress-then-parse"
    );
    assert_eq!(
        checksum, direct.latitude_checksum,
        "the streamed and decompress-then-parse routes disagree"
    );
}

/// Framing without an `uncompressed` declaration cannot drive parallel
/// decompression — there is no way to presize the output — and must say so
/// rather than guessing.
#[cfg(feature = "bgzf")]
#[test]
fn decompression_requires_a_declared_uncompressed_size() {
    let data = build_bgzf(&[b"hello"]);
    let without = Framing {
        uncompressed: None,
        ..bgzf()
    };
    assert!(
        falx::bgzf::decompress_framed_par(&data, &without, 4).is_err(),
        "decompression without a declared uncompressed size should be refused"
    );
}

/// The uncompressed-size field decodes correctly and survives the IR round trip.
#[test]
fn uncompressed_size_field_decodes_and_round_trips() {
    let payloads: [&[u8]; 3] = [b"a", b"bbbb", b"cc"];
    let data = build_bgzf(&payloads);
    let frames = framing::scan(&bgzf(), &data).expect("scan");
    let sizes: Vec<usize> = frames
        .iter()
        .map(|f| f.uncompressed.expect("isize"))
        .collect();
    assert_eq!(sizes, vec![1, 4, 2], "decoded ISIZE per block");

    let text = "\
falx-ir 1
format framed
structural 0a
frame header=18 length-at=16 width=u16 endian=le counts=total adjust=1 trailer=8 magic=0:1f,8b uncompressed=-4:u32:le
%0 = class 0a
output %0
terminators %0
";
    let module = falx::ir_text::parse(text).expect("parse");
    assert_eq!(
        module.framing.as_ref().unwrap().uncompressed,
        bgzf().uncompressed
    );
    let printed = falx::ir_text::print(&module);
    assert!(printed.contains("uncompressed=-4:u32:le"));
    let reparsed = falx::ir_text::parse(&printed).expect("reparse");
    assert_eq!(reparsed.framing, module.framing);
}

/// Records that straddle block boundaries must still be parsed exactly once
/// and in full.
///
/// The payload here is split at arbitrary byte positions — deliberately *not*
/// on record boundaries — so most blocks begin and end mid-record. The result
/// must match parsing the whole decompressed stream, which is the only
/// reference that cannot itself be wrong about boundaries.
#[cfg(feature = "bgzf")]
#[test]
fn records_spanning_block_boundaries_are_parsed_exactly_once() {
    let mut csv = Vec::new();
    for row in 0..20_000u64 {
        csv.extend_from_slice(
            format!(
                "cc,city{row},accent,region,999,{:.6},-1.500000\n",
                row as f64 / 8.0
            )
            .as_bytes(),
        );
    }
    let reference = falx::kernels::csv_geo::parse_csv_geo_stats(&csv);

    // Several block sizes, all chosen to be coprime-ish with the record length
    // so boundaries land inside records rather than between them.
    for chunk_size in [37usize, 512, 1000, 4096, 60_000] {
        let chunks: Vec<&[u8]> = csv.chunks(chunk_size).collect();
        let data = build_bgzf(&chunks);

        // Sanity: this really is a boundary-straddling layout.
        let inflated = falx::bgzf::decompress_framed_par(&data, &bgzf(), 4).expect("decompress");
        assert_eq!(inflated, csv, "round trip lost data at chunk {chunk_size}");

        for threads in [1usize, 2, 3, 8, 32] {
            let states = falx::bgzf::parse_framed_records_par(
                &data,
                &bgzf(),
                threads,
                b'\n',
                || (0u64, 0u64, 0u64),
                |state, records| {
                    let s = falx::kernels::csv_geo::parse_csv_geo_stats(records);
                    state.0 += s.records;
                    state.1 = state.1.wrapping_add(s.latitude_checksum);
                    state.2 = state.2.wrapping_add(s.longitude_checksum);
                },
            )
            .expect("record-aware framed parse");

            let rows: u64 = states.iter().map(|s| s.0).sum();
            let lat = states.iter().fold(0u64, |a, s| a.wrapping_add(s.1));
            let lon = states.iter().fold(0u64, |a, s| a.wrapping_add(s.2));
            assert_eq!(
                rows, reference.records,
                "row count differs at chunk {chunk_size}, {threads} threads"
            );
            assert_eq!(
                lat, reference.latitude_checksum,
                "latitude checksum differs at chunk {chunk_size}, {threads} threads"
            );
            assert_eq!(
                lon, reference.longitude_checksum,
                "longitude checksum differs at chunk {chunk_size}, {threads} threads"
            );
        }
    }
}

/// A record longer than an entire worker's block group: the carry has to span
/// several workers, and blocks with no terminator at all must not be dropped.
#[cfg(feature = "bgzf")]
#[test]
fn a_record_longer_than_a_worker_group_survives() {
    // One enormous record, then a few small ones.
    let mut csv = Vec::new();
    csv.extend_from_slice(b"cc,");
    csv.extend_from_slice(&vec![b'x'; 300_000]);
    csv.extend_from_slice(b",accent,region,999,1.000000,-1.500000\n");
    for row in 0..5u64 {
        csv.extend_from_slice(
            format!("cc,city{row},accent,region,999,2.000000,-2.500000\n").as_bytes(),
        );
    }
    let reference = falx::kernels::csv_geo::parse_csv_geo_stats(&csv);

    let chunks: Vec<&[u8]> = csv.chunks(1024).collect();
    let data = build_bgzf(&chunks);
    for threads in [1usize, 4, 16, 64] {
        let states = falx::bgzf::parse_framed_records_par(
            &data,
            &bgzf(),
            threads,
            b'\n',
            || (0u64, 0u64),
            |state, records| {
                let s = falx::kernels::csv_geo::parse_csv_geo_stats(records);
                state.0 += s.records;
                state.1 = state.1.wrapping_add(s.latitude_checksum);
            },
        )
        .expect("parse");
        let rows: u64 = states.iter().map(|s| s.0).sum();
        let lat = states.iter().fold(0u64, |a, s| a.wrapping_add(s.1));
        assert_eq!(rows, reference.records, "row count at {threads} threads");
        assert_eq!(
            lat, reference.latitude_checksum,
            "checksum at {threads} threads"
        );
    }
}

/// A stream whose final record has no terminator must still be delivered.
#[cfg(feature = "bgzf")]
#[test]
fn unterminated_final_record_is_delivered() {
    let csv = b"cc,a,accent,region,999,1.000000,-1.500000\n\
                cc,b,accent,region,999,2.000000,-2.500000";
    let reference = falx::kernels::csv_geo::parse_csv_geo_stats(csv);
    let chunks: Vec<&[u8]> = csv.chunks(17).collect();
    let data = build_bgzf(&chunks);
    for threads in [1usize, 2, 8] {
        let states = falx::bgzf::parse_framed_records_par(
            &data,
            &bgzf(),
            threads,
            b'\n',
            || 0u64,
            |state, records| *state += falx::kernels::csv_geo::parse_csv_geo_stats(records).records,
        )
        .expect("parse");
        assert_eq!(
            states.iter().sum::<u64>(),
            reference.records,
            "unterminated trailing record lost at {threads} threads"
        );
    }
}

/// States come back in stream order, so a caller that concatenates rather than
/// reduces still sees the input's record order.
#[cfg(feature = "bgzf")]
#[test]
fn states_are_returned_in_stream_order() {
    let mut csv = Vec::new();
    for row in 0..5_000u64 {
        csv.extend_from_slice(format!("{row}\n").as_bytes());
    }
    let chunks: Vec<&[u8]> = csv.chunks(97).collect();
    let data = build_bgzf(&chunks);

    let states = falx::bgzf::parse_framed_records_par(
        &data,
        &bgzf(),
        8,
        b'\n',
        Vec::<u64>::new,
        |state, records| {
            for line in records.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                state.push(std::str::from_utf8(line).unwrap().parse().unwrap());
            }
        },
    )
    .expect("parse");

    let flattened: Vec<u64> = states.concat();
    let expected: Vec<u64> = (0..5_000).collect();
    assert_eq!(
        flattened, expected,
        "concatenated states are not in stream order"
    );
}

/// The seam logic is format-agnostic: with an identity decoder it parses
/// records out of a plain (uncompressed) length-prefixed container, no
/// decompressor involved. This is what living in `framing` rather than `bgzf`
/// buys — the record stitching is reusable by any container.
#[test]
fn record_stitching_works_without_a_decompressor() {
    let f = Framing {
        header: 4,
        length_at: 0,
        width: Width::U32,
        endian: Endian::Be,
        counts: Counts::Payload,
        adjust: 0,
        trailer: 0,
        magic: None,
        skip_empty: false,
        uncompressed: None,
    };
    // Records split across frames at arbitrary offsets.
    let mut payload = Vec::new();
    for row in 0..3_000u64 {
        payload.extend_from_slice(format!("{row}\n").as_bytes());
    }
    for chunk_size in [3usize, 7, 64, 999] {
        let mut data = Vec::new();
        for chunk in payload.chunks(chunk_size) {
            data.extend_from_slice(&(chunk.len() as u32).to_be_bytes());
            data.extend_from_slice(chunk);
        }
        let frames = framing::scan(&f, &data).expect("scan");

        for threads in [1usize, 2, 5, 16] {
            let states: Vec<Vec<u64>> = framing::parse_records_par(
                &data,
                &frames,
                threads,
                b'\n',
                Vec::new,
                // Identity decoder: the payload is already plain bytes.
                || {
                    |frame: &framing::Frame, data: &[u8], out: &mut Vec<u8>| -> Result<(), ()> {
                        out.extend_from_slice(&data[frame.payload.clone()]);
                        Ok(())
                    }
                },
                |state: &mut Vec<u64>, records: &[u8]| {
                    for line in records.split(|&b| b == b'\n') {
                        if !line.is_empty() {
                            state.push(std::str::from_utf8(line).unwrap().parse().unwrap());
                        }
                    }
                },
            )
            .expect("parse");

            let flattened: Vec<u64> = states.concat();
            let expected: Vec<u64> = (0..3_000).collect();
            assert_eq!(
                flattened, expected,
                "records lost or reordered at chunk {chunk_size}, {threads} threads"
            );
        }
    }
}

/// A decoder error propagates out rather than being swallowed or panicking.
#[test]
fn decoder_errors_propagate() {
    let f = Framing {
        header: 4,
        length_at: 0,
        width: Width::U32,
        endian: Endian::Be,
        counts: Counts::Payload,
        adjust: 0,
        trailer: 0,
        magic: None,
        skip_empty: false,
        uncompressed: None,
    };
    let mut data = Vec::new();
    for chunk in [b"aa\n".as_slice(), b"bb\n".as_slice()] {
        data.extend_from_slice(&(chunk.len() as u32).to_be_bytes());
        data.extend_from_slice(chunk);
    }
    let frames = framing::scan(&f, &data).expect("scan");
    let result: Result<Vec<()>, &'static str> = framing::parse_records_par(
        &data,
        &frames,
        2,
        b'\n',
        || (),
        || {
            |_f: &framing::Frame, _d: &[u8], _o: &mut Vec<u8>| -> Result<(), &'static str> {
                Err("decoder failed")
            }
        },
        |_s: &mut (), _r: &[u8]| {},
    );
    assert_eq!(result, Err("decoder failed"));
}

/// A framed module emits the record-aware driver, so a standalone generated
/// parser can reassemble records across frames without linking falx. Decoding
/// is a caller parameter, which is what keeps the generated file std-only.
#[test]
fn framed_modules_emit_the_record_aware_driver() {
    let text = "\
falx-ir 1
format framed_csv
structural 2c,0a
frame header=4 length-at=0 width=u32 endian=be counts=payload adjust=0 trailer=0
%0 = class 0a
%1 = class 2c,0a
output %1
terminators %0
";
    let module = falx::ir_text::parse(text).expect("parse");
    let code = falx::codegen::emit_module(&module).expect("emit");
    for item in [
        "pub fn parse_records_par",
        "pub const RECORD_TERMINATOR: u8 = b'\\n';",
        "MakeDecoder",
    ] {
        assert!(code.contains(item), "generated code is missing `{item}`");
    }
    // std-only: no falx paths leak into the generated file.
    assert!(
        !code.contains("falx::"),
        "generated code must not reference falx"
    );
}

/// A record in a grouped-line format ends every Nth terminator, which the
/// single-terminator stitching does not track — so the driver is deliberately
/// not emitted, and the generated file says why rather than shipping something
/// that would cut records mid-group.
#[test]
fn grouped_line_formats_omit_the_record_aware_driver() {
    let text = "\
falx-ir 1
format framed_fastq
structural 0a
lines-per-record 4
frame header=4 length-at=0 width=u32 endian=be counts=payload adjust=0 trailer=0
%0 = class 0a
output %0
terminators %0
";
    let module = falx::ir_text::parse(text).expect("parse");
    let code = falx::codegen::emit_module(&module).expect("emit");
    assert!(
        code.contains("pub fn scan_frames"),
        "frame location should still be emitted"
    );
    assert!(
        !code.contains("pub fn parse_records_par"),
        "the record-aware driver must not be emitted for grouped-line formats"
    );
    assert!(
        code.contains("spans 4 terminators"),
        "the generated file should explain the omission, got:\n{}",
        &code[code.len().saturating_sub(600)..]
    );
}
