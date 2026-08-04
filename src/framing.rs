//! Length-prefixed framing: the outer container layer of the IR.
//!
//! The bitstream graph in [`crate::ir`] can only express framing determined by
//! *byte identity* — "this byte is a comma" — because every operation is
//! data-parallel over bytes. A length-prefixed format is the opposite: where
//! frame N+1 starts is a value decoded out of frame N, so the boundaries form a
//! sequential dependency chain that no amount of SIMD removes. That is a real
//! property of the format, not a gap in the implementation.
//!
//! What *is* exploitable is the two-level structure such formats share. The
//! chain is cheap — a bounded amount of pointer arithmetic per frame, and
//! frames are typically kilobytes — while everything downstream of it is
//! embarrassingly parallel, because the frames are independent. So the shape
//! is: scan boundaries once, sequentially; then decompress, parse, or reduce
//! the frames across every core. That is precisely how [`crate::bgzf`] reaches
//! ~10 GiB/s on a format whose boundaries are strictly sequential, and this
//! module generalizes that hand-written shape into something a spec or IR
//! module can describe.
//!
//! Block-compressed containers (bgzf, and the block layer of Parquet/ORC-style
//! formats) are the main beneficiary: their outer level *is* a length-prefixed
//! frame chain. falx does not decode entropy-coded payloads — it locates
//! frames and hands them to a decompressor and then to a payload parser.

use std::ops::Range;

/// Width of the encoded length field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Width {
    U8,
    U16,
    U32,
    U64,
    /// Unsigned LEB128, as used by protobuf-style length-delimited streams.
    /// The field's own size is data-dependent, so it also shifts where the
    /// payload starts.
    Varint,
}

impl Width {
    /// Fixed encoded size, or `None` for a variable-length encoding.
    pub const fn fixed_size(self) -> Option<usize> {
        match self {
            Width::U8 => Some(1),
            Width::U16 => Some(2),
            Width::U32 => Some(4),
            Width::U64 => Some(8),
            Width::Varint => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Width::U8 => "u8",
            Width::U16 => "u16",
            Width::U32 => "u32",
            Width::U64 => "u64",
            Width::Varint => "varint",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Endian {
    Le,
    Be,
}

/// What the decoded length counts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Counts {
    /// The whole frame, header and trailer included (bgzf's `BSIZE`).
    Total,
    /// Only the payload, so the frame is header + payload + trailer.
    Payload,
}

/// Where a frame records its *uncompressed* payload size.
///
/// Block-compressed containers usually carry this (bgzf's `ISIZE`), and it is
/// what makes parallel decompression possible: knowing every block's inflated
/// size up front lets the output be presized once and carved into disjoint
/// slots, so workers inflate concurrently with no locking and no merge pass.
/// Without it a decompressor has to grow buffers sequentially.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SizeField {
    /// Offset of the field. Non-negative counts from the frame start;
    /// negative counts back from the frame end, so bgzf's trailing `ISIZE`
    /// is `-4`.
    pub at: i64,
    pub width: Width,
    pub endian: Endian,
}

/// How a stream is divided into length-prefixed frames.
///
/// The canonical bgzf block is
/// `header 18, length-at 16, u16 le, counts total, adjust 1, trailer 8,
/// magic 0:1f,8b`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Framing {
    /// Offset where the payload begins, for fixed-width lengths. With a
    /// varint the payload begins immediately after the varint, and this is
    /// the number of extra header bytes after it (usually 0).
    pub header: usize,
    /// Offset of the length field within the frame.
    pub length_at: usize,
    pub width: Width,
    pub endian: Endian,
    pub counts: Counts,
    /// Added to the decoded length. bgzf stores "total size minus one", so
    /// it needs `+1`.
    pub adjust: i64,
    /// Bytes at the end of the frame that are not payload (bgzf's CRC32 +
    /// ISIZE trailer).
    pub trailer: usize,
    /// Bytes that must appear at a given offset, else the frame is rejected.
    pub magic: Option<(usize, Vec<u8>)>,
    /// Drop frames whose payload is empty.
    pub skip_empty: bool,
    /// Where the frame records its uncompressed payload size, if it does.
    pub uncompressed: Option<SizeField>,
}

/// One located frame: its extent in the input and the payload within it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Frame {
    /// Offset of the frame in the input.
    pub start: usize,
    /// Total frame size, header and trailer included.
    pub len: usize,
    /// The payload's range in the input.
    pub payload: Range<usize>,
    /// Decoded uncompressed payload size, when the framing declares one.
    pub uncompressed: Option<usize>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FramingError(pub String);

impl std::fmt::Display for FramingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FramingError {}

/// Decode an unsigned LEB128 value at `data[at..]`, returning it with the
/// number of bytes consumed.
fn read_varint(data: &[u8], at: usize) -> Result<(u64, usize), FramingError> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    let mut i = 0usize;
    loop {
        let byte = *data
            .get(at + i)
            .ok_or_else(|| FramingError(format!("truncated varint at offset {at}")))?;
        // 10 bytes is the most a u64 can occupy; past that the value cannot
        // be represented and the stream is malformed.
        if shift >= 64 {
            return Err(FramingError(format!("varint at offset {at} overflows u64")));
        }
        value |= u64::from(byte & 0x7f) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return Ok((value, i));
        }
        shift += 7;
    }
}

/// Locate every frame in `data`.
///
/// This is the sequential half — one bounded step per frame, no inflation and
/// no payload inspection — after which the returned frames are independent and
/// can be processed across threads.
pub fn scan(framing: &Framing, data: &[u8]) -> Result<Vec<Frame>, FramingError> {
    let mut frames = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let frame = frame_at(framing, data, pos)?;
        pos = frame.start + frame.len;
        if framing.skip_empty && frame.payload.is_empty() {
            continue;
        }
        frames.push(frame);
    }
    Ok(frames)
}

/// Decode the single frame beginning at `pos`.
pub fn frame_at(framing: &Framing, data: &[u8], pos: usize) -> Result<Frame, FramingError> {
    if let Some((offset, bytes)) = &framing.magic {
        let from = pos + offset;
        let to = from + bytes.len();
        if to > data.len() {
            return Err(FramingError(format!(
                "truncated frame header at offset {pos}"
            )));
        }
        if &data[from..to] != bytes.as_slice() {
            return Err(FramingError(format!(
                "frame at offset {pos} does not start with the declared magic"
            )));
        }
    }

    let (decoded, payload_start) = match framing.width {
        Width::Varint => {
            let (value, used) = read_varint(data, pos + framing.length_at)?;
            (value, framing.length_at + used + framing.header)
        }
        fixed => {
            let size = fixed.fixed_size().expect("non-varint width is fixed");
            let from = pos + framing.length_at;
            let to = from + size;
            if to > data.len() {
                return Err(FramingError(format!(
                    "truncated length field at offset {pos}"
                )));
            }
            let mut buf = [0u8; 8];
            buf[..size].copy_from_slice(&data[from..to]);
            let value = match framing.endian {
                Endian::Le => u64::from_le_bytes(buf),
                Endian::Be => {
                    // Big-endian fields occupy the *high* bytes of the value.
                    let mut be = [0u8; 8];
                    be[8 - size..].copy_from_slice(&data[from..to]);
                    u64::from_be_bytes(be)
                }
            };
            (value, framing.header)
        }
    };

    let adjusted = i128::from(decoded) + i128::from(framing.adjust);
    if adjusted < 0 {
        return Err(FramingError(format!(
            "frame at offset {pos} has a negative length after adjustment"
        )));
    }
    let adjusted = adjusted as u128;
    let len = match framing.counts {
        Counts::Total => adjusted,
        Counts::Payload => adjusted + payload_start as u128 + framing.trailer as u128,
    };
    let len = usize::try_from(len)
        .map_err(|_| FramingError(format!("frame at offset {pos} is impossibly large")))?;

    // A frame must at least hold its own header and trailer, must not be
    // empty (which would not advance the scan), and must fit in the input.
    if len < payload_start + framing.trailer || len == 0 {
        return Err(FramingError(format!(
            "frame at offset {pos} declares length {len}, too small for its header and trailer"
        )));
    }
    if pos + len > data.len() {
        return Err(FramingError(format!(
            "frame at offset {pos} declares length {len} but only {} bytes remain",
            data.len() - pos
        )));
    }

    let uncompressed = match framing.uncompressed {
        Some(field) => Some(read_size_field(&field, data, pos, len)?),
        None => None,
    };

    Ok(Frame {
        start: pos,
        len,
        payload: pos + payload_start..pos + len - framing.trailer,
        uncompressed,
    })
}

/// Read a size field relative to a located frame.
fn read_size_field(
    field: &SizeField,
    data: &[u8],
    pos: usize,
    len: usize,
) -> Result<usize, FramingError> {
    let size = field
        .width
        .fixed_size()
        .ok_or_else(|| FramingError("a size field cannot be varint-encoded".into()))?;
    let from = if field.at >= 0 {
        pos + field.at as usize
    } else {
        let back = field.at.unsigned_abs() as usize;
        if back > len {
            return Err(FramingError(format!(
                "size field at {} lies before the frame at offset {pos}",
                field.at
            )));
        }
        pos + len - back
    };
    let to = from + size;
    // The frame's extent was already validated against the input, so the only
    // way to land outside it is a size field declared out of range.
    if to > pos + len || to > data.len() {
        return Err(FramingError(format!(
            "size field at {} lies outside the frame at offset {pos}",
            field.at
        )));
    }
    let mut buf = [0u8; 8];
    Ok(match field.endian {
        Endian::Le => {
            buf[..size].copy_from_slice(&data[from..to]);
            u64::from_le_bytes(buf) as usize
        }
        Endian::Be => {
            buf[8 - size..].copy_from_slice(&data[from..to]);
            u64::from_be_bytes(buf) as usize
        }
    })
}

/// Parse **whole records** out of framed data across `threads` workers,
/// including records that straddle frame and worker boundaries.
///
/// Frames are independent, but records generally are not aligned to them, so
/// splitting work by frame and parsing each in isolation would cut records in
/// half. This handles that generically, for any container and any per-frame
/// decoding:
///
/// * Each worker takes a contiguous run of frames and decodes them one at a
///   time, holding over whatever follows the last terminator in each. The
///   working set is therefore one frame plus a partial record, which stays
///   cache-resident — decoding a worker's whole run into a single buffer first
///   is simpler but measurably slower once a run exceeds cache.
/// * The partial record at each end of a worker's run is kept aside, and the
///   seams are stitched serially afterwards: at most one record per worker.
/// * A stitched record is attributed to the worker whose run it *starts* in,
///   the same record-ownership rule the parallel structural parsers use, so
///   the returned states stay in stream order and concatenate directly.
///
/// Records longer than a worker's entire run are handled (the carry simply
/// spans as many workers as it needs), as is a final record with no
/// terminator. `process` is only ever called with complete records.
///
/// `make_decoder` is called once per worker and returns that worker's decoder,
/// which appends one frame's decoded bytes to the buffer it is given. For an
/// uncompressed container that is a copy; [`crate::bgzf::parse_framed_records_par`]
/// supplies one that inflates.
pub fn parse_records_par<S, E, Init, MakeDecoder, Decoder, Process>(
    data: &[u8],
    frames: &[Frame],
    threads: usize,
    terminator: u8,
    init: Init,
    make_decoder: MakeDecoder,
    process: Process,
) -> Result<Vec<S>, E>
where
    S: Send,
    E: Send,
    Init: Fn() -> S + Sync,
    MakeDecoder: Fn() -> Decoder + Sync,
    Decoder: FnMut(&Frame, &[u8], &mut Vec<u8>) -> Result<(), E>,
    Process: Fn(&mut S, &[u8]) + Sync,
{
    if frames.is_empty() {
        return Ok(Vec::new());
    }
    let threads = threads.max(1).min(frames.len());
    let per = frames.len().div_ceil(threads);

    // Per worker: the leading partial record, the trailing partial record, and
    // whether any record ended inside its run at all.
    type Edges = (Vec<u8>, Vec<u8>, bool);

    let parts: Vec<Result<(S, Edges), E>> = std::thread::scope(|s| {
        let handles: Vec<_> = frames
            .chunks(per)
            .enumerate()
            .map(|(w, group)| {
                let init = &init;
                let make_decoder = &make_decoder;
                let process = &process;
                s.spawn(move || -> Result<(S, Edges), E> {
                    let mut decode = make_decoder();
                    let mut state = init();
                    let mut pending: Vec<u8> = Vec::new();
                    let mut prefix: Vec<u8> = Vec::new();
                    let mut seen_terminator = false;

                    for frame in group {
                        decode(frame, data, &mut pending)?;
                        let Some(last) = pending.iter().rposition(|&b| b == terminator) else {
                            // No record ends in this frame; keep accumulating.
                            continue;
                        };
                        let mut from = 0;
                        if !seen_terminator && w != 0 {
                            // Worker 0's leading bytes are the stream's first
                            // record; every other worker's are the tail of the
                            // previous worker's record, handed back as a
                            // fragment for the seam pass.
                            let first = pending
                                .iter()
                                .position(|&b| b == terminator)
                                .expect("a last terminator implies a first");
                            prefix = pending[..=first].to_vec();
                            from = first + 1;
                        }
                        seen_terminator = true;
                        process(&mut state, &pending[from..=last]);
                        // What follows the last terminator starts a record that
                        // continues into the next frame.
                        pending.drain(..=last);
                    }

                    let edges = if seen_terminator {
                        (prefix, pending, true)
                    } else {
                        // The whole run is the middle of one long record.
                        (pending, Vec::new(), false)
                    };
                    Ok((state, edges))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("framing worker panicked"))
            .collect()
    });

    let mut states = Vec::with_capacity(parts.len());
    let mut edges = Vec::with_capacity(parts.len());
    for part in parts {
        let (state, edge) = part?;
        states.push(state);
        edges.push(edge);
    }

    // Serial seam pass. `carry_owner` tracks which worker the in-flight record
    // started in, which is what keeps the states in stream order.
    let mut carry: Vec<u8> = Vec::new();
    let mut carry_owner = 0usize;
    for (w, (prefix, suffix, had_terminator)) in edges.iter().enumerate() {
        carry.extend_from_slice(prefix);
        if *had_terminator {
            if !carry.is_empty() {
                process(&mut states[carry_owner], &carry);
            }
            carry.clear();
            carry.extend_from_slice(suffix);
            carry_owner = w;
        }
    }
    // Anything left is a final record with no terminator.
    if !carry.is_empty() {
        process(&mut states[carry_owner], &carry);
    }
    Ok(states)
}

/// The canonical bgzf block layout, as a framing descriptor.
///
/// Provided as the worked reference for the model — [`crate::bgzf`] keeps its
/// own hand-written scanner, which additionally walks the gzip extra field to
/// find `BSIZE` wherever it sits rather than assuming the canonical offset.
/// The two agree on every block boundary for canonical bgzf, which is what
/// `tests/framing.rs` checks.
pub fn bgzf_framing() -> Framing {
    Framing {
        header: 18,
        length_at: 16,
        width: Width::U16,
        endian: Endian::Le,
        counts: Counts::Total,
        adjust: 1,
        trailer: 8,
        magic: Some((0, vec![0x1f, 0x8b])),
        skip_empty: false,
        // ISIZE: the last four bytes of the block.
        uncompressed: Some(SizeField {
            at: -4,
            width: Width::U32,
            endian: Endian::Le,
        }),
    }
}
