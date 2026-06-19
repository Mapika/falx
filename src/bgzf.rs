//! Block-parallel bgzf decompression.
//!
//! bgzf (the genomics gzip — `.vcf.gz`, `.bam`, bgzipped FASTQ) frames the
//! stream as a sequence of independent gzip members of at most 64 KiB each,
//! with the compressed block length (`BSIZE`) carried in an extra subfield and
//! the uncompressed length (`ISIZE`) in the gzip trailer. Block independence is
//! the whole point: we scan the boundaries in one cheap pass, presize the
//! output from the `ISIZE` tags, and inflate the blocks across threads into
//! disjoint slots — the same shape as the parallel structural indexer's scatter
//! merge. The default inner DEFLATE core is `flate2`'s pure-Rust
//! `miniz_oxide` backend; enabling `bgzf-libdeflate` switches inflation to
//! bundled C libdeflate for maximum throughput. We do not reimplement entropy
//! decoding.
//!
//! This exists so the multi-GiB/s SIMD parsers are not starved behind a
//! ~0.3 GiB/s single-threaded gunzip on compressed inputs. Gated behind the
//! `bgzf` feature so the core and the generated kernels stay std-only.

use std::io::{self, Error, ErrorKind};
use std::ops::Range;

#[cfg(not(feature = "bgzf-libdeflate"))]
use flate2::{Decompress, FlushDecompress};

/// Minimum bytes of a bgzf block header before the variable extra field.
const HEADER_LEN: usize = 12;
/// gzip trailer: CRC32 (4) + ISIZE (4).
const TRAILER_LEN: usize = 8;

/// One decompressible block: the raw-DEFLATE payload range within the input
/// and its uncompressed size. Blocks are independent, so a consumer can inflate
/// any contiguous run on its own (the fusion path decompresses one group per
/// worker straight into the parser).
#[derive(Clone, Debug)]
pub struct Block {
    pub payload: Range<usize>,
    pub isize: usize,
}

/// Read the `BC` subfield (`SI1=66`, `SI2=67`, `SLEN=2`) from a block's gzip
/// extra field and return `BSIZE` (total block size minus one).
fn read_bsize(extra: &[u8]) -> io::Result<usize> {
    let mut i = 0;
    while i + 4 <= extra.len() {
        let si1 = extra[i];
        let si2 = extra[i + 1];
        let slen = u16::from_le_bytes([extra[i + 2], extra[i + 3]]) as usize;
        if si1 == b'B' && si2 == b'C' && slen == 2 && i + 4 + 2 <= extra.len() {
            return Ok(u16::from_le_bytes([extra[i + 4], extra[i + 5]]) as usize);
        }
        i += 4 + slen;
    }
    Err(Error::new(
        ErrorKind::InvalidData,
        "bgzf block missing BC extra subfield",
    ))
}

/// Scan every block boundary in one linear pass (no inflation), returning the
/// independent blocks in stream order. Empty bgzf EOF blocks are dropped.
pub fn scan(data: &[u8]) -> io::Result<Vec<Block>> {
    let mut blocks = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        if pos + HEADER_LEN > data.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "truncated bgzf header",
            ));
        }
        if data[pos] != 0x1f || data[pos + 1] != 0x8b {
            return Err(Error::new(ErrorKind::InvalidData, "not a bgzf/gzip member"));
        }
        // CM=8 (deflate), FLG must have FEXTRA (0x04) set for bgzf.
        if data[pos + 2] != 8 || data[pos + 3] & 0x04 == 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "not a bgzf block (no FEXTRA)",
            ));
        }
        let xlen = u16::from_le_bytes([data[pos + 10], data[pos + 11]]) as usize;
        let extra_end = pos + HEADER_LEN + xlen;
        if extra_end > data.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "truncated bgzf extra field",
            ));
        }
        let block_len = read_bsize(&data[pos + HEADER_LEN..extra_end])? + 1;
        if block_len <= HEADER_LEN + xlen + TRAILER_LEN || pos + block_len > data.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "bgzf block length out of range",
            ));
        }
        let payload = extra_end..pos + block_len - TRAILER_LEN;
        let isize = u32::from_le_bytes([
            data[pos + block_len - 4],
            data[pos + block_len - 3],
            data[pos + block_len - 2],
            data[pos + block_len - 1],
        ]) as usize;
        // The bgzf EOF marker is an empty block (ISIZE 0); skip empties.
        if isize > 0 {
            blocks.push(Block { payload, isize });
        }
        pos += block_len;
    }
    Ok(blocks)
}

#[cfg(feature = "bgzf-libdeflate")]
struct InflateBackend {
    decompressor: libdeflater::Decompressor,
}

#[cfg(feature = "bgzf-libdeflate")]
impl InflateBackend {
    fn new() -> Self {
        Self {
            decompressor: libdeflater::Decompressor::new(),
        }
    }

    fn inflate_one(&mut self, payload: &[u8], out: &mut [u8]) -> io::Result<()> {
        let n = self
            .decompressor
            .deflate_decompress(payload, out)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        if n != out.len() {
            return Err(inflated_len_error());
        }
        Ok(())
    }

    #[cfg(all(test, feature = "bgzf-libdeflate"))]
    fn name() -> &'static str {
        "libdeflate"
    }
}

#[cfg(not(feature = "bgzf-libdeflate"))]
struct InflateBackend;

#[cfg(not(feature = "bgzf-libdeflate"))]
impl InflateBackend {
    fn new() -> Self {
        Self
    }

    /// Inflate one raw-DEFLATE block into its exact-size output slice.
    fn inflate_one(&mut self, payload: &[u8], out: &mut [u8]) -> io::Result<()> {
        let mut d = Decompress::new(false); // raw DEFLATE: bgzf carries no zlib header
        d.decompress(payload, out, FlushDecompress::Finish)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        if d.total_out() as usize != out.len() {
            return Err(inflated_len_error());
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "bgzf-libdeflate"))]
fn inflate_backend_name_for_test() -> &'static str {
    InflateBackend::name()
}

fn inflated_len_error() -> io::Error {
    Error::new(
        ErrorKind::InvalidData,
        "bgzf block ISIZE disagrees with inflated length",
    )
}

/// Total uncompressed size of `blocks`.
fn span(blocks: &[Block]) -> usize {
    blocks.iter().map(|b| b.isize).sum()
}

/// Inflate `blocks` (a contiguous run) sequentially into `out`, which must be
/// exactly their combined uncompressed length.
fn inflate_blocks_into(data: &[u8], blocks: &[Block], out: &mut [u8]) -> io::Result<()> {
    let mut backend = InflateBackend::new();
    let mut off = 0usize;
    for b in blocks {
        backend.inflate_one(&data[b.payload.clone()], &mut out[off..off + b.isize])?;
        off += b.isize;
    }
    Ok(())
}

/// Decompress a contiguous run of already-scanned blocks into a fresh buffer.
/// The building block for the fusion path: a worker inflates exactly its group.
pub fn inflate_range(data: &[u8], blocks: &[Block]) -> io::Result<Vec<u8>> {
    let mut out = vec![0u8; span(blocks)];
    inflate_blocks_into(data, blocks, &mut out)?;
    Ok(out)
}

/// Decompress a whole bgzf stream on one thread.
pub fn decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    inflate_range(data, &scan(data)?)
}

/// Decompress a whole bgzf stream across `threads` workers. Blocks are
/// partitioned into contiguous runs whose output ranges are disjoint, so each
/// worker inflates straight into its own `split_at_mut` slice — no locking, no
/// post-merge copy.
pub fn decompress_par(data: &[u8], threads: usize) -> io::Result<Vec<u8>> {
    let blocks = scan(data)?;
    let threads = threads.max(1).min(blocks.len().max(1));
    let mut out = vec![0u8; span(&blocks)];
    if threads == 1 || blocks.len() <= 1 {
        inflate_blocks_into(data, &blocks, &mut out)?;
        return Ok(out);
    }

    // Contiguous block runs of roughly equal block count, each handed the exact
    // output slice its blocks cover.
    let per = blocks.len().div_ceil(threads);
    let groups: Vec<&[Block]> = blocks.chunks(per).collect();
    let mut slots: Vec<&mut [u8]> = Vec::with_capacity(groups.len());
    let mut rest = out.as_mut_slice();
    for g in &groups {
        let (head, tail) = rest.split_at_mut(span(g));
        slots.push(head);
        rest = tail;
    }

    std::thread::scope(|s| {
        let handles: Vec<_> = groups
            .into_iter()
            .zip(slots)
            .map(|(group, slot)| s.spawn(move || inflate_blocks_into(data, group, slot)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("bgzf worker panicked"))
            .collect::<io::Result<Vec<()>>>()
    })?;
    Ok(out)
}

/// Process a bgzf stream block-by-block across `threads` workers without
/// materializing one full decompressed output buffer.
///
/// Each worker owns one inflater, one reusable scratch buffer sized to the
/// largest block in its contiguous block group, and one caller-defined state
/// value from `init`. `process` is called in block order within that worker's
/// group. The returned states are ordered by their input ranges, so callers can
/// concatenate or reduce them in stream order.
///
/// This is the fastest shape for consumers that can process independent bgzf
/// blocks directly, such as counters, checksums, and parsers with explicit
/// carry handling.
pub fn process_blocks_par<S, Init, F>(
    data: &[u8],
    threads: usize,
    init: Init,
    process: F,
) -> io::Result<Vec<S>>
where
    S: Send,
    Init: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &[u8]) + Sync,
{
    let blocks = scan(data)?;
    let n = blocks.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let threads = threads.max(1).min(n);
    let per = n.div_ceil(threads);
    let init = &init;
    let process = &process;
    let blocks = &blocks;

    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .filter_map(|w| {
                let start = w * per;
                let end = ((w + 1) * per).min(n);
                (start < end).then(|| {
                    s.spawn(move || -> io::Result<S> {
                        let group = &blocks[start..end];
                        let max_isize = group.iter().map(|b| b.isize).max().unwrap_or(0);
                        let mut scratch = vec![0u8; max_isize];
                        let mut backend = InflateBackend::new();
                        let mut state = init();
                        for (local, b) in group.iter().enumerate() {
                            backend
                                .inflate_one(&data[b.payload.clone()], &mut scratch[..b.isize])?;
                            process(&mut state, start + local, &scratch[..b.isize]);
                        }
                        Ok(state)
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("bgzf block processor panicked"))
            .collect()
    })
}

/// Fused parallel parse of a bgzf stream: never materializes the full
/// decompressed buffer. Each worker inflates its own contiguous block group,
/// completes the record straddling its tail (inflating into following blocks
/// until `terminator`), drops its leading partial record (the previous worker
/// owns it via line ownership), and runs `parse` on the record-aligned local
/// buffer while it is hot in cache. Returns one `parse` result per worker, in
/// stream order — the caller concatenates them.
///
/// `parse` must treat its input as a self-contained run of whole `terminator`-
/// delimited records (exactly what a generated `parse_columns` expects: leading
/// comment/header lines are skipped, the final record is `terminator`-ended).
/// This is the building block behind `.vcf.gz`/`.fastq.gz` → columnar parsing.
pub fn parse_gz_par<C, F>(
    comp: &[u8],
    threads: usize,
    terminator: u8,
    parse: F,
) -> io::Result<Vec<C>>
where
    F: Fn(&[u8]) -> C + Sync,
    C: Send,
{
    let blocks = scan(comp)?;
    let n = blocks.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let threads = threads.max(1).min(n);
    let per = n.div_ceil(threads);
    let bounds: Vec<usize> = (0..=threads).map(|w| (w * per).min(n)).collect();
    let parse = &parse;
    let blocks = &blocks;
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .filter(|&w| bounds[w] < bounds[w + 1])
            .map(|w| {
                let bounds = &bounds;
                s.spawn(move || -> io::Result<C> {
                    let mut own = inflate_range(comp, &blocks[bounds[w]..bounds[w + 1]])?;
                    // Finish the record straddling this group's tail.
                    let mut k = bounds[w + 1];
                    while k < n {
                        let chunk = inflate_range(comp, &blocks[k..k + 1])?;
                        if let Some(nl) = chunk.iter().position(|&b| b == terminator) {
                            own.extend_from_slice(&chunk[..=nl]);
                            break;
                        }
                        own.extend_from_slice(&chunk);
                        k += 1;
                    }
                    // Drop the leading partial record (worker 0 keeps everything).
                    let lead = if w == 0 {
                        0
                    } else {
                        own.iter()
                            .position(|&b| b == terminator)
                            .map_or(own.len(), |i| i + 1)
                    };
                    Ok(parse(&own[lead..]))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("bgzf fusion worker panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// bgzip `data` with the same framing real tools emit (via flate2's gzip
    /// writer is not bgzf, so we hand-roll minimal bgzf blocks here).
    fn bgzip(data: &[u8]) -> Vec<u8> {
        use flate2::{Compress, Compression, FlushCompress};
        let mut out = Vec::new();
        // Up to 64 KiB uncompressed per block, per the bgzf spec.
        for chunk in data
            .chunks(0xFF00)
            .chain(if data.is_empty() { Some(data) } else { None })
        {
            let mut deflated = vec![0u8; chunk.len() + 1024];
            let mut c = Compress::new(Compression::default(), false);
            c.compress(chunk, &mut deflated, FlushCompress::Finish)
                .unwrap();
            let n = c.total_out() as usize;
            deflated.truncate(n);
            let block_len = HEADER_LEN + 6 + n + TRAILER_LEN; // 6 = BC subfield
            let mut hdr = [0u8; HEADER_LEN + 6];
            hdr[0] = 0x1f;
            hdr[1] = 0x8b;
            hdr[2] = 8; // deflate
            hdr[3] = 0x04; // FEXTRA
            hdr[8] = 0; // XFL
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
            out.write_all(&crc32(chunk).to_le_bytes()).unwrap();
            out.write_all(&(chunk.len() as u32).to_le_bytes()).unwrap();
        }
        out
    }

    fn crc32(data: &[u8]) -> u32 {
        // Minimal CRC32 (the decompressor ignores it, but keep blocks valid).
        let mut crc = !0u32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    #[test]
    fn roundtrip_serial_and_parallel_agree() {
        // Several blocks plus a partial final block.
        let mut raw = Vec::new();
        for i in 0..200_000u32 {
            raw.extend_from_slice(format!("line {i}: the quick brown fox\n").as_bytes());
        }
        let comp = bgzip(&raw);
        assert_eq!(decompress(&comp).unwrap(), raw, "serial");
        for t in [2usize, 3, 8] {
            assert_eq!(decompress_par(&comp, t).unwrap(), raw, "parallel t={t}");
        }
    }

    #[test]
    fn empty_input_decompresses_to_empty() {
        let comp = bgzip(&[]);
        assert!(decompress(&comp).unwrap().is_empty());
        assert!(decompress_par(&comp, 4).unwrap().is_empty());
    }

    #[test]
    fn rejects_non_bgzf() {
        assert!(decompress(b"not a gzip stream at all").is_err());
    }

    #[cfg(feature = "bgzf-libdeflate")]
    #[test]
    fn libdeflate_backend_is_active_and_correct() {
        let mut raw = Vec::new();
        for i in 0..20_000u32 {
            raw.extend_from_slice(format!("libdeflate record {i}\n").as_bytes());
        }
        let comp = bgzip(&raw);
        assert_eq!(inflate_backend_name_for_test(), "libdeflate");
        assert_eq!(decompress(&comp).unwrap(), raw);
        assert_eq!(decompress_par(&comp, 4).unwrap(), raw);
    }

    fn lines(s: &[u8]) -> Vec<Vec<u8>> {
        s.split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .map(<[u8]>::to_vec)
            .collect()
    }

    #[test]
    fn parse_gz_par_is_record_aligned_with_no_loss() {
        // Records of varied width so block boundaries land mid-record.
        let mut raw = Vec::new();
        for i in 0..300_000u32 {
            raw.extend_from_slice(
                format!("rec{i}\t{}\tend\n", "x".repeat((i % 37) as usize)).as_bytes(),
            );
        }
        let comp = bgzip(&raw);
        let expect = lines(&raw);
        // Each worker parses whole records; concatenation must reconstruct the
        // exact sequence regardless of where block/worker boundaries fall.
        for t in [1usize, 2, 3, 8, 16] {
            let parts = parse_gz_par(&comp, t, b'\n', lines).unwrap();
            let merged: Vec<Vec<u8>> = parts.into_iter().flatten().collect();
            assert_eq!(merged, expect, "record loss/dup at boundaries, t={t}");
        }
    }

    #[test]
    fn process_blocks_par_streams_blocks_in_order() {
        let mut raw = Vec::new();
        for i in 0..120_000u32 {
            raw.extend_from_slice(format!("block-stream record {i}\n").as_bytes());
        }
        let comp = bgzip(&raw);
        let parts = process_blocks_par(
            &comp,
            8,
            Vec::<u8>::new,
            |out: &mut Vec<u8>, _block_index, chunk| out.extend_from_slice(chunk),
        )
        .unwrap();
        let merged: Vec<u8> = parts.into_iter().flatten().collect();
        assert_eq!(merged, raw);
    }
}
