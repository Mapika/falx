//! Block-parallel bgzf decompression.
//!
//! bgzf (the genomics gzip — `.vcf.gz`, `.bam`, bgzipped FASTQ) frames the
//! stream as a sequence of independent gzip members of at most 64 KiB each,
//! with the compressed block length (`BSIZE`) carried in an extra subfield and
//! the uncompressed length (`ISIZE`) in the gzip trailer. Block independence is
//! the whole point: we scan the boundaries in one cheap pass, presize the
//! output from the `ISIZE` tags, and inflate the blocks across threads into
//! disjoint slots — the same shape as the parallel structural indexer's scatter
//! merge. The inner DEFLATE core is `flate2`'s pure-Rust `miniz_oxide` backend;
//! we do not reimplement entropy decoding.
//!
//! This exists so the multi-GiB/s SIMD parsers are not starved behind a
//! ~0.3 GiB/s single-threaded gunzip on compressed inputs. Gated behind the
//! `bgzf` feature so the core and the generated kernels stay std-only.

use std::io::{self, Error, ErrorKind};
use std::ops::Range;

use flate2::{Decompress, FlushDecompress};

/// Minimum bytes of a bgzf block header before the variable extra field.
const HEADER_LEN: usize = 12;
/// gzip trailer: CRC32 (4) + ISIZE (4).
const TRAILER_LEN: usize = 8;

/// One decompressible block: the raw-DEFLATE payload range within the input,
/// the uncompressed size, and the offset of its output in the result buffer.
struct Block {
    payload: Range<usize>,
    isize: usize,
    out_off: usize,
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

/// Scan every block boundary, returning the block list and the total
/// uncompressed length. One linear pass, no inflation.
fn scan_blocks(data: &[u8]) -> io::Result<(Vec<Block>, usize)> {
    let mut blocks = Vec::new();
    let mut pos = 0usize;
    let mut out_total = 0usize;
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
            blocks.push(Block {
                payload,
                isize,
                out_off: out_total,
            });
            out_total += isize;
        }
        pos += block_len;
    }
    Ok((blocks, out_total))
}

/// Inflate one raw-DEFLATE block into its exact-size output slice.
fn inflate_into(payload: &[u8], out: &mut [u8]) -> io::Result<()> {
    let mut d = Decompress::new(false); // raw DEFLATE: bgzf carries no zlib header
    d.decompress(payload, out, FlushDecompress::Finish)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    if d.total_out() as usize != out.len() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "bgzf block ISIZE disagrees with inflated length",
        ));
    }
    Ok(())
}

/// Decompress a whole bgzf stream on one thread.
pub fn decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    let (blocks, total) = scan_blocks(data)?;
    let mut out = vec![0u8; total];
    for b in &blocks {
        inflate_into(
            &data[b.payload.clone()],
            &mut out[b.out_off..b.out_off + b.isize],
        )?;
    }
    Ok(out)
}

/// Decompress a whole bgzf stream across `threads` workers. Blocks are
/// partitioned into contiguous runs whose output ranges are disjoint, so each
/// worker inflates straight into its own `split_at_mut` slice — no locking, no
/// post-merge copy.
pub fn decompress_par(data: &[u8], threads: usize) -> io::Result<Vec<u8>> {
    let (blocks, total) = scan_blocks(data)?;
    let threads = threads.max(1).min(blocks.len().max(1));
    if threads == 1 || blocks.len() <= 1 {
        let mut out = vec![0u8; total];
        for b in &blocks {
            inflate_into(
                &data[b.payload.clone()],
                &mut out[b.out_off..b.out_off + b.isize],
            )?;
        }
        return Ok(out);
    }

    let mut out = vec![0u8; total];
    // Contiguous block runs of roughly equal block count.
    let per = blocks.len().div_ceil(threads);
    let groups: Vec<&[Block]> = blocks.chunks(per).collect();

    // Hand each group the exact contiguous output slice its blocks cover.
    let mut slots: Vec<&mut [u8]> = Vec::with_capacity(groups.len());
    let mut rest = out.as_mut_slice();
    for g in &groups {
        let span: usize = g.iter().map(|b| b.isize).sum();
        let (head, tail) = rest.split_at_mut(span);
        slots.push(head);
        rest = tail;
    }

    let result = std::thread::scope(|s| {
        let handles: Vec<_> = groups
            .into_iter()
            .zip(slots)
            .map(|(group, slot)| {
                // Offsets within this group's slice are absolute out_off minus
                // the group's first block offset (chunks() never yields empty).
                let group_base = group[0].out_off;
                s.spawn(move || -> io::Result<()> {
                    for b in group {
                        let lo = b.out_off - group_base;
                        inflate_into(&data[b.payload.clone()], &mut slot[lo..lo + b.isize])?;
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("bgzf worker panicked"))
            .collect::<io::Result<Vec<()>>>()
    });
    result?;
    Ok(out)
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
}
