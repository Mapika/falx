//! Declarative `lines_per_record`: the generated `fastq` kernel
//! (lines_per_record = 4) groups every four newline-terminated lines into one
//! record, exposing the four lines as its fields. Differential against a
//! reference that groups newline positions by four — the hand-written framing
//! in examples/fastq.rs (`for_each_read`). Quality lines carry `@`/`+` on
//! purpose: a sigil split would mis-frame, line counting must not.

use falx::kernels::fastq as k;

mod common;
use common::Rng;

/// Reference grouping: every four newline positions are one record; a trailing
/// partial group (incl. a final unterminated line) is dropped. Returns each
/// record's full span and its four line slices.
fn reference(data: &[u8]) -> Vec<(&[u8], [&[u8]; 4])> {
    let nls: Vec<usize> = data
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b == b'\n')
        .map(|(i, _)| i)
        .collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    for g in nls.chunks_exact(4) {
        out.push((
            &data[start..g[3]],
            [
                &data[start..g[0]],
                &data[g[0] + 1..g[1]],
                &data[g[1] + 1..g[2]],
                &data[g[2] + 1..g[3]],
            ],
        ));
        start = g[3] + 1;
    }
    out
}

fn check(data: &[u8]) {
    let want = reference(data);
    let parsed = k::parse(data);
    let recs: Vec<_> = parsed.records().collect();
    let ctx = || String::from_utf8_lossy(data).into_owned();
    assert_eq!(recs.len(), want.len(), "record count on {:?}", ctx());
    assert_eq!(parsed.terminated_record_count(), want.len());
    for (rec, (full, lines)) in recs.iter().zip(&want) {
        assert_eq!(rec.field_count(), 4, "field_count on {:?}", ctx());
        for (i, line) in lines.iter().enumerate() {
            assert_eq!(rec.field_raw(i).unwrap(), *line, "line {i} on {:?}", ctx());
        }
        assert!(rec.field_raw(4).is_none(), "out-of-range field");
        // Iterators agree with indexed access.
        let raw: Vec<&[u8]> = rec.fields_raw().collect();
        assert_eq!(raw, lines.to_vec(), "fields_raw on {:?}", ctx());
        let cleaned: Vec<Vec<u8>> = rec.fields().map(|c| c.into_owned()).collect();
        assert_eq!(
            cleaned,
            lines.iter().map(|l| l.to_vec()).collect::<Vec<_>>(),
            "fields on {:?}",
            ctx()
        );
        // The record spans all four lines, final terminator excluded.
        assert_eq!(rec.as_bytes(), *full, "as_bytes on {:?}", ctx());
    }
}

#[test]
fn hand_picked() {
    // Quality line contains '@' and '+': a sigil split would mis-frame here.
    check(b"@read1 info\nACGTACGT\n+\n!''*@+#$%\n@read2\nTTTT\n+\nIIII\n");
    // Line count not a multiple of 4: the trailing 2-line partial is dropped.
    check(b"@a\nAAAA\n+\nIIII\n@b\nCCCC\n");
    // Final line unterminated (no trailing '\n'): trailing partial dropped.
    check(b"@a\nAAAA\n+\nIIII\n@b\nGGGG\n+\nno-newline-here");
    // Empty lines within a record are preserved as empty fields.
    check(b"@a\n\n+\n\n");
    check(b"");
    check(b"@x\nACGT\n+\nIIII\n");
}

#[test]
fn randomized() {
    let mut rng = Rng(0xF00D_BEEF_1234_5678);
    let bases = b"ACGT";
    let quals = b"!#@+ABCDEFG0123456789"; // includes '@' and '+'
    for _ in 0..3000 {
        let mut data = Vec::new();
        let reads = rng.next() % 20;
        for r in 0..reads {
            let len = (rng.next() % 40) as usize;
            data.extend_from_slice(b"@read");
            data.extend_from_slice(r.to_string().as_bytes());
            data.push(b'\n');
            for _ in 0..len {
                data.push(bases[(rng.next() % 4) as usize]);
            }
            data.push(b'\n');
            data.extend_from_slice(b"+\n");
            for _ in 0..len {
                data.push(quals[(rng.next() % quals.len() as u64) as usize]);
            }
            data.push(b'\n');
        }
        // Sometimes append a stray partial group (1-3 extra lines).
        for _ in 0..(rng.next() % 4) {
            data.extend_from_slice(b"partial\n");
        }
        // Sometimes drop the final newline (unterminated last line).
        if rng.next() & 1 == 0 && data.last() == Some(&b'\n') {
            data.pop();
        }
        check(&data);
    }
}
