//! Scalar reference kernel.
//!
//! Deliberately the dumbest possible implementation: one byte at a time, one
//! boolean of state. The SIMD kernels are differential-tested against this,
//! so its job is to be obviously correct, not fast. Quote handling is parity
//! (toggle on every `"`), which matches RFC 4180: an escaped quote `""`
//! toggles twice and lands back in the same state.

/// Fully general reference: arbitrary structural byte set, optional quote,
/// optional escape byte, optional line-start comment byte.
///
/// Escape semantics mirror the bitstream graphs: a byte preceded by an
/// odd-length run of the escape byte is escaped, and being escaped only
/// affects whether a quote toggles the in-quotes state — structural bytes
/// are suppressed by quoting, never by escaping directly.
///
/// Comment semantics: the comment byte at line start, outside quotes,
/// makes everything through the next newline inert — except the newline
/// itself, which stays structural (the comment line remains a record
/// boundary; walkers skip such records). Quote bytes inside comments are
/// inert; comment bytes inside quoted fields (including after a *quoted*
/// newline) never open a comment.
pub fn index_structurals_spec(
    data: &[u8],
    structural: &[u8],
    quote: Option<u8>,
    escape: Option<u8>,
    comment: Option<u8>,
    out: &mut Vec<u32>,
) {
    let mut in_quotes = false;
    let mut in_comment = false;
    let mut escape_run = 0usize;
    let mut line_start = true;
    for (i, &byte) in data.iter().enumerate() {
        let escaped = escape_run % 2 == 1;
        if escape == Some(byte) {
            escape_run += 1;
        } else {
            escape_run = 0;
        }
        if in_comment {
            if byte == b'\n' {
                in_comment = false;
                if structural.contains(&b'\n') {
                    out.push(i as u32);
                }
            }
        } else if quote == Some(byte) {
            if !escaped {
                in_quotes = !in_quotes;
            }
        } else if !in_quotes && comment == Some(byte) && line_start {
            in_comment = true;
        } else if !in_quotes && structural.contains(&byte) {
            out.push(i as u32);
        }
        line_start = byte == b'\n';
    }
}

/// Like [`index_structurals`] but for an arbitrary delimiter and quote byte.
pub fn index_structurals_dialect(data: &[u8], delimiter: u8, quote: u8, out: &mut Vec<u32>) {
    index_structurals_spec(data, &[delimiter, b'\n'], Some(quote), None, None, out);
}

/// Append the byte offsets of all unquoted `,` and `\n` in `data` to `out`.
pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {
    index_structurals_dialect(data, b',', b'"', out);
}
