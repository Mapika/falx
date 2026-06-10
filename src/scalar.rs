//! Scalar reference kernel.
//!
//! Deliberately the dumbest possible implementation: one byte at a time, one
//! boolean of state. The SIMD kernels are differential-tested against this,
//! so its job is to be obviously correct, not fast. Quote handling is parity
//! (toggle on every `"`), which matches RFC 4180: an escaped quote `""`
//! toggles twice and lands back in the same state.

/// Fully general reference: arbitrary structural byte set, optional quote,
/// optional escape byte.
///
/// Escape semantics mirror the bitstream graphs: a byte preceded by an
/// odd-length run of the escape byte is escaped, and being escaped only
/// affects whether a quote toggles the in-quotes state — structural bytes
/// are suppressed by quoting, never by escaping directly.
pub fn index_structurals_spec(
    data: &[u8],
    structural: &[u8],
    quote: Option<u8>,
    escape: Option<u8>,
    out: &mut Vec<u32>,
) {
    let mut in_quotes = false;
    let mut escape_run = 0usize;
    for (i, &byte) in data.iter().enumerate() {
        let escaped = escape_run % 2 == 1;
        if escape == Some(byte) {
            escape_run += 1;
        } else {
            escape_run = 0;
        }
        if quote == Some(byte) {
            if !escaped {
                in_quotes = !in_quotes;
            }
        } else if !in_quotes && structural.contains(&byte) {
            out.push(i as u32);
        }
    }
}

/// Like [`index_structurals`] but for an arbitrary delimiter and quote byte.
pub fn index_structurals_dialect(data: &[u8], delimiter: u8, quote: u8, out: &mut Vec<u32>) {
    index_structurals_spec(data, &[delimiter, b'\n'], Some(quote), None, out);
}

/// Append the byte offsets of all unquoted `,` and `\n` in `data` to `out`.
pub fn index_structurals(data: &[u8], out: &mut Vec<u32>) {
    index_structurals_dialect(data, b',', b'"', out);
}
