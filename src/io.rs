//! Memory-mapped file input for the parallel kernels (`mmap` feature).
//!
//! Handing an mmap'd slice straight to a parallel `parse_*_par` kernel lets the
//! page faults fault in *across the parse threads* and overlap with compute, so
//! file I/O is effectively free relative to a serial `std::fs::read` copy. On a
//! 1 GiB CSV this is the difference between ~1.2 GiB/s (serial read + parse) and
//! ~2.75 GiB/s (mmap + parse) end-to-end — competitive with DuckDB/polars, whose
//! CSV readers mmap for the same reason.
//!
//! Measured note: plain mmap beats `MAP_POPULATE` here — a single-threaded kernel
//! prefault is slower than letting the 24 parse threads fault their own regions;
//! `madvise(SEQUENTIAL)` was a no-op. So this module exposes the plain path.

use std::fs::File;
use std::io;
use std::path::Path;

pub use memmap2::Mmap;

/// Memory-map a file read-only for parallel parsing.
///
/// Hand the result (which derefs to `&[u8]`) to a `parse_*_par` kernel; the page
/// faults parallelize across the parse threads. The mapping must outlive any
/// borrowed parse output (e.g. a `Columns<'_>` that points into it).
///
/// # Example
/// ```no_run
/// # #[cfg(feature = "mmap")] {
/// let m = falx::io::map("cities.csv")?;
/// let cols = falx::kernels::csv_geo::parse_columns_par(&m, 24);
/// let total: f64 = cols.latitude.iter().sum();
/// # let _ = total;
/// # }
/// # Ok::<(), std::io::Error>(())
/// ```
///
/// # Safety
/// Memory-mapping is unsafe at the OS level: if another process truncates or
/// mutates the file while the map is live, reads can fault or observe torn data.
/// Callers are responsible for not mutating the file for the lifetime of the map.
pub fn map<P: AsRef<Path>>(path: P) -> io::Result<Mmap> {
    let file = File::open(path)?;
    // SAFETY: see the function-level contract — the file must not be mutated or
    // truncated while the returned map is alive.
    unsafe { Mmap::map(&file) }
}

/// Map a file and run `f` over its bytes, keeping the mapping alive for the call.
///
/// Convenience for the common case where the parse output borrows the mapped
/// bytes: do the parse *and* reduce it to an owned value inside `f`, so nothing
/// outlives the map.
///
/// # Example
/// ```no_run
/// # #[cfg(feature = "mmap")] {
/// let sum_lat = falx::io::with_mapped("cities.csv", |bytes| {
///     falx::kernels::csv_geo::parse_columns_par(bytes, 24)
///         .latitude
///         .iter()
///         .sum::<f64>()
/// })?;
/// # let _ = sum_lat;
/// # }
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn with_mapped<P: AsRef<Path>, R>(path: P, f: impl FnOnce(&[u8]) -> R) -> io::Result<R> {
    let m = map(path)?;
    Ok(f(&m))
}
