# falx-genomics

Python bindings for falx's fused `.vcf.gz` → Apache Arrow pipeline.

A bgzipped VCF is decompressed and parsed straight into typed Arrow columns in
one **block-parallel pass** — the bgzf blocks are inflated and parsed per worker
while hot in cache, with no full-file decompressed buffer materialized — and
returned as a `pyarrow.RecordBatch` that pandas / polars / DuckDB consume
zero-copy.

## Build

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin pyarrow
cd python
maturin develop --release          # builds the extension into the active venv
```

## Use

```python
import falx_genomics

rb = falx_genomics.read_vcf_gz_columns("clinvar.vcf.gz")   # -> pyarrow.RecordBatch
rb.num_rows                                                # 4_436_216

import polars as pl
df = pl.from_arrow(rb)                                     # zero-copy

import pyarrow as pa
pdf = pa.Table.from_batches([rb]).to_pandas()              # zero-copy
```

Columns: `pos` (int64, non-null), `quality` / `dp` / `af` (nullable — null where
the field or INFO key is absent). Pass `threads=N` to cap parallelism
(`threads=0`, the default, uses all cores). Parsing runs with the GIL released.

## Performance

On real ClinVar GRCh38 (183 MiB `.vcf.gz`, 4.4M records, 64-core box): ~0.5 s to
a `pyarrow.RecordBatch`. The underlying fused pipeline runs ~13–15× faster than a
`noodles`-based reader (the substrate behind oxbow / exon / biobear / polars-bio)
and is validated byte-for-byte against an independent scalar reference. See the
[falx](https://github.com/Mapika/falx) repo for the kernels, benchmarks, and the
`bgzf::parse_gz_par` fusion driver this wraps.

## Status

First binding: numeric VCF columns (`POS`, `QUAL`, `DP`, `AF`). REF/ALT strings,
arbitrary spec'd INFO keys, and FASTQ are natural extensions.
