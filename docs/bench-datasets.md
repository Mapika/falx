# Sustained Benchmark Datasets

Large local datasets live outside the repo under `/mnt/data/falx-bench`.
Regenerate the current pack with:

```bash
cargo run --release --example make_datasets -- \
  --out /mnt/data/falx-bench \
  --size 1g \
  --formats all
```

This creates one approximately 1 GiB file per generated format:

- `/mnt/data/falx-bench/csv-1g.csv`
- `/mnt/data/falx-bench/csv-geo-1g.csv`
- `/mnt/data/falx-bench/csv-hash-1g.csv` (CSV + `#` comments + quotes — exercises the `Regions` resolver; the scoreboard compares it vs the `csv` crate's `comment(Some(b'#'))` reader)
- `/mnt/data/falx-bench/tsv-1g.tsv`
- `/mnt/data/falx-bench/logfmt-1g.logfmt`
- `/mnt/data/falx-bench/ndjson-1g.ndjson`
- `/mnt/data/falx-bench/vcf-1g.vcf`
- `/mnt/data/falx-bench/fastq-1g.fastq`

Current file-backed benchmark entry points:

```bash
cargo run --release --example bench_sustained -- --formats all --runs 3 --warmup 1
cargo run --release --example bench_real -- /mnt/data/falx-bench/csv-1g.csv
cargo run --release --example bench_columns -- /mnt/data/falx-bench/csv-geo-1g.csv
cargo run --release --example json_parity -- /mnt/data/falx-bench/ndjson-1g.ndjson
```

`json_parity` is a same-work JSON comparison vs simd-json: both sum every
integer in each NDJSON document and must agree on the total (so it is a real
like-for-like query, not framing-vs-parsing). It contrasts falx's recursive
`Node`/`Items` navigation against a flat O(tape) scan — for an aggregate that
needs no hierarchy, the flat scan over `Nested::tape()` is ~2.3x faster and puts
falx ~3.5x ahead of simd-json on the identical query.

`bench_sustained` is the comparable-library scoreboard. Fair rows must produce
the same `Work` counters before timings are reported: record/pair count,
primary bytes, secondary bytes, and checksum. This prevents comparing a falx
framer with another crate's semantic parser and treating the numbers as
equivalent.

Current proper comparison crates:

- `csv` for CSV and TSV record parsing
- `arrow-csv` for typed CSV projection
- `logfmt-zerocopy` for logfmt key/value parsing
- `seq_io` and `needletail` for FASTQ parsing
- `noodles-vcf` for VCF typed record parsing
- `serde_json` and `simd-json` as full JSON parser baselines

The falx FASTQ row uses `falx::kernels::fastq::parse_fastq`, a generated
domain API backed by the synthesized/egraph-optimized newline kernel. The
generated sink validates four-line records, `@` headers, `+` separators, and
matching sequence/quality lengths while accumulating the benchmark counters.

NDJSON line framing is reported separately from full JSON parser baselines until
falx exposes equivalent full JSON semantic parsing.
