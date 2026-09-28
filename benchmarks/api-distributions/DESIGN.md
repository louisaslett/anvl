# Historical measurements and design notes

These observations were retained from the earlier README during the documentation
review on 2026-09-28. Their original measurement dates, commits and dependency
versions were not recorded here. They were not rerun during that review and do
not describe a guaranteed current result. Use fresh measurements for planning.

## Sweep timings

The earlier README reported the following M1 laptop timings over 160 anvl cells.
The full-depth figures were extrapolated, not measured.

| depth | serial | eight workers |
|---|---|---|
| smoke | 35 s | ~17 s |
| quick | ~1 h | ~8 min |
| full (extrapolated) | ~5 days | ~15 h |

The previous four-hour estimate for `nv_qnorm` was attached to an incorrect
20-cell count; the current grid has 32 cells. Time the current workload before
estimating a full run. Startup, compilation, scoring and reference validation
also affect elapsed time.

## Retained inputs

A previous `nv_qunif` f32 run's global worst 1,000 inputs clustered near 1/3.
Keeping worst inputs per sign and binade preserves evidence from other ranges.
An earlier smoke-store comparison reported 19 MB versus 51 MB, about 2.7 times
the size. This is a historical size comparison, not a current storage estimate.

## Browser export measurements

The following measurements describe an older seven-file export, before the
current table inventory. Library sizes and reader compatibility are historical
observations and should be checked for the versions used by a new consumer.

Measured over HTTP, on the full grid at smoke depth (53.7 MB in 7 files):

| operation | fetched | time |
|---|---|---|
| manifest + summary (the entire index and overview) | **31.9 kB** | 16 ms |
| drill into one cell of a 46 MB `detail.parquet` | **0.56 MB — 1.2%** | 111 ms |
| one column across all 1.28 M rows | 0.51 MB — 1.1% | 166 ms |

Aligning row groups costs about 23% in total size — smaller groups compress less
well — and buys roughly 80× less data transferred per drill-down.

**Verified readable from a browser.** `hyparquet` (pure JS, ~10 kB, no WASM)
reads nanoparquet's output bit-exactly: `+0`/`-0` distinguished, `NaN`, `±Inf`,
min subnormal, max double, `NA`→null, SNAPPY decompressed, HTTP range requests
via `asyncBufferFromUrl`. DuckDB-WASM was rejected: at ~30 MB it is larger than
the data it would query.

## Reference evaluation

An earlier 256-bit MPFR check reported errors within eight f64 ulps on roughly
8,700 inputs per output for the normal-family gradients. The current specs
instead declare a 16-ulp validation bound. The earlier observed maximum is not
a current validation verdict or proof of a global bound; consult the store's
validation records for the reference identity being used.

### Why the inverse Mills ratio needs care

The first
version of `pnorm.R` used `exp(log φ − log Φ)` for the log-scale gradient.
Both logs are ≈ −5e299 at q = −1e150 and their difference is only ≈ 345, far
below the ulp, so the reference returned 1 where the true value is 1e150 and
the sweep reported nv_pnorm as wrong by a factor of 1e231. anvl was right.
See `inv_mills()` in `sweeps/_normal.R`.
