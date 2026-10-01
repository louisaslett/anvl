# Historical measurements and design notes

These notes preserve earlier measurements. Their original dates, commits and
dependency versions were not recorded here, and they were not rerun for the
2026-10-01 documentation update. Missing details are listed below. Measure the
current workload before using these figures for planning.

## Sweep timings

| context | recorded details |
|---|---|
| Environment | M1 laptop; memory and runtime settings not recorded |
| Date and versions | not recorded |
| Workload | 160 anvl cells; worker counts shown below |
| Method | smoke and quick reported as timings; full extrapolated |

The earlier README reported:

| depth | serial | eight workers |
|---|---|---|
| smoke | 35 s | ~17 s |
| quick | ~1 h | ~8 min |
| full (extrapolated) | ~5 days | ~15 h |

A separate README estimate gave **about 36 days** for a full Normal and Uniform
sweep on one core of a “modern Intel CPU”. The CPU model, date, versions,
backend selection and whether this was measured or extrapolated were not
recorded. It cannot be compared reliably with the M1 figures.

Time the current workload before estimating a full run. Startup, compilation,
scoring and reference validation also affect elapsed time.

## Retained inputs

| context | recorded details |
|---|---|
| Environment, date and versions | not recorded |
| Workload | `nv_qunif` f32 retention example; smoke-store size comparison |
| Method | observed input clustering and reported store sizes |

A previous `nv_qunif` f32 run's global worst 1,000 inputs clustered near 1/3.
Keeping worst inputs per sign and binade preserves evidence from other ranges.
An earlier smoke-store comparison reported 19 MB versus 51 MB, about 2.7 times
the size. This is a historical size comparison, not a current storage estimate.

## Browser export measurements

The following measurements describe an older seven-file export, before the
current table inventory. Library sizes and reader compatibility are historical
observations and should be checked for the versions used by a new consumer.

| context | recorded details |
|---|---|
| Environment | HTTP delivery; browser, machine and network not recorded |
| Date and versions | not recorded |
| Workload | full grid at smoke depth; 53.7 MB in seven files |
| Method | measured transfer sizes and elapsed times |

Reported measurements:

| operation | fetched | time |
|---|---|---|
| manifest + summary (the entire index and overview) | **31.9 kB** | 16 ms |
| drill into one cell of a 46 MB `detail.parquet` | **0.56 MB — 1.2%** | 111 ms |
| one column across all 1.28 M rows | 0.51 MB — 1.1% | 166 ms |

In this measurement, aligning row groups increased total size by about 23%
and reduced data transferred per drill-down by roughly a factor of 80.

The earlier browser check reported that `hyparquet` (pure JS, ~10 kB, no WASM)
read nanoparquet's output with signed zeros, NaN, infinities, the smallest
subnormal and the largest double preserved; NA became null. It also handled
SNAPPY decompression and HTTP range requests via `asyncBufferFromUrl`. The
reported DuckDB-WASM size was about 30 MB, which led to choosing hyparquet.

## Reference evaluation

| context | recorded details |
|---|---|
| Environment | MPFR at 256 bits; machine not recorded |
| Date and versions | not recorded |
| Workload | roughly 8,700 inputs per normal-family gradient output |
| Method | sampled comparison; observed errors within eight f64 ulps |

An earlier 256-bit MPFR check reported errors within eight f64 ulps on roughly
8,700 inputs per output for the normal-family gradients. The current specs
instead declare a 16-ulp validation bound. The earlier observed maximum is not
a current validation verdict or proof of a global bound; consult the store's
validation records for the reference identity being used.

### Why the inverse Mills ratio needs care

The first version of `pnorm.R` used `exp(log φ − log Φ)` for the log-scale gradient.
Both logs are ≈ −5e299 at q = −1e150 and their difference is only ≈ 345, far
below the ulp, so the reference returned 1 where the true value is 1e150 and
the sweep reported nv_pnorm as wrong by a factor of 1e231. anvl was right.
See `inv_mills()` in `sweeps/_normal.R`.
