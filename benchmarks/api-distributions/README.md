# `api-distributions` — bit-pattern sweeps of the distribution API

Every function in `R/api-distributions.R` is swept against base R over the
float bit patterns themselves, in both precisions, for values and for every
derivative.

This is an accuracy benchmark and nothing more. It measures, and it shows you
what it measured. It does not decide whether a result is acceptable: that
depends on the function, the precision and what the caller needs, and it
belongs to the person reading.

This directory holds **sweeps only**. The algorithm-comparison reports for
candidate implementations live alongside it in `../nv_pnorm/` and `../nv_qnorm/`
and are a different kind of artifact; nothing here touches them.

```
run.R          list · run · report · browse · diff · export · status · selftest · merge
query.R        reading the store, from R or the shell
R/util.R       bit patterns, ulp spacing, number formatting
R/engine.R     enumeration, scoring, reducers, region classification
R/cells.R      the spec DSL, the grid, cell ids, filtering
R/render.R     how a set of results is drawn on a terminal
R/store.R      the write-once Parquet store
R/provenance.R the environment fingerprint recorded with every run
sweeps/        one small file per function — this is what you add to
HPC.md         notes on running this at scale on a cluster
```

## The loop

```bash
Rscript run.R run --depth smoke     # 1. measure   (~35 s for everything)
Rscript run.R status                # 2. the index: what ran, what differs most
Rscript run.R browse                # 3. drill down and read
Rscript run.R diff                  # 4. after a code change: what moved?
```

`Rscript run.R selftest` (~10 s) proves the engine still detects errors; run it
after touching anything in `R/`.

Nothing is written into this directory. Results go to the store (below).

## The three depths

A sweep walks a pattern-index space of 2³¹ values per sign with a stride, so a
shallower depth is the *same* sweep at coarser spacing — it still covers the
whole number line, subnormals, infinities and NaN included, and can fail
anywhere the full run can. Only the resolution changes.

Measured on an M1 laptop over the 160 anvl cells. Cost is linear in the sample
count, so the `full` row is extrapolated from the two that were timed:

| depth | samples per cell | whole grid, serial | with `--jobs 8` | for |
|---|---|---|---|---|
| `smoke` | 524,288 | **35 s** | ~17 s | every code change; the default |
| `quick` | 33.5 M | ~1 h | ~8 min | before opening a PR |
| `full` | 4.29 × 10⁹ | ~5 days | ~15 h | overnight per function, or a cluster |

In `f32`, `full` is genuinely exhaustive: every one of the 2³² float32 values is
visited exactly once. In `f64`, 2⁶⁴ is out of the question, so `full` takes one
sample from each of the 2³² contiguous blocks of 2³² — every (sign, exponent,
top-20-mantissa-bit) combination, with the low 32 bits drawn from a fixed seed
so another machine reproduces the same inputs.

`full` over the whole grid is not a laptop job — that is what `HPC.md` is for.
It is entirely reasonable for *one function*: `--filter spec=nv_qnorm` is 20
cells, about 4 hours at `--jobs 8`.

## Selecting what to run

```bash
Rscript run.R list                                    # what cells exist
Rscript run.R run --filter spec=nv_qnorm --depth full
Rscript run.R run --filter 'dtype=f64,kind=grad'
Rscript run.R run --filter 'spec=nv_punif|nv_qunif' --depth quick --jobs 8
Rscript run.R run --backends anvl,jax                 # JAX side too
Rscript run.R run --dry-run --depth full              # what would run
```

`--filter` takes `key=value` terms joined by commas; `|` inside a value means
"any of". Keys are `spec`, `family`, `backend`, `dtype`, `kind`, `param_set`,
`flags`, `cell_id` and `output`. An unknown key is an error.

`--jobs N` forks N workers. Cells are independent and each writes its own file,
so this needs no coordination and no locking.

## `browse` — drill down interactively

320 results is far too much to print, so `browse` and `report` summarise until
you have narrowed things enough for the pages to be readable. Both show f32 and
f64 **side by side at every level** rather than making you pick: their errors
differ by about nine orders of magnitude, so a pooled summary has a meaningless
maximum, and comparing the two is the commonest thing to want.

```
  nv_qnorm  ›  value  ›  lower_tail=TRUE,log_p=TRUE   (4 results)
────────────────────────────────────────────────────────────────────────
                 --------- f32 --------- --------- f64 ---------
  parameter set   worst rel unexplained   worst rel unexplained
  shifted          8.16e-04           1    2.96e-08           1
  standard         8.94e-06           1    1.63e-08           1

    1) shifted
    2) standard

   b) back   q) quit
```

Pick a number to go deeper, `b` to go back, `q` to leave. Five keystrokes takes
you from 320 results to one page. The axes narrow in the order
`function → backend → value/gradient → flags → parameter set → precision`, and
the last step prints the page itself. At a page, `d` walks through the retained
worst inputs, 20 at a time, with their bit patterns.

### The worst inputs are kept per binade, not globally

The store keeps the worst **10 inputs in each binade**. A single global list
clusters: every one of nv_qunif f32's worst 1000 sat around x = 1/3, so the
worst input anywhere in the tail had been computed and thrown away. Per binade,
roughly 250 binades are occupied and ~2,500 inputs kept per result, spanning
|x| from 1e-41 to 1 — so *"what is the worst input near 1e-8?"* is answerable.
The overall worst is still just the first row, because retained inputs are
ranked globally on the way out.

K is small because the cost multiplies by the number of occupied binades rather
than dividing by it: per-binade top-100 would be about twelve times the store,
per-binade top-10 is about five (19 MB → 51 MB for the whole grid at smoke).
`--topk N` changes it.

Each row carries its `binade` and `sign`, so the detail joins directly onto the
behaviour bands below.

Numbered menus rather than arrow keys is deliberate: R has no usable TUI
library, and raw-mode key capture means driving `stty`, which is brittle across
terminals and breaks as soon as output is piped.

## `report` — the same pages, non-interactively

`report` walks the same tree, but the level is set by how narrow your filter
already is, so it composes with scripting and needs no terminal:

```bash
Rscript run.R report                                   # summary by function
Rscript run.R report --filter spec=nv_qnorm            # summary by value/gradient
Rscript run.R report --filter 'spec=nv_qnorm,output=p' # summary by flags
Rscript run.R report --filter 'cell_id=nv_qnorm/anvl/f64/value/standard/lower_tail=TRUE,log_p=FALSE'
```

Each summary ends by printing the exact command to drill into whichever branch
looks worst. `--pages N` sets how many full pages it prints before summarising
instead (default 4).

A page carries the sample counts, the full distribution of relative error,
every region where the two sides disagree with no finite error, and the worst
individual inputs with their bit patterns:

```
nv_qunif  ·  uniform quantile function against base R qunif()
f32 value · wide (min = -3.1415926535897931, max = 6.2831853071795862) · lower tail
────────────────────────────────────────────────────────────────────────
swept   524,288 samples — 1 in every 8,192 float32 values

AGREEMENT WITH BASE R
  identical                         393,215   75.00%
  differ, finite error              130,050   24.81%
  differ, no finite error             1,023    0.20%   (see REGIONS below)

RELATIVE ERROR of the 130,050 that differ
  1e-9 ..1e-8    █                              5,967
  1e-8 ..1e-7    ██████████████████████████████ 122,627
  1e-7 ..1e-6    █                                745
  worst   1.04e-04 relative  (1365 ulp)  at x = 0.333251953125
          0x3EAAA000
          anvl    -0.00076706986874341965
          base R  -0.00076699039394282058
```

That histogram is the thing a top-K list can never show: the spike at
1e-8..1e-7 is the f32 rounding floor, so the bulk of those 130,050 "differences"
are one ulp of f32 and not a finding at all.

Each page also carries a structural map of **where along the number line the
behaviour changes** — a different question from how big the errors are:

```
BEHAVIOUR ACROSS THE NUMBER LINE
  range                        behaviour               ident.  differ no fin.
  -3.40e+38 .. -1.18e-38       all identical             100%      0%      0%
  -1.18e-38 .. 0               mixed, some non-finite      0%     <1%    >99%
  0 .. 1                       all differ                  0%    100%      0%
  1 .. 2                       identical + differ        >99%     <1%      0%
  2 .. 3.40e+38                all identical             100%      0%      0%
  +Inf and NaN                 all identical             100%      0%      0%
```

Counts accumulate **per binade**, then adjacent binades that behave alike are
merged, so a boundary in that table is a place where behaviour actually
changes. Per-bit-pattern runs would be useless: at f32 epsilon, identical and
differing values interleave constantly, so a well-behaved cell would produce
hundreds of thousands of alternating one-element runs. The highest exponent
field gets its own row rather than a range, because it is not an interval — it
holds ±Inf and every NaN side by side.

Reports are generated from the store on demand and never written to disk.
Regenerating is instant, and a file would only be a stale copy of queryable
data — which is exactly how the previous generation of `.txt` artifacts became
unusable.

## `diff` — did the change help?

```bash
Rscript run.R diff                       # newest run vs the previous result per cell
Rscript run.R diff --from <run> --to <run>
Rscript run.R diff --filter spec=nv_qnorm
```

```
  selftest  d/dscale f64  clean/broken
    worst rel err        1.00e-06 -> 0.001   (1e+03x worse)
    worst ulp            4.86e+09 -> 4.89e+12

SUMMARY
     4 regressed
     0 improved
    20 unchanged (bit-identical to the earlier run)
     0 had no earlier result to compare against
```

**The sweep is deterministic** — fixed seed, fixed stride, the same inputs every
time — so two runs of the same cell at the same depth on one machine are
bit-identical unless the code changed. That removes any need for a "meaningfully
different" threshold: every difference is real, and exact equality is a real *no
change*. (Verified: re-running unchanged code reports 8 of 8 unchanged.)

Results pair by cell, output, platform **and depth**. A smoke result and a full
result sample different inputs, so pairing across depths would report the extra
coverage as a regression; an unpaired result is reported as having nothing to
compare against instead.

## Reading what a disagreement means

**A disagreement does not mean anvl is wrong.** base R is the reference, not the
truth, and it is sometimes the weaker implementation. At `x = 5.551e-17`:

```r
punif(x, 0, 1, lower.tail = FALSE, log.p = TRUE)         # base R:  0
as.double(nv_punif(nv_array(x, dtype = "f64"), 0, 1,
                   lower_tail = FALSE, log_p = TRUE))    # anvl: -5.551e-17
log1p(-x)                                                # correct: -5.551e-17
```

base R forms `1 - x` first, which rounds to exactly 1, and loses the value
entirely; anvl goes through `log1p` and keeps it. The sweep correctly reports a
disagreement, and the right response is to leave `nv_punif` alone. Always
establish which side is right before acting on one.

### Regions with no finite error are classified, not just counted

When the two sides disagree with no meaningful denominator — one is NaN, or the
reference is zero or infinite — the sample carries no relative error, and these
arrive in huge contiguous blocks. The sweep collapses them into intervals and
labels each:

| class | meaning |
|---|---|
| `nan` | the whole interval is NaN (and ±Inf), where disagreement is a convention |
| `below_support` / `above_support` | outside the spec's declared support |
| `subnormal` | wholly below the smallest normal — see below |
| `unclassified` | nothing in the spec accounts for this |

This is measurement, not judgement: it says what the numbers are. Without it
every quantile cell would report billions of "failures" that are simply `p`
outside [0, 1].

### A known platform property: subnormal flush-to-zero

Every arithmetic and comparison operation on this PJRT/XLA CPU backend flushes
subnormals to zero, in **both** precisions. Storage round-trips correctly, but
`x >= 0` is `TRUE` for a negative subnormal, because the comparison sees `-0`:

```r
as.vector(nv_array(-1e-39, dtype = "f32") >= 0)             # TRUE  (R says FALSE)
as.double(nv_dunif(nv_array(-1e-39, dtype = "f32"), 0, 1))  # 1     (R says 0)
```

Nothing measured entirely inside that band describes the function under test,
so such intervals are classified `subnormal`. They are still recorded —
`Rscript query.R ranges --class subnormal` lists every one — and the day the
backend stops flushing, they become ordinary agreement.

## `export` — publishing a snapshot

The store accumulates: every run appends, and queries take the latest per cell.
A published artifact must instead be one coherent snapshot — **one artifact per
(anvl version, platform, backend)**, covering every function.

```bash
Rscript run.R export --out ../anvl-bench-darwin-arm64-cpu
```

```
manifest.json     index: schema version, platform, specs, depths, row counts
runs.parquet      the environment fingerprint of every run included
summary.parquet   the results table for every cell of every function
detail.parquet    the worst inputs, per binade
bands.parquet     the per-binade profile (unmerged; merged on render)
hist.parquet      the error distribution
ranges.parquet    the no-finite-error regions
```

One file per **table**, not per function. The overview page summarises every
function at once, so splitting by function would mean fetching all the pieces
anyway in more requests, and would turn "look at another platform" into many
downloads instead of one artifact.

`manifest.json` is deliberately JSON and carries no measurements: it is the
index, readable without a Parquet reader. The measurements stay in Parquet
because JSON cannot represent `NaN`, `±Inf` or `-0`, which is most of what makes
this data interesting.

### Row groups are aligned to cells, so drilling in is cheap

Parquet can only skip whole row groups, and nanoparquet writes **one** group of
up to ten million rows by default — which would mean reading a whole 46 MB file
to inspect a single cell. `export` sorts `detail` and `bands` by cell and starts
a new row group at every cell boundary.

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

## The store

Results are **Parquet, written once, never mutated** — one file per cell per
run, in a partitioned directory.

That is the whole concurrency design. Two processes never touch the same bytes,
so parallel jobs, a second machine and a cluster array are all the same case,
and merging two machines' results is a file copy. It is also why neither SQLite
nor DuckDB is the write target: SQLite's locking is documented-unreliable on NFS
and Lustre, and DuckDB allows only one writer process per database file. DuckDB
is excellent on the read side, and `sw_sql()` uses it when installed — but
nothing depends on it.

Parquet earns its place on **type fidelity**, not size: the store is small, but
it holds NaN, ±Inf, signed zero and subnormals exactly, all of which a CSV
round-trip would quietly destroy. (Verified: `nanoparquet` round-trips f64
bit-exactly. It does *not* round-trip 64-bit integers — they come back widened
to doubles — which is why bit patterns are stored as hex strings.)

```bash
export NV_SWEEP_STORE=/path/to/store    # default: R_user_dir("anvl-sweeps","cache")
Rscript run.R merge --from /other/machine/store
```

The store lives **outside the repo** and is never committed: it is
machine-specific, it is regenerated by re-running the sweeps, and it grows
without bound as runs accumulate.

Results **accumulate**; a re-run adds rows rather than replacing them. Queries
default to the latest per (cell, output, platform, depth), so history is free:
"which results predate this commit" is a column, not a file timestamp.

Every run records an environment fingerprint — host, CPU, OS, architecture,
device, R version, `default_dtypes()`, and the git SHA of each sibling checkout
(read straight out of `.git`, so no `git` command is ever invoked and it works
on a node without git installed). Without this a result from August cannot be
compared with one from September, which is exactly how the previous generation
of artifacts became unusable.

## Reading the store directly

```bash
Rscript query.R worst --spec nv_qnorm --n 20
Rscript query.R ranges --class unclassified
Rscript query.R detail --cell 'nv_qnorm/anvl/f64/value/standard/lower_tail=TRUE,log_p=FALSE'
Rscript query.R runs
```

or in R:

```r
source("query.R")
sw_worst(spec = "nv_pnorm", metric = "ulp")
sw_detail("nv_punif/anvl/f64/grad/wide/lower_tail=TRUE,log_p=TRUE", output = "min")
sw_hist("nv_pnorm/anvl/f64/value/standard/lower_tail=TRUE,log_p=FALSE")
sw_compare("darwin-arm64-cpu", "linux-x86_64-cuda")
sw_sql("select spec, dtype, max(worst_rel_err) from results group by 1, 2")
```

`sw_detail()` gives the worst individual inputs **with their bit patterns** —
paste a `bits` value straight into a regression test in
`tests/testthat/test-api-distributions.R`.

## Adding a function

Write one file in `sweeps/`. It returns a `sweep_spec()` and is discovered
automatically; nothing is registered anywhere. Files beginning with `_` are
shared helpers, not specs.

```r
source(file.path(here(), "sweeps", "_normal.R"), local = TRUE)

sweep_spec(
  name      = "nv_dnorm",
  family    = "normal",
  primary   = "x",
  params    = list(standard = list(mean = 0, sd = 1)),
  flags     = list(log = c(FALSE, TRUE)),
  support   = function(p, f) c(-Inf, Inf),

  value     = function(x, dtype, p, f) as.double(anvl::nv_dnorm(...)),
  ref_value = function(x, p, f) dnorm(x, p$mean, p$sd, log = f$log),

  grad_wrt  = c("x", "mean", "sd"),
  grad      = function(x, dtype, p, f) list(x = ..., mean = ..., sd = ...),
  ref_grad  = function(x, p, f)       list(x = ..., mean = ..., sd = ...),

  jax_value = function(x, dtype, p, f) ...,          # optional
  jax_grad  = function(x, dtype, p, f) ...,          # optional
  jax_covers = function(f, kind) !isTRUE(f$log_p)    # where JAX has no twin
)
```

The grid is `params × flags × dtypes × {value, grad} × backends`; an unknown
field name is an error, so a typo in a config that takes hours to run cannot
silently do nothing.

**`grad` returns every derivative from one call.** That is not a convenience: a
reverse pass computes them all together, so scoring them from a single sweep
rather than re-sweeping once per argument is a straight 3× saving on what is by
far the largest part of the grid.

Three things worth knowing before you write a reference:

1. **A wrong reference is indistinguishable from a real finding.** The first
   version of `pnorm.R` used `exp(log φ − log Φ)` for the log-scale gradient.
   Both logs are ≈ −5e299 at q = −1e150 and their difference is only ≈ 345, far
   below the ulp, so the reference returned 1 where the true value is 1e150 and
   the sweep reported nv_pnorm as wrong by a factor of 1e231. anvl was right.
   See `inv_mills()` in `sweeps/_normal.R`.
2. **Declare the support honestly.** It is what explains whole regions, and an
   over-wide support turns every out-of-domain block into a false finding while
   an over-narrow one hides real ones.
3. **Prose belongs in the spec.** The derivation of an analytic derivative and
   the reason for a parameter set are the most valuable things in these files.
   Keep them next to the code they justify.

Then check the engine still behaves:

```bash
Rscript run.R selftest
```

`sweeps/_selftest.R` is a synthetic family with errors injected at known places
— a one-ulp nudge on a known interval, a zeroed tail, a 1e-6 gradient error —
and `selftest` asserts the engine finds each at the right magnitude. It exists
because a sweep that silently sweeps nothing looks exactly like a sweep that
found nothing, and at full depth that is an expensive way to discover a
misspelled filter.

## Dependencies

`nanoparquet` and `cli` are required; `reticulate` only for `--backends jax`;
`duckdb` only for `sw_sql()`. The JAX side uses the shared virtualenv at
`../../../py-benchmarks/.venv` unless `RETICULATE_PYTHON` is already set.

Never run `devtools::install()` in this ecosystem — it resolves anvl's
`Remotes:` and upgrades the sibling packages from GitHub main. Use
`R CMD INSTALL <dir>`.
