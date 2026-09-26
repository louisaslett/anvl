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

`Rscript run.R selftest` (~30 s) proves the engine still detects errors; run it
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

### Every failure has a tested cause, not a location

When the two sides disagree and there is no finite relative error — one is NaN,
the reference is zero or infinite, the result is ±∞ against a finite reference,
or the error itself overflows — the sample is a *failure*. Every such case is,
not a list of them. Failures arrive in contiguous blocks, which the sweep
collapses into regions, one **cause** per region, each established by a test
rather than read off where the inputs lie:

| cause | tested how | category |
|---|---|---|
| `nan_input` | the input is NaN | failure |
| `input_flushing` | a subnormal input whose result is **bit-identical** to the function's own result at the same-signed zero, **and** that zero result is **validated**: identical to base R's, or base R's correctly rounded | backend limitation |
| `flush_inherits_zero_error` | as above, but the zero result is not validated — *any* error at zero, finite or not: the subnormals inherit it | failure |
| `domain_boundary` | an endpoint of the valid input domain (e.g. p = 0, p = 1), or a subnormal inheriting the behaviour of a zero that is one | boundary |
| `outside_domain` | wholly outside the valid input domain | failure — or, for a gradient, see below |
| `zero_input` | ±0 that is not a domain endpoint | failure |
| `inf_input` | ±∞ inside the domain | failure |
| `unidentified` | none of the above | failure |

The **category** is what a reader weighs, and all four stay visible:

- **failure** — numerical or behavioural failure on valid inputs.
- **backend limitation** — the platform, not the function: input flushing.
- **boundary** — behaviour at a domain endpoint, which needs an explicit
  convention or limiting value; the reference supplies the limiting value.
- **undefined domain** — a *gradient* outside the valid input domain where the
  forward values on both sides are NaN, so no derivative is defined and the two
  sides differ only in convention (e.g. `qnorm` for p > 1: anvl's d/dp is NaN,
  the reference's is 0). A gradient cell never computes forward values, so this
  is **established** from the matching value cell, which swept the identical
  inputs: only where it found both values NaN for every out-of-domain sample in
  every binade the region touches. Anything less stays a failure. Only this
  category is set aside from an accuracy verdict, and setting it aside
  validates nothing.

**The evidence for a convention must come from the same run.** The value cell
is looked up by the gradient region's own `run_id`, which fixes the platform,
the anvl build, the harness, the depth and the seed together, so "the identical
inputs" holds by construction. Evidence from any other run is never used: a
gradient cell re-run on its own stays a failure, and its region's `evidence`
column says why. Exact points follow the same rule, matched by bit pattern.
Resolution always reads the **whole store** — `status`, `report`, `browse`,
`diff`, `query.R` and `export` all go through the same `resolved_ranges()` /
`resolved_points()` — so a filter can narrow what is *shown* but never what is
used as evidence: a gradient-only export classifies exactly as the terminal
does.

Zero is never a subnormal: it is the value a subnormal is flushed *to*, and it
is checked, not exempted. The same holds in the aggregates: each sign's swept
zero is a band row of its own (`zero = TRUE`), binade 0 holds only the
subnormals, and the categories give zero its own input class. A value disagreement outside the domain — the value
is specified to be NaN and something else came back — is a failure.

Alongside the cause, each region records **what the two sides returned**, as a
tally of (value kind, reference kind) pairs over the kinds NaN, +∞, −∞, +0, −0,
subnormal and normal, plus a representative input with its bits and both
values. For f64 the region's bounds are those of the 2³²-pattern blocks its
failing samples fell in and are **not** evidence that the endpoints were
evaluated; the first and last *sampled* failing inputs are carried separately
(`sampled_from`, `sampled_to`) and are the ones to cite.

The same kind pairs are tallied for every binade in the `kinds` table, in and
out of the domain separately, and two further things are counted per binade
without being failures: **signed-zero disagreements** (`n_zero_sign`: both
sides zero, opposite signs, which `==` cannot see) and **differences caused by
input flushing**, split by whether the zero they were flushed onto is
validated: `n_flushed` (it is, so the difference is the flush's alone) and
`n_flushed_zero_error` (it is not, so the subnormals also carry the error at
zero). Their error magnitudes stay in the histogram and worst inputs.

This is measurement, not judgement: it says what the numbers are and why.

### When base R is the weaker side: candidate disputes

`punif(1e-100, 0, 1, lower.tail = FALSE, log.p = TRUE)` is 0; the answer is
`log1p(-1e-100)` = −1e−100, which anvl returns. Every figure above stays
against base R regardless — nothing replaces it. A spec may additionally
declare a **stable reference** (`ref_stable`), an accurate evaluation of the
same function with the same parameters, and each sample is then tested
against it. A sample is a **candidate base R dispute** only when all three
hold, with *s* the stable value and *B* its declared error bound in absolute
units (`ref_stable_bound_ulp64` double ulps at *s*):

| condition | test |
|---|---|
| base R is off | \|g − s\| > 4 ulp_f64(s) + B |
| anvl is accurate | \|f − s\| ≤ 2 ulp_dtype(s) + B |
| anvl is no further | \|f − s\| ≤ \|g − s\| |

Where no ulp comparison is meaningful — *s* is ±0 or ±∞, or beyond the f32
range in an f32 cell — anvl must equal *s* at the cell's precision, sign of
zero included, and base R must not. A NaN on any side is never a dispute. The
multipliers are provisional thresholds for exclusion, not accuracy criteria:
failing them only leaves a sample counted against base R. The same test
records the opposite case too — anvl and base R returning the same value,
both beyond anvl's tolerance (`n_ref_shared`) — which a comparison against
base R alone can never see.

**A candidate is not an exclusion.** The sweep records, beside the unfiltered
figures and never in their place: candidate counts, the worst error without
candidates (`worst_rel_err_excl`, `worst_out_normal_excl`, per band, class and
result), a histogram column, a `ref_candidate` flag on regions and exact points
(their cause and category are left alone), and a `disputes` table keeping, per
binade, the samples furthest beyond tolerance with all three values, both
distances and both thresholds. Candidates become exclusions only once the
stable reference passes validation against high precision under a matching
identity, and until then no screen excludes anything. The `ref_stable_id` on
each result hashes the code that decides candidacy as it actually runs: the
stable reference, the classifier and the spacing function, each followed
recursively through every function it calls and every value it reads that is
not base R's (so a change to `same_value()`, to `DISPUTE_K`, or to a constant
captured in the stable reference's closure changes it), packages by version;
plus the declared bound, the exact reference parameters, the flags, the dtype,
the parameter policy and the R build. Each result also carries `ref_params`,
the exact parameters its reference received as hex doubles, so a later
validation reproduces them without re-running the sweep and without trusting
that a parameter-set name still means the same values. A stable reference is checked, not trusted: `nv_punif`'s
takes the small tail directly on each side, which is anvl's own algorithm, so
in f64 only the high-precision validation makes it independent evidence.

It costs ~60% more on a covered cell, ~2% on a full run, since only
`nv_punif`'s log-scale value cells declare one.

### `validate-refs` — checking the references against high precision

```bash
Rscript run.R validate-refs                      # every result in the store
Rscript run.R validate-refs --filter spec=nv_punif --prec 512
```

It needs Rmpfr, and nothing else does: run it wherever Rmpfr is installed,
after the sweep, without re-running it. For every result with a stable
reference (the evidence in a base R dispute) or a gradient reference (what
every gradient is scored against), it:

1. **re-derives the reference's identity** from the current code and the exact
   parameters the sweep stored (`ref_params`, hex). If that is not the identity
   the sweep recorded — the reference, anything it calls, its bound or its
   parameters changed — it fails without evaluating anything. A reference or a
   truth that cannot be identified — it calls `get()`, `do.call()`, `eval()` or
   similar however qualified (`base::get` is `get`), or names a function by a
   string to `sapply()` and friends, none of which a reading of the code can
   follow — can never pass; a truth without an identity fails before it is
   evaluated;
2. **samples** the result's exact points, its retained disputes, the two worst
   retained inputs of each binade, every earlier failing sample for the same
   cell and output (whatever the reference or truth version, so a change must
   revisit the known counterexamples), `--per-binade` random inputs (default 1) in every exponent
   field of both signs, `--focus-per-binade` more (default 8) in every binade
   holding a dispute (stable) or an error within 1e3 of the result's worst
   (gradient), and `--random` (default 512)
   uniform over all bit patterns — deterministically seeded per cell;
3. **compares** each with the spec's MPFR truth (`ref_stable_mpfr`,
   `ref_grad_mpfr`) at `--prec` bits (default 256), in double ulps at the
   truth, **with the difference, the division and the comparison with the bound
   all in MPFR** — converting the difference to a double first loses every
   fraction of a subnormal ulp (4.49 became 4, and passed a bound of 4). An
   exactly-zero truth needs a zero reference, and a NaN, infinite, or
   out-of-range-once-rounded truth an identical one. It passes only if no
   sample exceeds the declared bound — which is never raised afterwards;
4. **records** each validation in the store (`validations`: identity, truth
   identity, precision, seed, how samples were chosen, how many, the worst
   error and where, pass/fail and why; `validation_samples`: the worst samples
   and every failing one), and exports both beside the results they justify.
   Each record carries the **truth's identity** — the spec's own MPFR function
   and everything it calls, with the output selected recorded separately;
   nothing about the run — and the **validation method's identity** (the
   sampler, the comparator and the procedure). Validations by any other method
   count for nothing, so a change to how validation is done voids every earlier
   record rather than inheriting its verdicts.

A sampled validation is evidence, not a proof of a global bound, and is
recorded as such. Each truth mirrors its reference's conventions (endpoint
values, out-of-domain zeros) and differs only in evaluating the mathematics
exactly — which takes the same care as the reference: at 256 bits
1 − 4e−78 is exactly 1, so a truth written as log(1 − u) loses the answer just
as base R does, and the truths use the small-tail forms (`log1p`, `expm1`,
log Φ as `log1p(−Q)` for positive z). The normal family's deep tails use a
continued fraction and a log-space log Φ, since MPFR's own `exp()` underflows
at |z| ≈ 4e6.

**The status of a reference identity fails closed:** *validated* only if a
validation of exactly that identity passed and none failed; *failed* if any
did — a later pass does not undo it; otherwise *not validated*, or *no
identity*. It is judged against the most recent MPFR truth used for that
identity: a validation is evidence about the pair (reference, truth), so once
the truth's own code changes, older validations are superseded — kept, and
counted as such in `report`.

**Only then are candidates excluded.** A candidate dispute whose result's
stable reference is *validated* becomes a **verified base R limitation**
(category `reference_limitation`), at resolution time and in every reader
alike. Its figures against base R are unchanged; the worst error with the
limitations set aside is shown beside them, `status` lists these results in a
section of their own, and a result differing only by conventions and verified
limitations is counted on its own line. Every other candidate stays exactly
what the sweep recorded. A gradient reference's status gates nothing, but
`status` shows how many are validated, failed or unchecked, and `report`
shows each one's latest validation.

### The gradient references are checked too

A gradient's reference is an analytic formula written here, not base R, and
being analytic does not make its floating-point evaluation trustworthy. The
normal family's are built to avoid the ways the obvious evaluation fails:

- `z = (x - mean)/sd` is carried as a double-double (`std_z()`), because a
  rounded z costs `exp(-z^2/2)` about z² ulp — 85 ulp at z ≈ 8 on the shifted
  parameters — and anvl and base R round z the same way, so a reference that
  did too would share their error instead of measuring it;
- `m · φ(z)` and `m / φ(z)` go through `phi_times()` / `phi_recip()`, which
  split z so its square is exact and scale so nothing under- or overflows
  before the answer does: φ underflows at |z| ≈ 37.5 while `(z² − 1)φ(z)` is
  still 1.7e−321 at z = 38.6, and for huge z the bracket overflows while the
  answer is 0;
- `(z² − 1)/sd` as `(z − 1) · ((z + 1)/sd)`, finite where z² overflows;
- the inverse Mills ratio as a direct ratio above z = −20 and a continued
  fraction below, not a difference of two logs of ~−z²/2.

Checked against 256-bit MPFR on ~8,700 inputs per output across both
parameter sets (both tails, the underflow region, ±∞, up to 1e200), every
normal-family gradient reference is within 8 f64 ulp; the remainder is mostly
base R's own `pnorm` (up to 4 ulp) inside the log-scale ones. `selftest`
pins the cases that used to be 0, NaN or Inf to their MPFR values.

### Exact points, beside every sweep

The f64 sweep draws its low 32 bits at random, so it essentially never lands on
±0, ±∞, or an exact point such as p = 1 — a boundary bug there is invisible
without a separate check. Every cell therefore also evaluates a fixed set of
**exact points**, kept in their own `points` table and never added to the
sweep's counts, so nothing is counted twice:

- ±0, ±∞, NaN, the smallest and largest subnormal, the smallest normal, the
  largest finite value, ±0.5, ±1;
- each spec's valid-domain endpoints and distribution-support edges;
- anvl's **branch points**, where its implementation switches algorithm (anvl
  cells only);

every point **at the cell's precision** — a boundary of an f32 cell is the
boundary after conversion to f32 — and every finite one, universal points
included, with its two representable neighbours. The ±0 results are recorded
explicitly, so "zero passes" is a recorded comparison, not an absence of
regions.

The points take part in every assessment without entering the sweep's counts.
Each results row carries their summary (`n_points`, `n_points_identical`,
`n_points_failure`, `n_points_boundary`, `n_points_backend`,
`n_points_domain`, `worst_point_rel_err` and where), `status` lists a result
with a failing point among its failures, `report` and `browse` print every
point that is not bit-identical, and `Rscript query.R points --category failure`
lists them.

### What `status` counts

Every category stays on screen, and each counts: **failures**, **domain
boundary behaviour** and **backend limitations** are separate sections, each
counting regions and exact points. The largest finite errors are ranked over
the sweep and the points together. A result is called **bit-identical** only if
every sampled input and every exact point matches base R down to the sign of
zero; results that differ *only* by undefined-domain conventions are counted on
a line of their own, as set aside. `diff` compares every one of these counts,
not just the worst error. Candidate base R disputes are counted on a line of
their own and excluded from nothing.

### The reference sees the parameters the implementation sees

anvl converts a bare `min = -pi` to f32 for an f32 argument, so an f32 cell's
reference is evaluated with its parameters rounded to f32 too. Otherwise the
two sides compute different functions: f32(−π) lies *below* the double −π, and
at x = f32(−π) anvl is inside the support while a double-parameter reference is
outside — a failure by construction. Domain, support and branch points are
taken from the same rounded parameters.

### Correctly rounded is not the same as zero error

The reference is base R's double and is never rounded before scoring, so the
relative error measures numerical error against it. Separately, each sample
records whether the result is that reference **correctly rounded to the
result's precision** (`n_rounded` on the summary and bands, `rounded` on each
worst input). The two agree almost everywhere and part company at the edges of
the f32 range:

| base R's value | correctly rounded f32 | relative error | routed as |
|---|---|---|---|
| magnitude ≥ f32 overflow threshold (2^128 − 2^103) | ±∞ | infinite | a match, not a failure |
| below half the smallest subnormal | ±0 | exactly 1 | a finite error, marked rounded |
| representable, result ±∞ | — | infinite | a failure (spurious overflow) |

Both statements are kept because both are true: the implementation can be
perfectly rounded while its error against the reference is still 1. For f64 the
reference is already a double, so rounded and identical coincide and
`n_rounded` is always 0. The rounding itself relies on the platform's
double-to-float conversion; `run.R selftest` checks it at both edges on the
machine that runs the sweep.

### A known platform property: subnormal flush-to-zero

Every arithmetic and comparison operation on this PJRT/XLA CPU backend flushes
subnormals to zero, in **both** precisions. Storage round-trips correctly, but
`x >= 0` is `TRUE` for a negative subnormal, because the comparison sees `-0`:

```r
as.vector(nv_array(-1e-39, dtype = "f32") >= 0)             # TRUE  (R says FALSE)
as.double(nv_dunif(nv_array(-1e-39, dtype = "f32"), 0, 1))  # 1     (R says 0)
```

A subnormal input is therefore evaluated as ±0. That is *tested*, per sample:
a failure is attributed to input flushing only if the result is bit-identical to
the function's own result at the same-signed zero and that zero result is
validated (see the cause table above). Flushing that produces a materially
wrong answer still counts, as `n_flushed`, with its error kept. `Rscript query.R ranges
--cause input_flushing` lists every region; the day the backend stops flushing,
they become ordinary agreement.

**Never write the literal `-0` inside a function in this harness.** R's
byte-code compiler (R 4.6.1, JIT level 3 — verified) folds it to `+0` in some
call shapes: `c(0, -0)` in a compiled function returns two positive zeros. Use
`NEG_ZERO` (`R/util.R`), built from its bit pattern; `selftest` checks that it
survives.

## `export` — publishing a snapshot

The store accumulates: every run appends, and queries take the latest per cell.
A published artifact must instead be one coherent snapshot — **one artifact per
(anvl version, platform)**, covering every function and **every backend in the
store**, so anvl and its JAX twin can be compared side by side.

```bash
Rscript run.R export --out ../anvl-bench-darwin-arm64-cpu
Rscript run.R export --out <dir> --backends anvl     # restrict, if you mean to
```

Unlike `run`, `export` does not default to anvl alone: it exports whatever
backends the store holds.

```
manifest.json     index: schema version, platform, specs, depths, row counts
runs.parquet      the environment fingerprint of every run included
summary.parquet   the results table for every cell of every function
detail.parquet    the worst inputs, per binade
bands.parquet     the per-binade profile (unmerged; merged on render)
hist.parquet      the error distribution
ranges.parquet    the no-finite-error regions, resolved (cause, category, evidence)
kinds.parquet     what each side returned, per binade
points.parquet    the exact points, resolved
categories.parquet  per-result figures by input class (normal, zero, subnormal,
                  outside the domain, ±∞ & NaN); each cell's domain and support
                  are on summary
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
Rscript query.R ranges --category failure
Rscript query.R points --category failure
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
  domain    = function(p, f) c(-Inf, Inf),         # valid input domain
  support   = function(p, f) c(p$min, p$max),      # optional; reporting only
  branch_points = function(p, f, dtype) c(...),    # optional; anvl's own

  value     = function(x, dtype, p, f) as.double(anvl::nv_dnorm(...)),
  ref_value = function(x, p, f) dnorm(x, p$mean, p$sd, log = f$log),
  ref_stable = function(x, p, f) ...,                # optional; see "base R disputes"
  ref_stable_bound_ulp64 = 8,                        #   its error bound, double ulps
  ref_stable_note = "why base R is weaker here",     #   required with ref_stable
  ref_stable_covers = function(f) isTRUE(f$log_p),   #   optional; which flags

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
2. **Declare the domain honestly, and keep it apart from the support.** The
   *domain* is where the function is defined at all (p in [0, 1] for a
   quantile); outside it the value is NaN by specification. The *support* is
   where the distribution lives, and excuses nothing: a CDF below its support
   has a perfectly good value. `branch_points` are read from anvl's source and
   go stale when it changes — keep the pointer to the source beside them.
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
