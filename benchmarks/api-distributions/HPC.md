# Distributed sweep execution

This guide describes the harness's requirements and behaviour when work is split
across processes or machines. For image builds, Slurm submission, calibration,
monitoring and release ZIP creation, use the
[anvl-bench cluster guide](../../../anvl-bench/hpc/README.md) in the sibling
checkout. The [harness README](README.md) owns scoring, reference validation,
result selection and the export schema.

## Assigning work

`run.R list --backends anvl,jax` reports the current grid. On 2026-09-29 it
contains 160 anvl cells and 96 JAX cells, producing 512 results. JAX covers only
a subset of variants; derive task counts from the image or checkout you run.

For a sweep, the harness sorts the grid by `cell_id`, applies the filter, then
assigns row position `k` to shard `((k - 1) %% n) + 1`. Shard numbers start at
1. Every worker must use the same specs, backends, filter and shard count.
Changing any of these can change the assignment. More shards than cells leaves
empty tasks. Related cells, including anvl/JAX twins, need not share a task or
node.

Validation uses a different partition: sorted reference units, identified by
reference kind, identity and output. Anvl and JAX results sharing a unit
contribute inputs to one validation. All validation workers must see the same
merged sweep data and use the same code and selection. Keep sweep data fixed
until validation finishes. Validation records can be written concurrently to
that store; they do not need a second merge.

## Requirements across workers

- Use matching harness code, installed ecosystem packages, runtime settings and
  sampling seed. The harness calls installed `anvl`; checkout SHAs alone do not
  establish which code is installed.
- Use the same depth for directly comparable inputs. The f64 low bits use the
  harness's fixed `SWEEP_SEED`; do not replace it with a per-task seed.
- Give every process a writable store and sufficient temporary/cache space.
  Containers and scheduler-specific paths belong in the deployment guide.
- Set `NV_SWEEP_DEVICE` to the device actually used. This is a provenance label,
  not a device selector. Keep publication stores separate by platform and
  intended software version.
- Run `selftest` in a separate store on the target compute environment before a
  long sweep. It checks selected scoring and platform behaviours; it does not
  establish full sweep coverage.

## Run identity and value/gradient evidence

**Current limitation:** every `run` invocation creates a separate `run_id`.
Workers created by one invocation's `--jobs` share that ID, but separate shard
invocations do not. Merging preserves the IDs.

Classifying an out-of-domain gradient disagreement as `undefined_domain`
requires evidence from the matching value cell in the **same run**. If the two
cells land in different shards, merging their stores cannot supply that
same-run evidence: the disagreement remains a failure, with an evidence note.
This affects classification, not the recorded numerical comparisons.

The ordinary row-based shard assignment does not keep these pairs together.
For complete convention classification with the current implementation, use
one invocation per selected group containing both value and gradient cells
(for example, one per function, without a `kind` filter), optionally with
`--jobs`. Any custom grouping must preserve those pairs. A shared-store path
alone does not create a shared run identity.

## Stores, merging and completeness

Each invocation writes separate files by table, run and cell. Workers can use
one shared store or separate stores. `merge` copies table files into the
destination; it does not turn their runs into one run or verify completeness.
Partial stores can be merged for inspection or recovery.

**Process success is not a completeness check.** `run` catches cell errors,
records them and prints `done: N ok, M error`, but does not return a failing
process exit merely because cells failed. `validate-refs` likewise records
failed comparisons without requiring a nonzero process exit. A scheduler's
successful dependency therefore establishes neither successful coverage nor
passing reference validation.

Before publication, compare successful cells with the expected filtered grid
at the intended depth, inspect error records and logs, and inspect reference
statuses. Existing results in an accumulating store can mask missing work in a
new submission. Prefer a fresh store for each publication version/platform;
consult the README's result-selection rules when reusing one.

Validation distinguishes failed comparisons from interrupted jobs. A completed
check can record `failed`; an interrupted invocation may write no new records.
Earlier records remain and can still determine status. Retry an interrupted
validation against unchanged inputs and code. If a reference changes identity,
rerun the affected sweeps before validating: old results record the old
reference identity. The README describes truth and method identity changes.

## Scheduler-independent example

Run these commands from this directory. Paths below are examples: choose new,
absolute writable locations for a campaign. Execute the sweep command once for
each `i` from 1 through `n`; this row-based example has the evidence limitation
above. Choose `n` after inspecting the grid and measuring the workload.

```bash
parts=/scratch/my-sweep/parts
analysis=/scratch/my-sweep/analysis
n=8
i=1

Rscript run.R list --backends anvl,jax
Rscript run.R run --dry-run --backends anvl,jax --shard "$i" --shards "$n"
Rscript run.R run --store "$parts/shard-$i" --depth full \
  --backends anvl,jax --quiet --shard "$i" --shards "$n"

# After all sweep tasks finish, inspect their logs and merge each store.
for i in $(seq 1 "$n"); do
  Rscript run.R merge --store "$analysis" --from "$parts/shard-$i"
done
Rscript run.R status --store "$analysis" --backends anvl,jax

# Execute once per j from 1 through m, against the same merged data.
m=4
j=1
Rscript run.R validate-refs --store "$analysis" --backends anvl,jax \
  --shard "$j" --shards "$m"

# After validation finishes and coverage/statuses have been inspected:
Rscript run.R export --store "$analysis" --out /scratch/my-sweep/export
```

Sweep processing uses chunks, but that does not bound every job's memory:
retained regions, compilation, worker count and analysis-table reads also
matter. Measure representative value and gradient cells, and measure merge,
validation and export separately. Cluster resource settings and measurements
belong in the deployment guide.
