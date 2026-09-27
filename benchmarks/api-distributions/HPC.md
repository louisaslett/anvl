# Running the sweeps on an HPC cluster

The harness is built to be sharded; this note says why, and what to get right.
It is deliberately not a job script: how the stack gets onto a cluster (a
container is far more likely to work than a native build) and how tasks are
scheduled are site-specific.

## Why bother

At `full` depth a cell is minutes and the grid is 160 cells (320 with the JAX
twins), so a complete pass is many hours on one laptop. The better reason is
**breadth**: an x86-64 CPU column separates "XLA does this" from "XLA on NEON
does this", and a CUDA column is the one comparison a laptop cannot make.

## Why it shards cleanly

- **The grid is canonically ordered.** `build_grid()` sorts by `cell_id`, so
  "shard 3 of 40" means the same cells on every machine with no coordination.
- **Every cell writes its own Parquet file**, named by run and cell: no locking,
  no append, no shared handle — what a parallel filesystem wants.
- **Merging is a file copy:** `run.R merge --from <dir>`.
- **Validation shards too**, by reference rather than by cell (see below).

## The three steps

```bash
# 1. the sweep, one task per shard, each into its own store
Rscript run.R run --depth full --backends anvl,jax --quiet --shard "$i" --shards "$n"

# 2. once every shard has finished: fold them into the analysis store
Rscript run.R merge --from <the shards' store>

# 3. then validate the references against MPFR, sharded again, writing
#    straight into the analysis store (needs Rmpfr)
Rscript run.R validate-refs --backends anvl,jax --shard "$j" --shards "$m"
```

Then `status`, `report` and `export` as usual.

Validation must follow the merge: a reference is validated from each result's
exact points, disputes and worst inputs and from every earlier counterexample,
all of which are in the merged store. It shards by **reference unit** — a
distinct reference identity — so an anvl cell and its JAX twin, which share a
reference, are validated once from both cells' inputs; each task writes its own
record, so there is nothing to merge afterwards. A task that fails leaves its
references *not validated*, which excludes nothing: the result is only more
conservative.

`--backends anvl,jax` needs a Python with `jax` visible to reticulate
(`RETICULATE_PYTHON`). Leave it off for anvl alone.

## What will actually be hard

1. **Getting the stack onto the cluster at all.** anvl needs stablehlo, pjrt
   and tengen, and pjrt carries a compiled XLA runtime. Never
   `devtools::install()`: it resolves anvl's `Remotes:` and upgrades the
   siblings from GitHub main, which breaks anvl whenever the ecosystem is
   mid-migration. `R CMD INSTALL <dir>` per package, in dependency order.
2. **Matching the PJRT plugin to the node's CUDA and driver**, for a GPU run.
   Check `nvidia-smi` on a compute node, not the login node.
3. **Where caches go.** R, reticulate and XLA write under `HOME`; point `HOME`
   and `TMPDIR` at scratch, not a quota'd home directory, and make sure every
   task sees the same R library.
4. **Cold-start cost per task.** Each task pays R startup, package load and an
   XLA compile per distinct cell shape, so prefer fewer, longer shards.
5. **Walltime, not memory.** The reducers are streaming, so memory is flat in
   the number of samples; time one `full` cell first and multiply.

## Things to get right the first time

- **Threads are governed by CPU affinity, not `OMP_NUM_THREADS`.** XLA sizes its
  intra-op pool from the schedulable-CPU count, so bind each task to its cores
  (e.g. `srun --cpu-bind=cores`). Pin the thread-count variables to 1: they only
  reach R's BLAS, which this workload barely uses, and a pool sized to the
  allocation would contend with XLA's for the same cores.
- **`NV_SWEEP_DEVICE`** must say what actually ran. It feeds `platform_key`,
  and a CUDA run mislabelled `cpu` silently merges into the CPU column.
- **Do not merge a partial sweep** into the analysis store; `status` counts
  cells against the declared grid and a half-merged run reads as "never run".
- **The f64 sweep's low mantissa bits are seeded** (`SWEEP_SEED` in
  `R/engine.R`), so two machines sweep identical inputs and a cross-platform
  difference is a real one. Do not make the seed per task.
- **Run `selftest` on the cluster before a long job**, not just where the
  environment was built: its rounding and `-0` checks exist to catch a platform
  difference. **Check one shard** first too: `run --dry-run` prints the cells a
  shard would take without running anything.

## Afterwards

```r
source("query.R")
sw_compare("darwin-arm64-cpu", "linux-x86_64-cpu")
sw_sql("select platform_key, dtype, count(*), max(worst_rel_err)
        from results group by 1, 2 order by 1, 2")
```
