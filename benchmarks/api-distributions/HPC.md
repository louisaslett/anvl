# Running the sweeps on an HPC cluster

Notes for taking the `full` grid to the university cluster. Nothing here has
been executed yet — this is the plan and the known hazards, not a tested
recipe. Phase 5 of the harness work.

## Why bother

At `full` depth a cell is minutes and the grid is 160 cells (320 with the JAX
twins), so a complete pass is many hours on one laptop. That is survivable
overnight. The real reason to go to the cluster is **breadth, not speed**:

- a **CUDA** `platform_key` alongside `darwin-arm64-cpu`, which is the one
  comparison this harness is built for and cannot make on a laptop;
- an **x86-64 CPU** column, to separate "XLA does this" from "XLA on NEON does
  this" — the subnormal flush-to-zero documented in the README is currently a
  single-platform observation;
- room to run `full` routinely rather than as an event.

## The design already fits

Nothing needs to change to shard this.

- **The grid is canonically ordered.** `build_grid()` sorts by `cell_id`, which
  is deterministic and identical on every machine, so "shard 3 of 40" means the
  same cells everywhere with no coordination.
- **Every cell writes its own Parquet file**, named by run and cell. No locking,
  no append, no shared handle — which is exactly what a parallel filesystem
  wants, and why neither SQLite nor DuckDB is the write target.
- **Merging is a file copy.** `run.R merge --from <dir>`.

```bash
Rscript run.R run --depth full --shard "$SLURM_ARRAY_TASK_ID" --shards 40
```

Each task writes into `$NV_SWEEP_STORE`; point that at scratch, then merge into
the store you analyse on.

## A sketch of the job script

```bash
#!/bin/bash
#SBATCH --array=1-40
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/sweep-%A_%a.out

export NV_SWEEP_STORE="$SCRATCH/anvl-sweeps/$SLURM_ARRAY_JOB_ID"
export NV_SWEEP_DEVICE=cpu            # or cuda, on a GPU partition
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

Rscript run.R run --depth full --quiet \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT"
```

Then, once:

```bash
Rscript run.R merge --from "$SCRATCH/anvl-sweeps/<jobid>"
Rscript run.R status
```

## What will actually be hard

The storage and sharding are the easy part. Budget the time for these.

1. **Getting the stack onto the cluster at all.** anvl needs stablehlo, pjrt
   and tengen, and pjrt carries a compiled XLA runtime. This is the whole
   difficulty of the exercise. Decide early between a container (Apptainer/
   Singularity — far more likely to work, and reproducible) and a module-based
   native build. Do not attempt `devtools::install()`: it resolves anvl's
   `Remotes:` and upgrades the siblings from GitHub main, which breaks anvl
   whenever the ecosystem is mid-migration. `R CMD INSTALL <dir>` per package,
   in dependency order.
2. **Matching the PJRT plugin to the node's CUDA and driver.** A GPU run needs
   the plugin built against a compatible CUDA, and cluster drivers are usually
   older than a laptop's. Check `nvidia-smi` on a compute node, not the login
   node.
3. **`R_LIBS_USER` on a shared filesystem.** Put the library on scratch or in
   the container, not in a home directory with a small quota; and make sure
   every array task sees the same one.
4. **Cold-start cost per task.** Each task pays R startup, package load, and an
   XLA compile per distinct cell shape. With 40 shards that is 40× the
   compile cost, so prefer fewer, longer shards over many short ones.
5. **Walltime and the top-K reducer.** Memory is flat in the number of samples
   (the reducers are streaming), so a cell cannot grow out of its allocation;
   walltime is the binding constraint. Time one `full` cell locally first and
   multiply.

## Things to get right the first time

- **`NV_SWEEP_DEVICE`** must say what actually ran. It feeds `platform_key`,
  and a CUDA run mislabelled `cpu` silently merges into the CPU column and
  corrupts exactly the comparison the exercise is for.
- **Do not merge a partial array into the analysis store** until the array has
  finished; `status` counts cells against the declared grid and a half-merged
  run reads as "never run" for the rest.
- **The f64 sweep's low mantissa bits are seeded** (`SWEEP_SEED` in
  `R/engine.R`), so two machines sweep identical inputs and a cross-platform
  difference is a real difference. Do not make the seed per-task.
- **Check one shard interactively before submitting 40.** `--dry-run` prints
  the cells a shard would take without running anything.

## Afterwards

```r
source("query.R")
sw_compare("darwin-arm64-cpu", "linux-x86_64-cuda")
sw_sql("select platform_key, dtype, count(*), max(worst_rel_err)
        from results group by 1, 2 order by 1, 2")
```

If the CUDA column disagrees with CPU anywhere outside the `subnormal` class,
that is the finding the trip was for.
