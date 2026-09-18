## ---------------------------------------------------------------------------
## The result store.
##
## Parquet, in a partitioned directory, written once and never mutated.
##
## That last clause is the whole design. Every cell writes its own file, so two
## processes never touch the same bytes and there is no lock to take, no
## contention to tune and nothing that behaves differently on a network
## filesystem. Parallel jobs, a second machine and an HPC array are all the
## same case, and merging results from different machines is `cp -r`.
##
## This is why SQLite and DuckDB are not used as the write target: SQLite's
## locking is documented-unreliable on NFS/Lustre, and DuckDB permits only one
## writer process per database file. DuckDB is excellent on the *read* side and
## query.R will use it when it is installed -- but nothing depends on it.
##
## Parquet earns its place on type fidelity rather than size: the store is
## small, but it holds NaN, +-Inf, signed zero and subnormals, all of which a
## CSV round-trip would quietly destroy. Verified against nanoparquet, not
## assumed. The one thing that does *not* survive is a 64-bit integer, which
## comes back widened to a double -- so bit patterns are stored as hex strings.
## ---------------------------------------------------------------------------

TABLES <- c("runs", "results", "detail", "ranges", "hist", "bands")

## Where the store lives. Out of the package tree by default, so a result file
## can never be committed by accident and the package stays what upstream
## tracks: scripts, config and docs. NV_SWEEP_STORE overrides it, which is also
## how a compute node is pointed at cluster scratch.
store_dir <- function(override = NULL) {
  d <- if (is.null(override)) Sys.getenv("NV_SWEEP_STORE", "") else override
  if (!nzchar(d)) {
    d <- file.path(tools::R_user_dir("anvl-sweeps", "cache"), "store")
  }
  d
}

store_init <- function(dir) {
  for (t in TABLES) {
    dir.create(file.path(dir, t), recursive = TRUE, showWarnings = FALSE)
  }
  invisible(dir)
}

## One cell's contribution to one table. The filename carries the run and the
## cell, so a collision is impossible by construction rather than by locking.
store_write <- function(dir, table, run_id, key, df) {
  if (is.null(df) || !nrow(df)) {
    return(invisible(NULL))
  }
  d <- file.path(dir, table, paste0("run=", run_id))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  f <- file.path(d, paste0(key, ".parquet"))
  ## Write to a temporary name in the same directory and rename into place:
  ## rename is atomic, so a reader never sees a half-written file and an
  ## interrupted job leaves no torn part behind.
  tmp <- paste0(f, ".tmp", Sys.getpid())
  nanoparquet::write_parquet(df, tmp)
  file.rename(tmp, f)
  invisible(f)
}

## Read one table whole. The store is small enough that this is always fine;
## query.R offers DuckDB for anyone who would rather push predicates down.
store_read <- function(dir, table) {
  d <- file.path(dir, table)
  if (!dir.exists(d)) {
    return(NULL)
  }
  files <- list.files(d, pattern = "\\.parquet$", recursive = TRUE, full.names = TRUE)
  if (!length(files)) {
    return(NULL)
  }
  parts <- lapply(files, function(f) as.data.frame(nanoparquet::read_parquet(f)))
  ## Shards written by different versions of a spec can differ in columns;
  ## union them rather than failing, so an older run stays readable.
  cols <- unique(unlist(lapply(parts, names)))
  parts <- lapply(parts, function(p) {
    for (m in setdiff(cols, names(p))) {
      p[[m]] <- NA
    }
    p[cols]
  })
  do.call(rbind, parts)
}

## Merge one store into another -- the cross-machine case. Parts are immutable
## and uniquely named, so this is a copy that can be repeated safely; an
## existing destination file is left alone rather than rewritten.
store_merge <- function(from, into) {
  n <- 0L
  for (t in TABLES) {
    src <- file.path(from, t)
    if (!dir.exists(src)) {
      next
    }
    for (f in list.files(src, pattern = "\\.parquet$", recursive = TRUE)) {
      dst <- file.path(into, t, f)
      if (file.exists(dst)) {
        next
      }
      dir.create(dirname(dst), recursive = TRUE, showWarnings = FALSE)
      file.copy(file.path(src, f), dst)
      n <- n + 1L
    }
  }
  n
}

## Collapse to one row per (cell, output): the deepest sweep available.
##
## `latest_results()` keeps a row per depth, which is what the coverage table
## wants, but every other consumer wants one current verdict per measurement.
## A shallower sweep visits a subset of a deeper one's inputs -- same index
## space, larger stride -- so the deeper row is strictly better evidence and
## its bound is the one that matters. Shared by `status` and `baseline` so the
## two can never disagree about what "the current result" is.
deepest_per_cell <- function(res, depths) {
  if (is.null(res) || !nrow(res)) {
    return(res)
  }
  rank <- match(res$depth, depths)
  res <- res[order(res$cell_id, res$output, -rank), , drop = FALSE]
  res <- res[!duplicated(paste(res$cell_id, res$output, sep = "\r")), , drop = FALSE]
  rownames(res) <- NULL
  res
}

## The most recent result for each (cell, platform), which is what "current
## state" means when history is retained rather than overwritten.
latest_results <- function(dir) {
  res <- store_read(dir, "results")
  if (is.null(res)) {
    return(NULL)
  }
  runs <- store_read(dir, "runs")
  if (!is.null(runs)) {
    res <- merge(
      res,
      runs[c("run_id", "started_at", "host", "cpu", "branch", "anvl_sha")],
      by = "run_id",
      all.x = TRUE,
      suffixes = c("", ".run")
    )
  }
  o <- order(res$started_at %||% res$run_id, decreasing = TRUE)
  res <- res[o, , drop = FALSE]
  k <- paste(res$cell_id, res$output, res$platform_key, res$depth, sep = "\r")
  res <- res[!duplicated(k), , drop = FALSE]
  rownames(res) <- NULL
  res
}
