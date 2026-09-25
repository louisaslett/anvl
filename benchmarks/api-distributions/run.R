#!/usr/bin/env Rscript
## ---------------------------------------------------------------------------
## The single entry point. Run it from anywhere.
##
##   Rscript run.R list                       what cells exist
##   Rscript run.R run --depth smoke          run them all, coarsely
##   Rscript run.R run --filter spec=nv_punif,dtype=f64 --depth full
##   Rscript run.R status                     what has run, where, and what fails
##   Rscript run.R selftest                   prove the engine still detects errors
##   Rscript run.R merge --from <dir>         fold another machine's store in
##
## Everything it writes goes to the store (NV_SWEEP_STORE), never into the
## package tree. See README.md.
## ---------------------------------------------------------------------------

suppressWarnings(suppressMessages({
  ## `here()` is this script's directory, so the harness is runnable from any
  ## working directory -- the old drivers depended on being cd'd into their own
  ## folder to resolve source("../f64-sweep.R").
  HERE <- local({
    a <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", grep("^--file=", a, value = TRUE))
    if (length(f)) dirname(normalizePath(f[1L])) else normalizePath(".")
  })
}))
here <- function() HERE

for (f in c("util.R", "engine.R", "cells.R", "render.R", "provenance.R", "store.R")) {
  source(file.path(HERE, "R", f))
}

## ---- argument parsing ------------------------------------------------------

parse_args <- function(argv) {
  cmd <- if (length(argv) && !startsWith(argv[1L], "--")) argv[1L] else "run"
  argv <- argv[argv != cmd]
  o <- list(
    depth = "smoke",
    depth_given = NULL,
    filter = "",
    jobs = 1L,
    store = NULL,
    from = NULL,
    to = NULL,
    out = NULL,
    dry_run = FALSE,
    shard = NA_integer_,
    shards = NA_integer_,
    backends = "anvl",
    ## Whether --backends was given. `run` defaults to anvl alone because the
    ## JAX side needs Python; `export` must not inherit that default, or it
    ## silently drops every JAX result already in the store.
    backends_given = FALSE,
    ## worst inputs kept *per binade*, not globally -- see reducer_topk()
    topk = 10L,
    ## How many full report pages to print before summarising instead. Four is
    ## about what fits on a screen at a glance.
    pages = 4L,
    quiet = FALSE
  )
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[i]
    val <- function() {
      if (i + 1L > length(argv)) {
        stop("option ", a, " needs a value", call. = FALSE)
      }
      argv[i + 1L]
    }
    switch(
      sub("^--", "", a),
      depth = {
        o$depth <- val()
        o$depth_given <- o$depth
        i <- i + 1L
      },
      filter = {
        o$filter <- val()
        i <- i + 1L
      },
      jobs = {
        o$jobs <- as.integer(val())
        i <- i + 1L
      },
      store = {
        o$store <- val()
        i <- i + 1L
      },
      to = {
        o$to <- val()
        i <- i + 1L
      },
      out = {
        o$out <- val()
        i <- i + 1L
      },
      from = {
        o$from <- val()
        i <- i + 1L
      },
      shard = {
        o$shard <- as.integer(val())
        i <- i + 1L
      },
      shards = {
        o$shards <- as.integer(val())
        i <- i + 1L
      },
      backends = {
        o$backends <- trimws(strsplit(val(), ",")[[1L]])
        o$backends_given <- TRUE
        i <- i + 1L
      },
      topk = {
        o$topk <- as.integer(val())
        i <- i + 1L
      },
      pages = {
        o$pages <- as.integer(val())
        i <- i + 1L
      },
      `dry-run` = o$dry_run <- TRUE,
      quiet = o$quiet <- TRUE,
      stop("unknown option: ", a, call. = FALSE)
    )
    i <- i + 1L
  }
  list(cmd = cmd, opt = o)
}

## ---- running one cell ------------------------------------------------------

## Build the swept function and its reference for one grid row. Both return a
## named list of outputs: one entry for a value cell, one per differentiated
## argument for a gradient cell. Scoring every gradient from a single reverse
## pass is a straight 3x saving on the bulk of the grid -- the old drivers
## re-swept the whole space once per argument and discarded two thirds of each
## pass.
cell_functions <- function(spec, row) {
  params <- spec$params[[row$param_set]]
  flags <- if (nzchar(row$flags) && row$flags != "-") {
    kv <- strsplit(strsplit(row$flags, ",", fixed = TRUE)[[1L]], "=", fixed = TRUE)
    setNames(lapply(kv, function(p) as.logical(p[2L])), vapply(kv, `[`, "", 1L))
  } else {
    list()
  }

  if (row$kind == "value") {
    fn <- if (row$backend == "jax") spec$jax_value else spec$value
    list(
      fun = function(x) list(value = fn(x, row$dtype, params, flags)),
      ref = function(x) list(value = spec$ref_value(x, params, flags)),
      outputs = "value",
      params = params,
      flags = flags
    )
  } else {
    fn <- if (row$backend == "jax") spec$jax_grad else spec$grad
    list(
      fun = function(x) fn(x, row$dtype, params, flags),
      ref = function(x) spec$ref_grad(x, params, flags),
      outputs = spec$grad_wrt,
      params = params,
      flags = flags
    )
  }
}

run_cell <- function(spec, row, opt, pv, dir) {
  cf <- cell_functions(spec, row)
  key <- cell_key(row$cell_id)

  t0 <- Sys.time()
  out <- tryCatch(
    run_sweep(
      cf$fun,
      cf$ref,
      row$dtype,
      opt$depth,
      cf$outputs,
      progress = !opt$quiet,
      topk = opt$topk
    ),
    error = function(e) {
      structure(conditionMessage(e), class = "sweep_error")
    }
  )
  if (inherits(out, "sweep_error")) {
    res <- cbind(
      row[rep(1L, 1L), ],
      data.frame(
        run_id = pv$run_id,
        output = "-",
        platform_key = pv$platform_key,
        device = pv$device,
        depth = opt$depth,
        n_samples = NA_real_,
        n_exact = NA_real_,
        n_rounded = NA_real_,
        n_inf = NA_real_,
        n_inf_runs = NA_integer_,
        n_runs_unclassified = 0L,
        worst_rel_err = NA_real_,
        worst_ulp_err = NA_real_,
        worst_x = NA_real_,
        worst_bits = NA_character_,
        worst_value = NA_real_,
        worst_reference = NA_real_,
        unexplained_from = NA_real_,
        unexplained_to = NA_real_,
        elapsed_sec = NA_real_,
        error = as.character(out)
      )
    )
    store_write(dir, "results", pv$run_id, key, res)
    return(list(status = "ERROR", msg = as.character(out)))
  }

  support <- if (is.null(spec$support)) NULL else spec$support(cf$params, cf$flags)

  res_rows <- list()
  for (o in cf$outputs) {
    r <- out[[o]]
    cls <- classify_ranges(r$ranges, support, row$dtype)
    if (nrow(r$ranges)) {
      r$ranges$class <- cls
      r$ranges <- cbind(
        data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o),
        r$ranges
      )
      store_write(dir, "ranges", pv$run_id, paste0(key, "-", o), r$ranges)
    }
    if (nrow(r$detail)) {
      d <- r$detail
      store_write(
        dir,
        "detail",
        pv$run_id,
        paste0(key, "-", o),
        cbind(
          data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o),
          d
        )
      )
    }
    if (nrow(r$bands)) {
      store_write(
        dir,
        "bands",
        pv$run_id,
        paste0(key, "-", o),
        cbind(
          data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o),
          r$bands
        )
      )
    }
    store_write(
      dir,
      "hist",
      pv$run_id,
      paste0(key, "-", o),
      cbind(
        data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o),
        r$hist
      )
    )

    res_rows[[o]] <- cbind(
      row,
      data.frame(
        run_id = pv$run_id,
        output = o,
        platform_key = pv$platform_key,
        device = pv$device,
        depth = opt$depth,
        n_samples = r$summary$n_samples,
        n_exact = r$summary$n_exact,
        n_rounded = r$summary$n_rounded,
        n_inf = r$summary$n_inf,
        n_inf_runs = r$summary$n_inf_runs,
        n_runs_unclassified = sum(cls == "unclassified"),
        worst_rel_err = r$summary$worst_rel_err,
        worst_ulp_err = r$summary$worst_ulp_err,
        ## The input that produced the worst error, and where the first
        ## unexplained disagreement starts, both carried on the summary row so
        ## that `status` can say "how bad, and where" without a second lookup.
        worst_x = r$summary$worst_x,
        worst_bits = r$summary$worst_bits,
        worst_value = r$summary$worst_value,
        worst_reference = r$summary$worst_reference,
        unexplained_from = if (any(cls == "unclassified")) r$ranges$x_from[cls == "unclassified"][1L] else NA_real_,
        unexplained_to = if (any(cls == "unclassified")) r$ranges$x_to[cls == "unclassified"][1L] else NA_real_,
        elapsed_sec = as.numeric(difftime(Sys.time(), t0, units = "secs")),
        error = NA_character_
      )
    )
  }
  res <- do.call(rbind, res_rows)
  store_write(dir, "results", pv$run_id, key, res)
  list(status = "OK", res = res)
}

## ---- commands --------------------------------------------------------------

cmd_list <- function(opt) {
  g <- apply_filter(build_grid(load_specs(), opt$backends), opt$filter)
  cat(sprintf("%d cells, %d result rows\n\n", nrow(g), sum(g$n_outputs)))
  print(g[c("cell_id", "n_outputs")], right = FALSE)
  invisible(g)
}

cmd_run <- function(opt) {
  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, opt$backends), opt$filter)
  if (!nrow(g)) {
    stop("filter matched no cells", call. = FALSE)
  }

  ## HPC sharding: the grid is canonically ordered, so shard i of n means the
  ## same set of cells on every node without any coordination between them.
  if (!is.na(opt$shards)) {
    if (is.na(opt$shard)) {
      stop("--shards needs --shard", call. = FALSE)
    }
    g <- g[(seq_len(nrow(g)) - 1L) %% opt$shards == (opt$shard - 1L), , drop = FALSE]
    cat(sprintf("shard %d/%d: %d cells\n", opt$shard, opt$shards, nrow(g)))
  }

  if (opt$dry_run) {
    cat(sprintf("would run %d cells at depth '%s':\n", nrow(g), opt$depth))
    cat(paste0("  ", g$cell_id, collapse = "\n"), "\n")
    return(invisible(g))
  }

  dir <- store_dir(opt$store)
  store_init(dir)
  pv <- collect_provenance(opt$depth)
  store_write(dir, "runs", pv$run_id, "run", provenance_row(pv))

  cat(sprintf(
    "run %s | depth %s | %d cells | store %s\n",
    pv$run_id,
    opt$depth,
    nrow(g),
    dir
  ))

  one <- function(i) {
    row <- g[i, , drop = FALSE]
    rownames(row) <- NULL
    r <- run_cell(specs[[row$spec]], row, opt, pv, dir)
    cat(sprintf("[%3d/%3d] %-6s %s\n", i, nrow(g), r$status, row$cell_id))
    r$status
  }

  ## Cells are independent and write to distinct files, so forking needs no
  ## coordination at all. Each worker inherits the same run_id, which is what
  ## makes the parts reassemble into one run.
  st <- if (opt$jobs > 1L) {
    unlist(parallel::mclapply(seq_len(nrow(g)), one, mc.cores = opt$jobs, mc.preschedule = FALSE))
  } else {
    vapply(seq_len(nrow(g)), one, "")
  }

  cat(sprintf("\ndone: %d ok, %d error\n", sum(st == "OK"), sum(st != "OK")))
  cat(sprintf("run id: %s\n", pv$run_id))
  invisible(pv$run_id)
}

## The index: what has been swept, and what disagrees with base R most.
##
## Measurements only. Nothing here decides whether a result is acceptable --
## that judgement depends on the function, the precision and what the caller
## needs, and it belongs to the person reading, not to a file.
cmd_status <- function(opt) {
  dir <- store_dir(opt$store)
  res <- latest_results(dir)
  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, opt$backends), opt$filter, extra = "output")

  if (is.null(res)) {
    cat("The result store is empty.\n  ", dir, "\n\n")
    cat(sprintf("%d cells are declared. To fill them:\n", nrow(g)))
    cat("  Rscript run.R run --depth smoke\n")
    return(invisible(NULL))
  }
  res <- filter_results(res[res$cell_id %in% g$cell_id, , drop = FALSE], opt$filter)
  runs <- store_read(dir, "runs")

  ## `all_depths` drives coverage, which is about what has been run. `res` is
  ## collapsed to the deepest result per cell, because everything else here is
  ## about the best evidence available.
  all_depths <- res
  res <- deepest_per_cell(res, names(DEPTHS))

  cat("\nanvl distribution sweeps \u2014 status\n")
  cat("store: ", dir, "\n", sep = "")
  if (!is.null(runs)) {
    cat(sprintf("       %d run(s) recorded, most recent %s\n", nrow(runs), max(runs$started_at)))
  }

  ## ---- coverage ------------------------------------------------------------
  rule("COVERAGE")
  cat("A *cell* is one combination of function, precision, value-or-gradient,\n")
  cat("parameter set and flags. A gradient cell reports one *result* per\n")
  cat("differentiated argument, so cells and results are different counts.\n\n")
  for (pk in unique(all_depths$platform_key)) {
    cat(sprintf("  %s\n", pk))
    for (d in names(DEPTHS)) {
      n <- length(unique(all_depths$cell_id[all_depths$depth == d & all_depths$platform_key == pk]))
      cat(
        trimws(
          sprintf(
            "    %-6s %4d of %4d cells   %s",
            d,
            n,
            nrow(g),
            if (n == nrow(g)) {
              "(complete)"
            } else if (n == 0L) {
              "(not run)"
            } else {
              "(partial)"
            }
          ),
          "right"
        ),
        "\n",
        sep = ""
      )
    }
  }
  swept <- length(unique(all_depths$cell_id))
  cat(sprintf(
    "\n  %d of %d cells swept at some depth, producing %d results.\n",
    swept,
    nrow(g),
    nrow(res)
  ))
  if (swept < nrow(g)) {
    miss <- setdiff(g$cell_id, unique(all_depths$cell_id))
    cat(sprintf("  %d cell(s) never swept, e.g. %s\n", length(miss), miss[1L]))
  }

  ## ---- the measurements ----------------------------------------------------
  unexp <- res$n_runs_unclassified > 0
  line <- function(r, i) {
    row <- g[g$cell_id == r$cell_id[i], , drop = FALSE][1L, , drop = FALSE]
    what <- if (r$kind[i] == "value") "value" else paste0("d/d", r$output[i])
    cat(sprintf(
      "  %-9s %-3s %-7s %-22s %s\n",
      r$spec[i],
      r$dtype[i],
      what,
      short_cell(row),
      if (r$n_runs_unclassified[i] > 0) {
        sprintf(
          "%s .. %s",
          fmt_num(min(r$unexplained_from[i], r$unexplained_to[i])),
          fmt_num(max(r$unexplained_from[i], r$unexplained_to[i]))
        )
      } else {
        sprintf("rel %-10s %s ulp", fmt_num(r$worst_rel_err[i]), fmt_num(r$worst_ulp_err[i]))
      }
    ))
  }
  section <- function(r, title, blurb, n = 10L) {
    rule(title)
    cat(blurb, "\n\n", sep = "")
    if (!nrow(r)) {
      cat("  none\n")
      return(invisible(NULL))
    }
    r <- r[order(-r$worst_rel_err), , drop = FALSE]
    for (i in seq_len(min(nrow(r), n))) {
      line(r, i)
    }
    if (nrow(r) > n) cat(sprintf("  ... and %d more.\n", nrow(r) - n))
  }

  ## Two different kinds of finding. Ranked together, whichever is rarer gets
  ## buried: 56 unexplained regions once filled every slot and the largest
  ## finite errors never appeared at all.
  section(
    res[unexp, , drop = FALSE],
    sprintf("DISAGREEMENTS NOTHING EXPLAINS (%d)", sum(unexp)),
    paste0(
      "Every sampled input across these ranges differs from base R, for a\n",
      "reason the spec does not account for. The range is shown."
    )
  )
  f <- res[!unexp & res$worst_rel_err > 0, , drop = FALSE]
  section(
    f,
    sprintf("LARGEST ERRORS (%d of %d results differ at all)", nrow(f), nrow(res)),
    paste0(
      "base R is the reference, not the truth; it is sometimes the weaker\n",
      "implementation, so check which side is right before acting."
    )
  )
  cat(sprintf(
    "\n  %d of %d results are bit-identical to base R at every sampled input.\n",
    sum(res$worst_rel_err == 0 & !unexp, na.rm = TRUE),
    nrow(res)
  ))

  ## ---- what to do next -----------------------------------------------------
  rule("NEXT")
  top <- res[order(!unexp, -res$worst_rel_err), , drop = FALSE][1L, , drop = FALSE]
  cat("  To see what a sweep actually measured \u2014 sample counts, the error\n")
  cat("  distribution, where behaviour changes, the worst inputs:\n")
  cat(sprintf("    Rscript run.R report --filter spec=%s\n", top$spec))
  cat("    Rscript run.R browse\n")
  if (!is.null(runs) && nrow(runs) > 1L) {
    cat("\n  To see what changed since the previous run:\n    Rscript run.R diff\n")
  }
  if (all(all_depths$depth == "smoke")) {
    cat("\n  Everything so far is at 'smoke' depth, the coarsest:\n")
    cat("    Rscript run.R run --depth quick --jobs 8\n")
  }
  cat("\n")
  invisible(res)
}


cmd_report <- function(opt) {
  dir <- store_dir(opt$store)
  all <- latest_results(dir)
  if (is.null(all)) {
    stop("store is empty; run a sweep first", call. = FALSE)
  }
  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, opt$backends), opt$filter, extra = "output")
  res <- deepest_per_cell(all[all$cell_id %in% g$cell_id, , drop = FALSE], names(DEPTHS))
  res <- filter_results(res, opt$filter)
  if (!nrow(res)) {
    stop("no results for that filter; run a sweep first", call. = FALSE)
  }

  detail <- store_read(dir, "detail")
  ranges <- store_read(dir, "ranges")
  hist <- store_read(dir, "hist")
  bands <- store_read(dir, "bands")
  key <- function(tbl, r) {
    if (is.null(tbl)) {
      return(NULL)
    }
    tbl[tbl$run_id == r$run_id & tbl$cell_id == r$cell_id & tbl$output == r$output, , drop = FALSE]
  }

  res <- res[order(res$cell_id, res$output), , drop = FALSE]

  ## Print a summary rather than every page until the filter has narrowed
  ## things enough that the pages are readable. The filter itself decides the
  ## level: whichever axis still has more than one value is the one worth
  ## summarising by, so narrowing walks down the tree with no separate notion
  ## of "depth" to keep in sync.
  axis <- next_axis(res)
  if (!is.null(axis) && nrow(res) > opt$pages) {
    cat(sprintf(
      "\n%d results match%s. Summarising by %s:\n\n",
      nrow(res),
      if (nzchar(opt$filter)) sprintf(" '%s'", opt$filter) else "",
      DRILL_LABEL[[axis]]
    ))
    sm <- print_summary(res, axis)
    cat(sprintf("\nNarrow with --filter %s=<value>, e.g.\n", axis))
    worst <- sm$key[which.max(pmax(sm$f32_worst, sm$f64_worst, na.rm = TRUE))]
    nf <- if (nzchar(opt$filter)) paste0(opt$filter, ",") else ""
    cat(sprintf("  Rscript run.R report --filter '%s%s=%s'\n", nf, axis, worst))
    cat("  Rscript run.R browse    (to step through interactively)\n")
    return(invisible(sm))
  }

  for (i in seq_len(nrow(res))) {
    r <- res[i, , drop = FALSE]
    row <- g[g$cell_id == r$cell_id, , drop = FALSE][1L, , drop = FALSE]
    report_cell(
      specs[[row$spec]],
      row,
      r,
      key(detail, r),
      key(ranges, r),
      key(hist, r),
      key(bands, r)
    )
  }
  cat(sprintf("%d cell result(s).\n", nrow(res)))
  invisible(res)
}


## ---- interactive drill-down ------------------------------------------------
##
## A numbered menu rather than arrow keys. R has no usable TUI library, and
## raw-mode key capture means driving `stty`, which is brittle across terminals
## and breaks the moment output is piped.
##
## Every screen is drawn by print_summary(), the same function `report` uses,
## so the two can never disagree about what a level looks like.
cmd_browse <- function(opt) {
  dir <- store_dir(opt$store)
  all <- latest_results(dir)
  if (is.null(all)) {
    stop("store is empty; run a sweep first", call. = FALSE)
  }
  specs <- load_specs()
  grid <- build_grid(specs, opt$backends)
  base <- deepest_per_cell(all[all$cell_id %in% grid$cell_id, , drop = FALSE], names(DEPTHS))
  if (!nrow(base)) {
    stop("no results yet; run a sweep first", call. = FALSE)
  }

  con <- file("stdin")
  open(con)
  on.exit(close(con))

  detail <- store_read(dir, "detail")
  ranges <- store_read(dir, "ranges")
  hist <- store_read(dir, "hist")
  bands <- store_read(dir, "bands")
  keyf <- function(tbl, r) {
    if (is.null(tbl)) {
      return(NULL)
    }
    tbl[tbl$run_id == r$run_id & tbl$cell_id == r$cell_id & tbl$output == r$output, , drop = FALSE]
  }

  path <- list()
  page <- 0L # which page of the worst-inputs list is on screen
  PAGE <- 20L

  repeat {
    res <- base
    for (p in path) {
      res <- res[res[[p$axis]] == p$value, , drop = FALSE]
    }

    cat("\n", strrep("─", 72), "\n", sep = "")
    ## Name every axis that is pinned, by choice or because the whole store has
    ## only one value for it, so the reader always knows what they are looking
    ## at. An axis that never varied is left out as noise.
    pinned <- character(0)
    for (a in DRILL) {
      if (length(unique(base[[a]])) < 2L) {
        next
      }
      u <- unique(res[[a]])
      if (length(u) == 1L) pinned <- c(pinned, u)
    }
    crumb <- if (length(pinned)) paste(pinned, collapse = "  ›  ") else "all functions"
    cat(sprintf("  %s   (%d result%s)\n", crumb, nrow(res), if (nrow(res) == 1L) "" else "s"))
    cat(strrep("─", 72), "\n", sep = "")

    choices <- character(0)
    leaf <- NULL
    if (!nrow(res)) {
      cat("\n  No results here.\n")
    } else if (nrow(res) == 1L) {
      leaf <- res
      cat("\n")
      report_cell(
        specs[[grid$spec[grid$cell_id == leaf$cell_id][1L]]],
        grid[grid$cell_id == leaf$cell_id, , drop = FALSE][1L, , drop = FALSE],
        leaf,
        keyf(detail, leaf),
        keyf(ranges, leaf),
        keyf(hist, leaf),
        keyf(bands, leaf)
      )

      ## The store keeps the worst 1000 inputs; `d` walks through them.
      dl <- keyf(detail, leaf)
      if (!is.null(dl) && nrow(dl) > 0L && page > 0L) {
        dl <- dl[order(-dl$rel_err), , drop = FALSE]
        from <- (page - 1L) * PAGE + 1L
        if (from > nrow(dl)) {
          page <- 1L
          from <- 1L
        }
        to <- min(from + PAGE - 1L, nrow(dl))
        cat(sprintf("  WORST INPUTS %d-%d of %d\n", from, to, nrow(dl)))
        cat(sprintf("  %-6s %-24s %-20s %12s %12s\n", "rank", "x", "bits", "rel err", "ulp"))
        for (j in from:to) {
          cat(sprintf(
            "  %-6d %-24s %-20s %12s %12s\n",
            j,
            exact_num(dl$x[j]),
            dl$bits[j],
            fmt_num(dl$rel_err[j]),
            fmt_num(dl$ulp_err[j])
          ))
        }
        cat("\n")
      }
    } else {
      axis <- next_axis(res)
      cat("\n")
      sm <- print_summary(res, axis)
      choices <- sm$key
      cat("\n")
      for (i in seq_along(choices)) {
        cat(sprintf("   %2d) %s\n", i, choices[i]))
      }
    }

    opts <- c(
      if (!is.null(leaf)) sprintf("d) %s worst inputs", if (page > 0L) "more" else "list"),
      if (length(path)) "b) back",
      "q) quit"
    )
    cat(sprintf("\n   %s\n\n> ", paste(opts, collapse = "   ")))
    ans <- readLines(con, n = 1L)
    if (!length(ans)) {
      cat("\n")
      break
    }
    ans <- tolower(trimws(ans))

    if (ans == "q") {
      cat("\n")
      break
    } else if (ans == "d" && !is.null(leaf)) {
      page <- page + 1L
    } else if (ans == "b") {
      if (length(path)) {
        path <- path[-length(path)]
      }
      page <- 0L
    } else if (nzchar(ans)) {
      k <- suppressWarnings(as.integer(ans))
      if (!is.na(k) && k >= 1L && k <= length(choices)) {
        path <- c(path, list(list(axis = next_axis(res), value = choices[k])))
        page <- 0L
      } else {
        cat("   ? enter a number from the list, or one of the letters shown\n")
      }
    }
  }
  invisible(NULL)
}


cmd_selftest <- function(opt) {
  opt$filter <- "spec=selftest"
  opt$depth <- "smoke"
  opt$quiet <- TRUE
  run_id <- cmd_run(opt)

  res <- store_read(store_dir(opt$store), "results")
  res <- res[res$run_id == run_id, , drop = FALSE]
  get <- function(id, out = "value") {
    r <- res[res$cell_id == id & res$output == out, , drop = FALSE]
    if (nrow(r) != 1L) {
      stop("selftest: expected exactly one row for ", id, "/", out, call. = FALSE)
    }
    r
  }
  check <- function(label, ok) {
    cat(sprintf("  %-4s %s\n", if (isTRUE(ok)) "ok" else "FAIL", label))
    isTRUE(ok)
  }

  ## Scoring, checked directly on the machine that will run the sweep: the f32
  ## rounding edges rest on the platform's double-to-float conversion, which C
  ## leaves implementation-defined out of range, so they are verified, not assumed.
  fmax <- (2 - 2^-23) * 2^127
  smin <- 2^-149
  sc <- function(f, g, dtype) score_pair(f, g, dtype)
  negzero <- function(x) x == 0 & 1 / x < 0
  cat("\nscoring:\n")
  sp <- c(
    check("f32: overflow threshold rounds to +-Inf, the tie included",
      identical(as_f32(c(1e40, -1e40, fmax + 2^103, fmax + 2^103 * 0.99)), c(Inf, -Inf, Inf, fmax))),
    check("f32: below half the smallest subnormal rounds to 0, sign kept, the tie included",
      all(as_f32(c(1e-46, smin / 2)) == 0) && negzero(as_f32(-1e-46)) && as_f32(smin * 0.51) == smin),
    check("f32 overflow that is correctly rounded is a match, its error still infinite", {
      x <- sc(Inf, 1e40, "f32"); x$rounded && !x$bad && is.infinite(x$rel) }),
    check("f32 underflow that is correctly rounded is a match, its error still 1", {
      x <- sc(0, 1e-46, "f32"); x$rounded && !x$bad && x$rel == 1 }),
    check("f32 overflow where the value fits is a failure", {
      x <- sc(Inf, 1e30, "f32"); !x$rounded && x$bad }),
    check("f64 flushing a subnormal is not correct rounding: a finite error of 1", {
      x <- sc(0, 1e-310, "f64"); !x$rounded && !x$bad && x$rel == 1 }),
    check("f64 -Inf against a finite reference is a failure", sc(-Inf, 2.5, "f64")$bad),
    check("f64 difference overflow is measured, not infinite", {
      x <- sc(-1e308, 1e308, "f64"); !x$bad && x$rel == 2 }),
    check("f64 relative error overflowing a tiny reference is a failure", sc(1e-10, 5e-324, "f64")$bad),
    check("an ordinary error is unchanged", sc(1.0000001, 1, "f64")$rel == 1.0000001 - 1),
    check("nothing is left with a non-finite score and no route", {
      f <- c(Inf, -Inf, -1e308, 1e-10, NaN, 1e-300, 0, 2, Inf, NaN)
      g <- c(2.5, 2.5, 1e308, 5e-324, 2.5, 0, 1e-310, NaN, -Inf, NaN)
      x <- sc(f, g, "f64")
      all(is.finite(x$rel) | x$bad | x$rounded)
    })
  )

  cat("\nassertions:\n")
  p <- "selftest/anvl/%s/%s/%s/broken=%s"
  ok <- c(sp,
    check(
      "clean f64 value reproduces the reference exactly",
      get(sprintf(p, "f64", "value", "clean", "FALSE"))$worst_rel_err == 0
    ),
    check(
      "a one-ulp nudge is measured as exactly one ulp",
      get(sprintf(p, "f64", "value", "nudged", "FALSE"))$worst_ulp_err == 1
    ),
    check(
      "a zeroed tail is measured as relative error 1",
      get(sprintf(p, "f64", "value", "clean", "TRUE"))$worst_rel_err == 1
    ),
    check(
      "the same break against an infinite reference leaves an unclassified region",
      get(sprintf(p, "f32", "value", "clean", "TRUE"))$n_runs_unclassified == 1
    ),
    check(
      "a 1e-6 gradient error is ~4.5e9 ulp in f64",
      get(sprintf(p, "f64", "grad", "clean", "TRUE"), "scale")$worst_ulp_err > 4e9
    ),
    check(
      "the same error is ~8 ulp in f32",
      get(sprintf(p, "f32", "grad", "clean", "TRUE"), "scale")$worst_ulp_err > 8
    ),
    check(
      "the untouched gradient output is untouched",
      get(sprintf(p, "f64", "grad", "clean", "TRUE"), "x")$worst_rel_err == 0
    ),
    check("no cell errored", all(is.na(res$error)))
  )
  cat(sprintf("\n%d/%d assertions passed\n", sum(ok), length(ok)))
  if (!all(ok)) {
    quit(status = 1L)
  }
  invisible(TRUE)
}

## ---- comparing two runs -----------------------------------------------------
##
## "Did my fix help?" `status` cannot answer it: a result that got better simply
## stays PASS, so an improvement is invisible and only a regression past a
## recorded bound ever shows.
##
## The sweep is deterministic -- fixed seed, fixed stride, same inputs every
## time -- so two runs of the same cell at the same depth on the same machine
## are bit-identical unless the code changed. That is a strong enough property
## to drop the usual fuzzy "meaningfully different" threshold entirely: any
## difference at all is a real one, and exact equality is a real "no change".

## Every result with its run's timestamp, so "the previous result for this
## cell" is well defined. Deliberately not deepest_per_cell(): the whole point
## here is the history that collapses away.
results_with_time <- function(dir) {
  res <- store_read(dir, "results")
  if (is.null(res)) {
    return(NULL)
  }
  runs <- store_read(dir, "runs")
  if (is.null(runs)) {
    return(NULL)
  }
  merge(res, runs[c("run_id", "started_at", "anvl_sha", "branch")], by = "run_id", all.x = TRUE)
}

## Compare like with like. A smoke result and a full result of the same cell
## sample different inputs, so pairing them across depths would report the
## extra coverage as a regression.
diff_key <- function(d) paste(d$cell_id, d$output, d$platform_key, d$depth, sep = "\r")

cmd_diff <- function(opt) {
  dir <- store_dir(opt$store)
  all <- results_with_time(dir)
  if (is.null(all) || !nrow(all)) {
    stop("store is empty; run a sweep first", call. = FALSE)
  }
  runs <- store_read(dir, "runs")
  runs <- runs[order(runs$started_at, decreasing = TRUE), , drop = FALSE]

  specs <- load_specs(include_selftest = grepl("selftest", opt$filter))
  g <- apply_filter(build_grid(specs, opt$backends), opt$filter, extra = "output")
  matching <- function(id) {
    r <- all[all$run_id == id & all$cell_id %in% g$cell_id, , drop = FALSE]
    filter_results(r, opt$filter)
  }

  ## The newest run that actually contains matching results, not simply the
  ## newest run: a `selftest` or single-cell sweep in between would otherwise
  ## be chosen and then reported as empty.
  if (!is.null(opt$to)) {
    to_id <- opt$to
    if (!to_id %in% all$run_id) {
      stop("no results for run '", to_id, "'", call. = FALSE)
    }
    now <- matching(to_id)
    if (!nrow(now)) {
      stop("run '", to_id, "' has no results matching the filter", call. = FALSE)
    }
  } else {
    now <- NULL
    for (id in runs$run_id) {
      cand <- matching(id)
      if (nrow(cand)) {
        to_id <- id
        now <- cand
        break
      }
    }
    if (is.null(now)) {
      stop("no run contains results matching that filter", call. = FALSE)
    }
  }

  ## The comparison point: an explicit --from, or otherwise the most recent
  ## earlier result for each cell, which handles partial re-runs without
  ## needing them to line up as whole runs.
  t_now <- now$started_at[1L]
  past <- all[all$started_at < t_now, , drop = FALSE]
  if (!is.null(opt$from)) {
    past <- past[past$run_id == opt$from, , drop = FALSE]
    if (!nrow(past)) stop("no earlier results for run '", opt$from, "'", call. = FALSE)
  }
  past <- past[order(past$started_at, decreasing = TRUE), , drop = FALSE]
  past <- past[!duplicated(diff_key(past)), , drop = FALSE]

  i <- match(diff_key(now), diff_key(past))
  fresh <- is.na(i)
  before <- past[i, , drop = FALSE]

  cat("\nanvl distribution sweeps \u2014 diff\n")
  if (any(!fresh)) {
    b <- before[!fresh, , drop = FALSE]
    cat(sprintf(
      "  from  %s  anvl %s  (%s)\n",
      if (length(unique(b$run_id)) == 1L) unique(b$run_id) else sprintf("%d earlier runs", length(unique(b$run_id))),
      paste(unique(substr(b$anvl_sha, 1L, 7L)), collapse = ","),
      paste(unique(b$depth), collapse = ",")
    ))
  }
  cat(sprintf(
    "  to    %s  anvl %s  (%s)\n",
    to_id,
    substr(now$anvl_sha[1L], 1L, 7L),
    paste(unique(now$depth), collapse = ",")
  ))
  if (all(fresh)) {
    ## Nothing to compare against is the normal state after a store reset or a
    ## first run, and reads as an error if the screen does not say so.
    cat(sprintf(
      paste0(
        "\n  Nothing to compare against: all %d result(s) are the first of their\n",
        "  cell at this depth. Re-run after a code change to see what moved.\n\n"
      ),
      nrow(now)
    ))
    return(invisible(NULL))
  }
  cat("\n  The sweep is deterministic, so on one machine any difference below was\n")
  cat("  caused by the code, not by measurement noise.\n")

  ## direction, per result
  same <- !fresh &
    (now$worst_rel_err == before$worst_rel_err | (is.na(now$worst_rel_err) & is.na(before$worst_rel_err))) &
    now$n_runs_unclassified == before$n_runs_unclassified
  worse <- !fresh &
    !same &
    (now$worst_rel_err > before$worst_rel_err | now$n_runs_unclassified > before$n_runs_unclassified)
  better <- !fresh & !same & !worse

  line <- function(k) {
    what <- if (now$kind[k] == "value") "value" else sprintf("d/d%s", now$output[k])
    row <- g[g$cell_id == now$cell_id[k], , drop = FALSE][1L, , drop = FALSE]
    cat(sprintf(
      "\n  %-9s %-7s %-4s %s\n",
      now$spec[k],
      what,
      now$dtype[k],
      short_cell(row)
    ))
    a <- before$worst_rel_err[k]
    z <- now$worst_rel_err[k]
    note <- if (is.na(a) || is.na(z)) {
      ""
    } else if (a == 0 && z > 0) {
      "   (was bit-identical everywhere)"
    } else if (z == 0 && a > 0) {
      "   (now bit-identical everywhere)"
    } else if (a > 0 && z > 0 && is.finite(z / a)) {
      sprintf("   (%.3gx %s)", max(z / a, a / z), if (z > a) "worse" else "better")
    } else {
      ""
    }
    if (!isTRUE(all.equal(a, z))) {
      cat(sprintf("    worst rel err        %s -> %s%s\n", fmt_num(a), fmt_num(z), note))
    }
    if (before$worst_ulp_err[k] != now$worst_ulp_err[k]) {
      cat(sprintf(
        "    worst ulp            %s -> %s\n",
        fmt_num(before$worst_ulp_err[k]),
        fmt_num(now$worst_ulp_err[k])
      ))
    }
    if (before$n_runs_unclassified[k] != now$n_runs_unclassified[k]) {
      cat(sprintf(
        "    unexplained regions  %d -> %d\n",
        before$n_runs_unclassified[k],
        now$n_runs_unclassified[k]
      ))
    }
  }

  section <- function(which, title) {
    k <- which(which)
    if (!length(k)) {
      return(invisible(NULL))
    }
    rule(sprintf("%s (%d)", title, length(k)))
    k <- k[order(
      -abs(
        log10(pmax(now$worst_rel_err[k], 1e-300)) -
          log10(pmax(before$worst_rel_err[k], 1e-300))
      )
    )]
    for (j in utils::head(k, 15L)) {
      line(j)
    }
    if (length(k) > 15L) cat(sprintf("\n  ... and %d more.\n", length(k) - 15L))
  }

  section(worse, "REGRESSED")
  section(better, "IMPROVED")

  rule("SUMMARY")
  cat(sprintf("  %4d regressed\n", sum(worse)))
  cat(sprintf("  %4d improved\n", sum(better)))
  cat(sprintf("  %4d unchanged (bit-identical to the earlier run)\n", sum(same)))
  cat(sprintf("  %4d had no earlier result to compare against\n", sum(fresh)))
  cat("\n")
  invisible(data.frame(
    cell_id = now$cell_id,
    output = now$output,
    change = ifelse(fresh, "new", ifelse(same, "same", ifelse(worse, "regressed", "improved")))
  ))
}

## ---- publishing a snapshot --------------------------------------------------
##
## The store accumulates: every run appends, and queries pick the latest per
## cell. A published artifact must instead be a single coherent snapshot --
## one result per cell, output and platform -- with the provenance that
## produced it, laid out so a browser can fetch only the part it needs.
##
## Layout, under --out -- one artifact per (anvl version, platform), holding
## every backend swept, so anvl and its JAX twin can be compared in one place:
##
##   manifest.json     what is here: schema version, platform, specs, depths,
##                     row counts. Small, fetched first, and readable without a
##                     Parquet reader so it can drive navigation on its own.
##   runs.parquet      the environment fingerprint of every run included
##   summary.parquet   the results table for every cell of every function --
##                     a few hundred rows, enough to drive the whole index and
##                     the cross-function overview
##   detail.parquet    the worst inputs, per binade
##   bands.parquet     the per-binade profile (unmerged; merged on render)
##   hist.parquet      the error distribution
##   ranges.parquet    the no-finite-error regions
##   categories.parquet  per-result figures split by input class: normal,
##                     zero & subnormal, out of support, Inf & NaN
##
## One file per *table*, covering every function -- not one per function. The
## overview page summarises all functions at once, so splitting by function
## would mean fetching every piece anyway, in more requests, and would make
## switching platform a download of many files rather than one artifact.
##
## detail and bands are sorted by cell and their row groups aligned to cell
## boundaries. Parquet can only skip whole row groups, and nanoparquet defaults
## to one row group of ten million rows -- which would mean reading the entire
## file to drill into any single cell. Aligned, a reader fetches the footer and
## then just the groups it needs.

## Minimal JSON writer, so the manifest costs no extra dependency. Only the
## shapes used below are supported: named lists, atomic vectors, data frames.
## Non-finite numerics become null -- JSON cannot represent them, which is
## exactly why the measurements themselves travel as Parquet and never as JSON.
to_json <- function(x, indent = 0L) {
  pad <- strrep(" ", indent)
  esc <- function(s) {
    s <- gsub("\\", "\\\\", s, fixed = TRUE)
    s <- gsub('"', '\\"', s, fixed = TRUE)
    gsub("[[:cntrl:]]", "", s)
  }
  scalar <- function(v) {
    if (is.na(v)) {
      return("null")
    }
    if (is.logical(v)) {
      return(if (v) "true" else "false")
    }
    if (is.numeric(v)) {
      return(if (is.finite(v)) format(v, scientific = FALSE, trim = TRUE) else "null")
    }
    paste0('"', esc(as.character(v)), '"')
  }
  if (is.data.frame(x)) {
    rows <- vapply(
      seq_len(nrow(x)),
      function(i) to_json(as.list(x[i, , drop = FALSE]), indent + 2L),
      ""
    )
    return(paste0("[\n", paste0(strrep(" ", indent + 2L), rows, collapse = ",\n"), "\n", pad, "]"))
  }
  if (is.list(x)) {
    if (!length(x)) {
      return("{}")
    }
    kv <- vapply(
      names(x),
      function(n) paste0(pad, "  \"", esc(n), "\": ", to_json(x[[n]], indent + 2L)),
      ""
    )
    return(paste0("{\n", paste(kv, collapse = ",\n"), "\n", pad, "}"))
  }
  ## A field that is conceptually a list stays an array even with one element,
  ## so a reader never has to handle both shapes for the same key.
  if (length(x) == 1L && !inherits(x, "json_array")) {
    return(scalar(x))
  }
  paste0("[", paste(vapply(x, scalar, ""), collapse = ", "), "]")
}

json_array <- function(x) structure(x, class = c("json_array", class(x)))

cmd_export <- function(opt) {
  if (is.null(opt$out)) {
    stop("export needs --out <dir>", call. = FALSE)
  }
  dir <- store_dir(opt$store)
  all <- latest_results(dir)
  if (is.null(all)) {
    stop("store is empty; run a sweep first", call. = FALSE)
  }
  specs <- load_specs()
  ## Export what the store holds. Falling back to run's anvl-only default here
  ## once published a full anvl+JAX sweep with every JAX result missing.
  backends <- if (isTRUE(opt$backends_given)) opt$backends else sort(unique(all$backend))
  g <- apply_filter(build_grid(specs, backends), opt$filter, extra = "output")
  res <- deepest_per_cell(all[all$cell_id %in% g$cell_id, , drop = FALSE], names(DEPTHS))
  res <- filter_results(res, opt$filter)
  if (!nrow(res)) {
    stop("no results to export for that filter", call. = FALSE)
  }

  ## Each cell's support, resolved from its params and flags exactly as the
  ## sweep resolved it. Carried on the summary, so a reader can shade the part
  ## of the axis that is off the support without asking the harness.
  cells <- g[match(unique(res$cell_id), g$cell_id), , drop = FALSE]
  bounds <- lapply(seq_len(nrow(cells)), function(i) {
    row <- cells[i, ]
    spec <- specs[[row$spec]]
    if (is.null(spec$support)) {
      return(c(-Inf, Inf))
    }
    cf <- cell_functions(spec, row)
    spec$support(cf$params, cf$flags)
  })
  support <- data.frame(
    cell_id = cells$cell_id,
    support_lo = vapply(bounds, `[`, 0, 1L),
    support_hi = vapply(bounds, `[`, 0, 2L)
  )
  res$support_lo <- support$support_lo[match(res$cell_id, support$cell_id)]
  res$support_hi <- support$support_hi[match(res$cell_id, support$cell_id)]

  out <- normalizePath(opt$out, mustWork = FALSE)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  runs <- store_read(dir, "runs")
  runs <- runs[runs$run_id %in% unique(res$run_id), , drop = FALSE]

  ## Keyed on exactly the rows kept above, so a detail row from a superseded
  ## run can never leak in beside a newer summary row.
  keep <- paste(res$run_id, res$cell_id, res$output)
  pick <- function(tbl) {
    x <- store_read(dir, tbl)
    if (is.null(x) || !nrow(x)) {
      return(NULL)
    }
    x[paste(x$run_id, x$cell_id, x$output) %in% keep, , drop = FALSE]
  }

  ## Row groups aligned to cell boundaries: sort, then start a new group
  ## wherever the cell changes, so a reader can fetch one cell's rows alone.
  write_by_cell <- function(x, path) {
    x <- x[order(x$cell_id, x$output), , drop = FALSE]
    rownames(x) <- NULL
    starts <- which(!duplicated(x$cell_id))
    nanoparquet::write_parquet(x, path, row_groups = as.integer(starts))
    nrow(x)
  }

  nanoparquet::write_parquet(runs, file.path(out, "runs.parquet"))
  nanoparquet::write_parquet(res[order(res$cell_id, res$output), , drop = FALSE], file.path(out, "summary.parquet"))

  counts <- c(runs = nrow(runs), summary = nrow(res))
  kept <- list()
  for (tbl in c("detail", "bands", "hist", "ranges")) {
    x <- pick(tbl)
    if (is.null(x) || !nrow(x)) {
      next
    }
    kept[[tbl]] <- x
    counts[tbl] <- write_by_cell(x, file.path(out, paste0(tbl, ".parquet")))
  }

  ## Per-result figures split by input class (see input_class()). A few rows
  ## per result, so the overview can show normal-input accuracy without ever
  ## fetching `bands`.
  if (!is.null(kept$bands) && !is.null(kept$detail)) {
    cats <- category_table(kept$bands, kept$detail, support)
    nanoparquet::write_parquet(cats, file.path(out, "categories.parquet"))
    counts["categories"] <- nrow(cats)
  }

  manifest <- list(
    schema_version = SCHEMA_VERSION,
    exported_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    platforms = json_array(sort(unique(res$platform_key))),
    backends = json_array(sort(unique(res$backend))),
    devices = json_array(sort(unique(res$device))),
    depths = json_array(sort(unique(res$depth))),
    specs = json_array(sort(unique(res$spec))),
    n_cells = length(unique(res$cell_id)),
    n_results = nrow(res),
    anvl_version = json_array(sort(unique(runs$anvl_version))),
    anvl_sha = json_array(sort(unique(runs$anvl_sha))),
    files = data.frame(table = names(counts), rows = as.integer(counts))
  )
  writeLines(to_json(manifest), file.path(out, "manifest.json"))

  files <- list.files(out, recursive = TRUE, full.names = TRUE)
  cat(sprintf(
    "exported %d results across %d cell(s) to %s\n",
    nrow(res),
    length(unique(res$cell_id)),
    out
  ))
  cat(sprintf(
    "  platforms: %s | backends: %s | depth: %s | %.1f MB in %d file(s)\n",
    paste(manifest$platforms, collapse = ", "),
    paste(manifest$backends, collapse = ", "),
    paste(manifest$depths, collapse = ", "),
    sum(file.size(files)) / 1024^2,
    length(files)
  ))
  invisible(out)
}

cmd_merge <- function(opt) {
  if (is.null(opt$from)) {
    stop("merge needs --from <dir>", call. = FALSE)
  }
  into <- store_dir(opt$store)
  store_init(into)
  n <- store_merge(normalizePath(opt$from, mustWork = TRUE), into)
  cat(sprintf("merged %d new part file(s) into %s\n", n, into))
}

## ---- dispatch --------------------------------------------------------------

main <- function() {
  a <- parse_args(commandArgs(trailingOnly = TRUE))
  switch(
    a$cmd,
    list = cmd_list(a$opt),
    run = cmd_run(a$opt),
    status = cmd_status(a$opt),
    selftest = cmd_selftest(a$opt),
    report = cmd_report(a$opt),
    browse = cmd_browse(a$opt),
    diff = cmd_diff(a$opt),
    export = cmd_export(a$opt),
    merge = cmd_merge(a$opt),
    stop(
      "unknown command '",
      a$cmd,
      "'; expected list, run, report, browse, diff, export, status, selftest or merge",
      call. = FALSE
    )
  )
}

if (!interactive()) {
  main()
}
