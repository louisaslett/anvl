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
  ## The implementation receives its parameters at the cell's precision: anvl
  ## converts a bare `min = -pi` to f32 for an f32 argument. The reference is
  ## evaluated with the parameters exactly as received, so a comparison
  ## measures the distribution function, not parameter conversion. Without
  ## this, f32(-pi) lies below the double -pi and every f32 uniform cell
  ## "fails" at its own boundary by construction. Domain, support and branch
  ## points are taken from the same parameters.
  rparams <- if (row$dtype == "f32") {
    lapply(params, function(v) if (is.numeric(v)) as_f32(v) else v)
  } else {
    params
  }

  if (row$kind == "value") {
    fn <- if (row$backend == "jax") spec$jax_value else spec$value
    list(
      fun = function(x) list(value = fn(x, row$dtype, params, flags)),
      ref = function(x) list(value = spec$ref_value(x, rparams, flags)),
      outputs = "value",
      params = params,
      ref_params = rparams,
      flags = flags
    )
  } else {
    fn <- if (row$backend == "jax") spec$jax_grad else spec$grad
    list(
      fun = function(x) fn(x, row$dtype, params, flags),
      ref = function(x) spec$ref_grad(x, rparams, flags),
      outputs = spec$grad_wrt,
      params = params,
      ref_params = rparams,
      flags = flags
    )
  }
}

run_cell <- function(spec, row, opt, pv, dir) {
  cf <- cell_functions(spec, row)
  key <- cell_key(row$cell_id)

  ## Where the function is defined, where its distribution lives, and where
  ## anvl switches algorithm -- all at the parameters the implementation
  ## receives. Branch points are anvl's own, so a JAX cell does not get them.
  domain <- if (is.null(spec$domain)) c(-Inf, Inf) else spec$domain(cf$ref_params, cf$flags)
  support <- if (is.null(spec$support)) NULL else spec$support(cf$ref_params, cf$flags)
  branch <- if (row$backend == "anvl" && !is.null(spec$branch_points)) {
    spec$branch_points(cf$ref_params, cf$flags, row$dtype)
  }
  pts <- exact_points(row$dtype, domain, support, branch)

  t0 <- Sys.time()
  out <- tryCatch(
    {
      pr <- run_points(cf$fun, cf$ref, row$dtype, cf$outputs, pts, domain)
      sw <- run_sweep(
        cf$fun,
        cf$ref,
        row$dtype,
        opt$depth,
        cf$outputs,
        progress = !opt$quiet,
        topk = opt$topk,
        domain = domain,
        ctx = attr(pr, "context")
      )
      attr(pr, "context") <- NULL
      attr(sw, "points") <- pr
      sw
    },
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
        n_zero_sign = NA_real_,
        n_flushed = NA_real_,
        n_flushed_zero_error = NA_real_,
        n_out_normal = NA_real_,
        n_out_normal_identical = NA_real_,
        worst_out_normal = NA_real_,
        n_inf_runs = NA_integer_,
        n_runs_unclassified = 0L,
        n_regions_backend = 0L,
        n_regions_boundary = 0L,
        n_regions_domain = 0L,
        n_failing_failure = 0,
        n_failing_backend = 0,
        n_failing_boundary = 0,
        n_failing_domain = 0,
        worst_rel_err = NA_real_,
        worst_ulp_err = NA_real_,
        worst_x = NA_real_,
        worst_bits = NA_character_,
        worst_value = NA_real_,
        worst_reference = NA_real_,
        unexplained_from = NA_real_,
        unexplained_to = NA_real_,
        as.data.frame(point_summary(NO_POINTS))[rep(1L, 1L), ],
        elapsed_sec = NA_real_,
        error = as.character(out)
      )
    )
    store_write(dir, "results", pv$run_id, key, res)
    return(list(status = "ERROR", msg = as.character(out)))
  }

  store_write(
    dir,
    "points",
    pv$run_id,
    key,
    cbind(data.frame(run_id = pv$run_id, cell_id = row$cell_id), attr(out, "points"))
  )

  res_rows <- list()
  pts <- attr(out, "points")
  for (o in cf$outputs) {
    r <- out[[o]]
    rs <- region_summary(r$ranges)
    ps <- point_summary(pts[pts$output == o, , drop = FALSE])
    if (nrow(r$kinds)) {
      store_write(
        dir,
        "kinds",
        pv$run_id,
        paste0(key, "-", o),
        cbind(data.frame(run_id = pv$run_id, cell_id = row$cell_id, output = o), r$kinds)
      )
    }
    if (nrow(r$ranges)) {
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
        n_zero_sign = r$summary$n_zero_sign,
        n_flushed = r$summary$n_flushed,
        n_flushed_zero_error = r$summary$n_flushed_zero_error,
        n_out_normal = r$summary$n_out_normal,
        n_out_normal_identical = r$summary$n_out_normal_identical,
        worst_out_normal = r$summary$worst_out_normal,
        n_inf_runs = r$summary$n_inf_runs,
        ## regions whose category is "failure" -- the name predates categories
        n_runs_unclassified = rs$n_runs_unclassified,
        n_regions_backend = rs$n_regions_backend,
        n_regions_boundary = rs$n_regions_boundary,
        n_regions_domain = rs$n_regions_domain,
        n_failing_failure = rs$n_failing_failure,
        n_failing_backend = rs$n_failing_backend,
        n_failing_boundary = rs$n_failing_boundary,
        n_failing_domain = rs$n_failing_domain,
        worst_rel_err = r$summary$worst_rel_err,
        worst_ulp_err = r$summary$worst_ulp_err,
        ## The input that produced the worst error, and where the first
        ## unexplained disagreement starts, both carried on the summary row so
        ## that `status` can say "how bad, and where" without a second lookup.
        worst_x = r$summary$worst_x,
        worst_bits = r$summary$worst_bits,
        worst_value = r$summary$worst_value,
        worst_reference = r$summary$worst_reference,
        unexplained_from = rs$unexplained_from,
        unexplained_to = rs$unexplained_to,
        ## the exact points, summarised apart from the sweep's counts
        as.data.frame(ps),
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
  ## Every category is shown, and exact points count alongside regions: a
  ## failure only at p = 1, or only at a point the f64 sweep never lands on,
  ## must reach this screen. Only undefined-domain conventions are set aside,
  ## and the closing tally says how many results that applies to.
  st <- result_state(res)
  res$worst_any <- st$worst_any
  where_pt <- function(label, x) sprintf("exact point %s (x = %s)", label, exact_num(x))
  line <- function(r, i, show) {
    row <- g[g$cell_id == r$cell_id[i], , drop = FALSE][1L, , drop = FALSE]
    what <- if (r$kind[i] == "value") "value" else paste0("d/d", r$output[i])
    cat(sprintf(
      "  %-9s %-3s %-7s %-22s %s\n",
      r$spec[i],
      r$dtype[i],
      what,
      short_cell(row),
      show(r, i)
    ))
  }
  show_failure <- function(r, i) {
    parts <- character(0)
    if (r$n_runs_unclassified[i] > 0) {
      parts <- c(parts, sprintf(
        "%d region(s) from %s .. %s",
        r$n_runs_unclassified[i],
        fmt_num(min(r$unexplained_from[i], r$unexplained_to[i])),
        fmt_num(max(r$unexplained_from[i], r$unexplained_to[i]))
      ))
    }
    np <- r$n_points_failure[i] %||% 0
    if (!is.na(np) && np > 0) {
      parts <- c(parts, sprintf("%d point(s), first %s", np, where_pt(r$first_point_failure[i], r$first_point_failure_x[i])))
    }
    paste(parts, collapse = "; ")
  }
  show_counts <- function(regions, pts) {
    function(r, i) {
      n1 <- r[[regions]][i] %||% 0
      n2 <- r[[pts]][i] %||% 0
      sprintf("%d region(s), %d exact point(s)", n1, if (is.na(n2)) 0L else n2)
    }
  }
  show_error <- function(r, i) {
    from_pt <- (r$worst_point_rel_err[i] %||% 0) > r$worst_rel_err[i]
    sprintf(
      "rel %-10s %s",
      fmt_num(r$worst_any[i]),
      if (isTRUE(from_pt)) {
        sprintf("at %s", where_pt(r$worst_point_label[i], r$worst_point_x[i]))
      } else {
        sprintf("%s ulp", fmt_num(r$worst_ulp_err[i]))
      }
    )
  }
  section <- function(r, title, blurb, show, n = 10L) {
    rule(title)
    cat(blurb, "\n\n", sep = "")
    if (!nrow(r)) {
      cat("  none\n")
      return(invisible(NULL))
    }
    r <- r[order(-r$worst_any), , drop = FALSE]
    for (i in seq_len(min(nrow(r), n))) {
      line(r, i, show)
    }
    if (nrow(r) > n) cat(sprintf("  ... and %d more.\n", nrow(r) - n))
  }

  ## Separate kinds of finding. Ranked together, whichever is rarer gets
  ## buried: 56 failure regions once filled every slot and the largest finite
  ## errors never appeared at all.
  section(
    res[st$failing, , drop = FALSE],
    sprintf("FAILURES (%d)", sum(st$failing)),
    paste0(
      "Inputs where the result differs from base R with no finite error, for no\n",
      "cause that accounts for it: regions of the sweep, and exact points."
    ),
    show_failure
  )
  section(
    res[st$boundary, , drop = FALSE],
    sprintf("DOMAIN BOUNDARY BEHAVIOUR (%d)", sum(st$boundary)),
    paste0(
      "Results at an endpoint of the valid input domain that differ from base R's\n",
      "convention or limiting value. Visible, and not set aside."
    ),
    show_counts("n_regions_boundary", "n_points_boundary")
  )
  section(
    res[st$backend, , drop = FALSE],
    sprintf("BACKEND LIMITATIONS (%d)", sum(st$backend)),
    paste0(
      "Subnormal inputs the backend flushed to a zero whose result is itself\n",
      "correct. The platform's doing, not the function's; not set aside."
    ),
    show_counts("n_regions_backend", "n_points_backend")
  )
  f <- res[!st$failing & st$worst_any > 0, , drop = FALSE]
  section(
    f,
    sprintf("LARGEST FINITE ERRORS (%d of %d results differ at all)", sum(st$worst_any > 0), nrow(res)),
    paste0(
      "Over the sweep and the exact points. base R is the reference, not the\n",
      "truth; it is sometimes the weaker implementation, so check which side is\n",
      "right before acting."
    ),
    show_error
  )
  cat(sprintf(
    "\n  %d of %d results are bit-identical to base R, down to the sign of zero,\n  at every sampled input and every exact point.\n",
    sum(st$identical),
    nrow(res)
  ))
  if (any(st$identical_but_conventions)) {
    cat(sprintf(
      "  %d more differ only by undefined-domain conventions, which are set aside.\n",
      sum(st$identical_but_conventions)
    ))
  }
  nz <- sum(res$n_zero_sign > 0 | (res$n_points_zero_sign %||% 0) > 0, na.rm = TRUE)
  if (nz) cat(sprintf("  %d result(s) return a zero of the opposite sign somewhere.\n", nz))

  ## ---- what to do next -----------------------------------------------------
  rule("NEXT")
  top <- res[order(!st$failing, -res$worst_any), , drop = FALSE][1L, , drop = FALSE]
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
  ranges <- resolved_ranges(dir)
  points <- resolved_points(dir)
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
      key(bands, r),
      key(points, r)
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
  ranges <- resolved_ranges(dir)
  points <- resolved_points(dir)
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
        keyf(bands, leaf),
        keyf(points, leaf)
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

  ## Causes and categories, on constructed cases: every branch of the cause
  ## test, each for the reason it should fire and not merely by location.
  facts <- function(x, fx, gx, domain, zero_value, zero_fails, dtype = "f64") {
    z <- c(0, NEG_ZERO)
    ctx <- list(
      dtype = dtype, domain = domain, boundaries = domain[is.finite(domain)],
      zero_is_boundary = any(domain[is.finite(domain)] == 0),
      zero = list(v = list(value = zero_value, reference = z, validated = !zero_fails))
    )
    s <- list(rel = rep(Inf, length(x)), bad = rep(TRUE, length(x)), rounded = rep(FALSE, length(x)))
    CAUSES[sample_facts(x, fx, gx, s, ctx, "v")$cause]
  }
  sub <- 1e-310
  whole <- c(-Inf, Inf)
  unit <- c(0, 1)
  cat("\ncauses:\n")
  ca0 <- check("negative zero survives inside the compiled harness (NEG_ZERO)",
    1 / sweep_context(function(x) list(v = x), function(x) list(v = x), "f64", "v")$zero$v$value[2] == -Inf)
  ca <- c(
    check("the seven value kinds, judged at the result's precision", identical(
      value_kind(c(NaN, Inf, -Inf, 0, NEG_ZERO, 1e-40, 1, 1e-40), "f32")[1:7],
      1:7) && value_kind(1e-40, "f64") == 7L),
    check("-0 and +0 are not the same value; two NaNs are",
      !same_value(0, NEG_ZERO) && same_value(NEG_ZERO, NEG_ZERO) && same_value(NaN, NaN)),
    check("a NaN input is a nan_input failure, never excused",
      facts(NaN, 0, NaN, whole, c(1, 1), c(FALSE, FALSE)) == "nan_input"),
    check("a subnormal that behaves exactly as +0, where +0 is right, is input flushing",
      facts(sub, 5, Inf, whole, c(5, 5), c(FALSE, FALSE)) == "input_flushing"),
    check("the same subnormal, where +0 itself is wrong, inherits the error at zero",
      facts(sub, 5, Inf, whole, c(5, 5), c(TRUE, TRUE)) == "flush_inherits_zero_error"),
    check("... and is boundary behaviour when zero is a domain endpoint",
      facts(sub, 5, Inf, unit, c(5, 5), c(TRUE, TRUE)) == "domain_boundary"),
    check("a subnormal that does not behave as its signed zero is not excused as flushing",
      facts(sub, 6, Inf, whole, c(5, 5), c(FALSE, FALSE)) == "unidentified"),
    check("-0 is not +0 for the flush test: a negative subnormal must match f(-0)",
      facts(-sub, 5, Inf, whole, c(5, 7), c(FALSE, FALSE)) == "unidentified"),
    check("zero is a failure in its own right, not a subnormal",
      facts(0, 5, Inf, whole, c(5, 5), c(FALSE, FALSE)) == "zero_input"),
    check("an infinite input inside the domain is a failure",
      facts(Inf, 5, NaN, whole, c(5, 5), c(FALSE, FALSE)) == "inf_input"),
    check("outside the valid input domain is recorded as such",
      facts(c(2, -3, Inf), c(1, 1, 1), c(NaN, NaN, NaN), unit, c(5, 5), c(FALSE, FALSE)) %in% "outside_domain" |> all()),
    check("a domain endpoint is boundary behaviour",
      facts(1, 5, Inf, unit, c(5, 5), c(FALSE, FALSE)) == "domain_boundary")
  )

  ## A zero result is validated only when it is identical or correctly rounded:
  ## f(0) = 2 against g(0) = 1 has a finite error, and must not excuse the
  ## subnormals flushed onto it.
  cat("\nzero validation and flushing:\n")
  zctx <- function(f0, g0) context_from(list(v = f0), list(v = g0), "f64", "v", whole)
  zc <- zctx(c(2, 2), c(1, 1))
  zx <- c(sub, 2 * sub)
  zf <- c(2, 2)
  zg <- c(0, 1.5)
  zs <- score_pair(zf, zg, "f64")
  zfa <- sample_facts(zx, zf, zg, zs, zc, "v")
  zok <- zctx(c(1, 1), c(1, 1))
  zfo <- sample_facts(zx, c(1, 1), c(0, 1.5), score_pair(c(1, 1), c(0, 1.5), "f64"), zok, "v")
  zv <- c(
    check("a zero result with a finite 100% error is not validated", !any(zc$zero$v$validated)),
    check("a zero result one ulp off is not validated either",
      !any(zctx(c(1 + 2^-52, 1 + 2^-52), c(1, 1))$zero$v$validated)),
    check("a subnormal flushed onto that zero inherits its error: a failure, not the backend",
      CAUSES[zfa$cause[1]] == "flush_inherits_zero_error" && CAUSE_CATEGORY[["flush_inherits_zero_error"]] == "failure"),
    check("flushing onto an unvalidated zero is counted apart from flushing onto a validated one",
      !any(zfa$flushed) && all(zfa$flushed_zero_error) && all(zfo$flushed) && !any(zfo$flushed_zero_error)),
    check("the same flush onto a validated zero is input flushing",
      CAUSES[zfo$cause[1]] == "input_flushing")
  )

  cat("\nexact points and conventions:\n")
  ep <- exact_points("f32", domain = c(0, 1), support = c(-pi, 2 * pi), branch = c(mid = 0.3))
  lab <- function(l) ep$x[grepl(paste0("(^|\\+)", l, "($|\\+)"), ep$label)]
  rg <- data.frame(
    run_id = c("A", "A", "B"),
    cell_id = "s/anvl/f64/grad/p/f", output = "x", cause = "outside_domain",
    category = "failure", sign = 1, binade_from = c(1030, 1040, 1030), binade_to = c(1031, 1040, 1031))
  kd <- data.frame(
    run_id = "A",
    cell_id = "s/anvl/f64/value/p/f", output = "value", sign = 1, binade = c(1030, 1031, 1040, 1040),
    in_domain = FALSE, value_kind = c("nan", "nan", "nan", "normal"),
    reference_kind = c("nan", "nan", "nan", "nan"), n = c(10, 10, 5, 1))
  rr <- resolve_domain_conventions(rg, kd)
  pt <- data.frame(
    run_id = c("A", "A", "B", "A"),
    cell_id = c("s/anvl/f64/grad/p/f", "s/anvl/f64/value/p/f", "s/anvl/f64/grad/p/f", "s/anvl/f64/grad/p/f"),
    output = c("x", "value", "x", "x"), label = c("+inf", "+inf", "+inf", "+one"),
    bits = c("0x7FF0000000000000", "0x7FF0000000000000", "0x7FF0000000000000", "0x7FF0000000000000"),
    failure = c(TRUE, FALSE, TRUE, TRUE), cause = c("outside_domain", NA, "outside_domain", "outside_domain"),
    category = c("failure", NA, "failure", "failure"),
    value_kind = c("normal", "nan", "normal", "normal"), reference_kind = c("nan", "nan", "nan", "nan"))
  pr <- resolve_point_conventions(pt)
  ec <- c(
    check("exact points include +-0, +-Inf and NaN, each once",
      sum(ep$x == 0, na.rm = TRUE) == 2 && sum(is.infinite(ep$x)) == 2 && sum(is.nan(ep$x)) == 1 &&
        !anyDuplicated(ep$bits)),
    check("a domain edge comes with both representable neighbours",
      all(c(1 - 2^-24, 1, 1 + 2^-23) %in% ep$x)),
    check("an f32 cell's support edge is the edge after conversion to f32",
      as_f32(-pi) %in% ep$x && !(-pi %in% ep$x)),
    check("a gradient outside the domain is a convention only where both values are NaN",
      rr$category[1] == "undefined_domain"),
    check("... and stays a failure where the value cell found a finite value",
      rr$category[2] == "failure"),
    check("evidence from another run never settles a convention, and says why",
      rr$category[3] == "failure" && grepl("no value cell", rr$evidence[3])),
    check("a gradient point is a convention only against the same run's value point, by bits",
      identical(pr$category, c("undefined_domain", NA, "failure", "undefined_domain")) &&
        grepl("no value cell", pr$evidence[3])),
    check("the universal points 1/2 and 1 come with both neighbours",
      all(c(0.5 - 2^-25, 0.5, 0.5 + 2^-24, 1 - 2^-24, 1 + 2^-23) %in% exact_points("f32")$x)),
    check("nv_punif declares its log/log1p switch at the midpoint, with neighbours", {
      sp_all <- load_specs()
      if (is.null(sp_all$nv_punif)) TRUE else {
        bp <- sp_all$nv_punif$branch_points(list(min = -1, max = 3), list(log_p = TRUE), "f64")
        e <- exact_points("f64", branch = bp)
        bp == 1 && sum(grepl("branch_", e$label)) == 3 &&
          is.null(sp_all$nv_punif$branch_points(list(min = -1, max = 3), list(log_p = FALSE), "f64"))
      }
    }),
    check("the reference sees f32-rounded parameters in an f32 cell", {
      sp_all <- load_specs()
      if (is.null(sp_all$nv_dunif)) TRUE else {
        cf <- cell_functions(sp_all$nv_dunif, list(spec = "nv_dunif", backend = "anvl", dtype = "f32",
          kind = "value", param_set = names(sp_all$nv_dunif$params)[1], flags = "log=FALSE"))
        identical(unname(unlist(cf$ref_params)), as_f32(unname(unlist(cf$params))))
      }
    })
  )

  pts <- store_read(store_dir(opt$store), "points")
  pts <- pts[pts$run_id == run_id, , drop = FALSE]
  rng <- store_read(store_dir(opt$store), "ranges")
  rng <- rng[rng$run_id == run_id, , drop = FALSE]
  knd <- store_read(store_dir(opt$store), "kinds")
  knd <- knd[knd$run_id == run_id, , drop = FALSE]
  bnd <- store_read(store_dir(opt$store), "bands")
  bnd <- bnd[bnd$run_id == run_id, , drop = FALSE]
  p <- "selftest/anvl/%s/%s/%s/broken=%s"
  cat("\nwhat the sweep stored:\n")
  sw <- c(
    check("every cell stored its exact points, +-0 among them",
      nrow(pts) > 0 && all(tapply(pts$x %in% 0, paste(pts$cell_id, pts$output), sum) == 2)),
    check("the clean selftest cells pass every exact point",
      !any(pts$failure[grepl("broken=FALSE", pts$cell_id) & !grepl("/pinhole/", pts$cell_id)])),
    check("the f32 break at +Inf is recorded as an infinite-input failure, returning 0 against Inf", {
      r <- rng[rng$cell_id == sprintf("selftest/anvl/f32/value/clean/broken=TRUE"), , drop = FALSE]
      nrow(r) == 1 && r$cause == "inf_input" && r$category == "failure" && grepl("+0 vs +inf", r$pairs, fixed = TRUE)
    }),
    check("what each side returned is tallied for every cell", nrow(knd) > 0),
    check("zero inputs are a band of their own; binade 0 holds only the subnormals", {
      b <- bnd[grepl("/f32/value/clean/broken=FALSE", bnd$cell_id), , drop = FALSE]
      z <- b[b$zero, , drop = FALSE]
      s0 <- b[!b$zero & b$binade == 0, , drop = FALSE]
      nrow(z) == 2 && all(z$x_from == 0 & z$x_to == 0) && all(z$n_identical == 1) &&
        all(abs(s0$x_from) > 0 & abs(s0$x_to) > 0) &&
        sum(b$n_identical + b$n_differ + b$n_nonfinite) == get(sprintf(p, "f32", "value", "clean", "FALSE"))$n_samples
    }),
    check("a zero error is not displaced from its class by ten worse subnormals", {
      ## +0 with a 0.1 error, then ten subnormals with error 1, in one chunk
      x <- f32_from_bits(0:10)
      fx <- c(1.1, rep(2, 10))
      gx <- c(1, rep(1, 10))
      ctx <- context_from(list(v = 1.1), list(v = 1), "f32", "v", c(-Inf, Inf))
      sc <- score_pair(fx, gx, "f32")
      sc <- c(sc, sample_facts(x, fx, gx, sc, ctx, "v"))
      tk <- reducer_topk("f32", 10L)
      bd <- reducer_bands("f32")
      tk$add(list(idx = 0:10, x = x), sc, fx, gx, 1)
      bd$add(0:10, sc, 1, x, fx, gx)
      tag <- function(d) cbind(data.frame(run_id = "r", cell_id = "c", output = "v"), d)
      ct <- category_table(tag(binade_profile(bd, "f32")), tag(tk$get()),
        data.frame(cell_id = "c", domain_lo = -Inf, domain_hi = Inf))
      z <- ct[ct$input_class == "zero", ]
      abs(z$worst_rel_err - 0.1) < 1e-6 && z$worst_x == 0 &&
        ct$worst_rel_err[ct$input_class == "subnormal"] == 1
    }),
    check("... and a class of their own in the categories", {
      b <- bnd[grepl("/f32/value/clean/broken=FALSE", bnd$cell_id), , drop = FALSE]
      ct <- category_table(b, data.frame(run_id = character(0), cell_id = character(0),
        output = character(0), sign = numeric(0), binade = numeric(0), x = numeric(0),
        rel_err = numeric(0), bits = character(0), value = numeric(0), reference = numeric(0)),
        data.frame(cell_id = b$cell_id[1], domain_lo = -Inf, domain_hi = Inf))
      identical(ct$n[ct$input_class == "zero"], 2) && "subnormal" %in% ct$input_class
    })
  )

  ## Every category counts on the status screen, and exact points with them.
  cat("\nwhat status and report count:\n")
  lr <- latest_results(store_dir(opt$store))
  lr <- lr[lr$run_id == run_id, , drop = FALSE]
  one <- function(id, out = "value") lr[lr$cell_id == id & lr$output == out, , drop = FALSE]
  ph <- one(sprintf(p, "f64", "value", "pinhole", "FALSE"))
  row <- function(...) {
    base <- list(n_samples = 100, n_exact = 100, n_zero_sign = 0, worst_rel_err = 0,
      n_runs_unclassified = 0, n_regions_boundary = 0, n_regions_backend = 0, n_regions_domain = 0,
      n_failing_domain = 0, n_points = 5, n_points_identical = 5, n_points_failure = 0,
      n_points_boundary = 0, n_points_backend = 0, n_points_domain = 0, worst_point_rel_err = 0)
    as.data.frame(utils::modifyList(base, list(...)))
  }
  stt <- c(
    check("a failure only at x = 1 is caught by the exact points while the f64 sweep sees nothing",
      nrow(ph) == 1 && ph$n_inf == 0 && ph$worst_rel_err == 0 && ph$n_points_failure == 1 &&
        ph$first_point_failure %in% c("+one", "+one+domain_hi")),
    check("... and makes the result failing, and not bit-identical", {
      s <- result_state(ph); s$failing && !s$identical
    }),
    check("a boundary region alone makes a result not bit-identical, and is shown", {
      s <- result_state(row(n_exact = 99, n_regions_boundary = 1)); s$boundary && !s$identical && !s$failing
    }),
    check("a failing exact point alone does too", {
      s <- result_state(row(n_points_identical = 4, n_points_boundary = 1)); s$boundary && !s$identical
    }),
    check("a signed-zero difference is not bit-identical", !result_state(row(n_zero_sign = 1))$identical),
    check("undefined-domain conventions alone are set aside, and said so", {
      s <- result_state(row(n_exact = 90, n_regions_domain = 1, n_failing_domain = 10))
      !s$identical && s$identical_but_conventions && !s$failing
    })
  )

  cat("\nassertions:\n")
  ok <- c(sp, ca0, ca, zv, ec, sw, stt,
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
## "Did my fix help?" `status` cannot answer it: it shows the current state
## only, so an improvement is as invisible there as a regression.
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
  ## region and point counts resolved exactly as status and export see them
  res <- resummarise(res, resolved_ranges(dir), resolved_points(dir))
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

  ## direction, per result: the worst finite error over sweep and points, and
  ## every category's region and point counts. More failures, boundary or
  ## backend findings is worse; a change only in conventions or signed zeros
  ## is a change, shown, but neither worse nor better.
  now$worst_rel_err <- result_state(now)$worst_any
  before$worst_rel_err <- result_state(before)$worst_any
  counts <- c(
    "failure regions" = "n_runs_unclassified",
    "failing points" = "n_points_failure",
    "boundary regions" = "n_regions_boundary",
    "boundary points" = "n_points_boundary",
    "backend regions" = "n_regions_backend",
    "backend points" = "n_points_backend",
    "convention regions" = "n_regions_domain",
    "convention points" = "n_points_domain",
    "signed-zero samples" = "n_zero_sign"
  )
  cnt <- function(d, nm) {
    v <- d[[nm]]
    if (is.null(v)) rep(0, nrow(d)) else ifelse(is.na(v), 0, v)
  }
  weighs <- counts[1:6]
  same_counts <- Reduce(`&`, lapply(counts, function(nm) cnt(now, nm) == cnt(before, nm)))
  more <- Reduce(`|`, lapply(weighs, function(nm) cnt(now, nm) > cnt(before, nm)))
  fewer <- Reduce(`|`, lapply(weighs, function(nm) cnt(now, nm) < cnt(before, nm)))
  same <- !fresh &
    (now$worst_rel_err == before$worst_rel_err | (is.na(now$worst_rel_err) & is.na(before$worst_rel_err))) &
    same_counts
  worse <- !fresh & !same & (now$worst_rel_err > before$worst_rel_err | more) %in% TRUE
  better <- !fresh & !same & !worse & (now$worst_rel_err < before$worst_rel_err | fewer) %in% TRUE
  moved <- !fresh & !same & !worse & !better

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
    for (lab in names(counts)) {
      a <- cnt(before[k, , drop = FALSE], counts[[lab]])
      z <- cnt(now[k, , drop = FALSE], counts[[lab]])
      if (a != z) cat(sprintf("    %-20s %s -> %s\n", lab, format(a, big.mark = ","), format(z, big.mark = ",")))
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
  section(moved, "CHANGED (conventions or signed zeros only)")

  rule("SUMMARY")
  cat(sprintf("  %4d regressed\n", sum(worse)))
  cat(sprintf("  %4d improved\n", sum(better)))
  if (any(moved)) cat(sprintf("  %4d changed in conventions or signed zeros only\n", sum(moved)))
  cat(sprintf("  %4d unchanged (bit-identical to the earlier run)\n", sum(same)))
  cat(sprintf("  %4d had no earlier result to compare against\n", sum(fresh)))
  cat("\n")
  invisible(data.frame(
    cell_id = now$cell_id,
    output = now$output,
    change = ifelse(fresh, "new", ifelse(same, "same", ifelse(worse, "regressed", ifelse(better, "improved", "changed"))))
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

  ## Each cell's valid input domain and distribution support, resolved from its
  ## params and flags exactly as the sweep resolved them -- at the precision
  ## the implementation receives. Carried on the summary, so a reader can shade
  ## either part of the axis without asking the harness.
  cells <- g[match(unique(res$cell_id), g$cell_id), , drop = FALSE]
  bounds <- lapply(seq_len(nrow(cells)), function(i) {
    row <- cells[i, ]
    spec <- specs[[row$spec]]
    cf <- cell_functions(spec, row)
    d <- if (is.null(spec$domain)) c(-Inf, Inf) else spec$domain(cf$ref_params, cf$flags)
    u <- if (is.null(spec$support)) c(NA_real_, NA_real_) else spec$support(cf$ref_params, cf$flags)
    c(d, u)
  })
  support <- data.frame(
    cell_id = cells$cell_id,
    domain_lo = vapply(bounds, `[`, 0, 1L),
    domain_hi = vapply(bounds, `[`, 0, 2L),
    support_lo = vapply(bounds, `[`, 0, 3L),
    support_hi = vapply(bounds, `[`, 0, 4L)
  )
  m <- match(res$cell_id, support$cell_id)
  res$domain_lo <- support$domain_lo[m]
  res$domain_hi <- support$domain_hi[m]
  res$support_lo <- support$support_lo[m]
  res$support_hi <- support$support_hi[m]

  out <- normalizePath(opt$out, mustWork = FALSE)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  runs <- store_read(dir, "runs")
  runs <- runs[runs$run_id %in% unique(res$run_id), , drop = FALSE]

  ## Keyed on exactly the rows kept above, so a detail row from a superseded
  ## run can never leak in beside a newer summary row.
  ##
  ## Regions and points are taken already resolved against the WHOLE store --
  ## the same call the terminal makes -- and only then filtered. Resolving
  ## after filtering let a gradient-only export drop the value cells whose
  ## evidence settles a convention, turning a terminal convention into an
  ## exported failure. `res` came from latest_results(), which summarised the
  ## same resolved tables, so its region and point counts agree with these.
  keep <- paste(res$run_id, res$cell_id, res$output)
  resolved <- list(ranges = resolved_ranges(dir), points = resolved_points(dir))
  pick <- function(tbl) {
    x <- if (tbl %in% names(resolved)) resolved[[tbl]] else store_read(dir, tbl)
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

  kept <- list()
  for (tbl in c("detail", "bands", "hist", "ranges", "kinds", "points")) {
    x <- pick(tbl)
    if (!is.null(x) && nrow(x)) kept[[tbl]] <- x
  }

  nanoparquet::write_parquet(runs, file.path(out, "runs.parquet"))
  nanoparquet::write_parquet(res[order(res$cell_id, res$output), , drop = FALSE], file.path(out, "summary.parquet"))
  counts <- c(runs = nrow(runs), summary = nrow(res))
  for (tbl in names(kept)) {
    counts[tbl] <- write_by_cell(kept[[tbl]], file.path(out, paste0(tbl, ".parquet")))
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
