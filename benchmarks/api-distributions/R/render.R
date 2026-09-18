## ---------------------------------------------------------------------------
## Rendering: how a set of results is drawn on a terminal.
##
## Everything here formats and prints. It computes no measurements and reads no
## files, so `run.R` stays what it says on the tin -- argument parsing and the
## commands -- and the two screens that summarise a level (`report` and
## `browse`) cannot drift apart, because they call the same functions.
## ---------------------------------------------------------------------------

## A section heading, padded to a fixed width so the screen reads as a screen
## rather than as scrollback.
rule <- function(title) {
  cat("\n", title, " ", strrep("\u2500", max(4L, 68L - nchar(title))), "\n", sep = "")
}

## ---- summarising a group of results ---------------------------------------
##
## The axes, in the order it is useful to narrow them. dtype is deliberately
## last: f32 and f64 errors differ by about nine orders of magnitude, so a
## summary that pools them has a meaningless maximum, and comparing the two is
## the single most common thing to want. They are columns at every level until
## the final step, where acting on a result means picking one.
##
## `output` covers value-vs-gradient on its own: a value cell reports output
## "value", a gradient cell reports one row per differentiated argument.
DRILL <- c("spec", "backend", "output", "flags", "param_set", "dtype")

DRILL_LABEL <- c(
  spec = "function",
  backend = "backend",
  output = "value / gradient",
  flags = "flags",
  param_set = "parameter set",
  dtype = "precision"
)

## The first axis that still has more than one value in this set -- i.e. the
## next useful thing to narrow by. NULL when everything is pinned.
next_axis <- function(res) {
  for (a in DRILL) {
    if (length(unique(res[[a]])) > 1L) {
      return(a)
    }
  }
  NULL
}

## One row per group, with f32 and f64 side by side. The two numbers that
## matter for deciding where to look next are the worst relative error and the
## count of regions nothing explains, so those are what a row carries.
summarise_by <- function(res, axis) {
  keys <- sort(unique(res[[axis]]))
  do.call(
    rbind,
    lapply(keys, function(k) {
      g <- res[res[[axis]] == k, , drop = FALSE]
      row <- data.frame(key = k, n = nrow(g))
      for (dt in c("f32", "f64")) {
        d <- g[g$dtype == dt, , drop = FALSE]
        row[[paste0(dt, "_worst")]] <- if (nrow(d)) max(d$worst_rel_err, na.rm = TRUE) else NA_real_
        row[[paste0(dt, "_unexp")]] <- if (nrow(d)) sum(d$n_runs_unclassified > 0) else NA_integer_
        row[[paste0(dt, "_n")]] <- nrow(d)
      }
      row
    })
  )
}

print_summary <- function(res, axis, indent = "  ") {
  sm <- summarise_by(res, axis)

  ## At the precision level the f32/f64 columns would repeat the rows, so show
  ## a plain list instead. This is the last step before a single result.
  if (axis == "dtype") {
    cat(sprintf("%s%-10s %12s %12s %s\n", indent, "precision", "worst rel", "worst ulp", "regions"))
    for (i in seq_len(nrow(sm))) {
      d <- res[res$dtype == sm$key[i], , drop = FALSE][1L, , drop = FALSE]
      cat(sprintf(
        "%s%-10s %12s %12s %s\n",
        indent,
        sm$key[i],
        fmt_num(d$worst_rel_err),
        fmt_num(d$worst_ulp_err),
        if (d$n_runs_unclassified > 0) sprintf("%d unexplained", d$n_runs_unclassified) else "."
      ))
    }
    return(invisible(sm))
  }

  ## Every column width is derived from what will actually be printed, headers
  ## included. Hard-coding %10s under a header reading "unexplained" -- which is
  ## 11 characters -- made each header cell run one column wider than its data,
  ## because sprintf pads to a *minimum* width and never truncates.
  cells <- lapply(seq_len(nrow(sm)), function(i) {
    one <- function(dt) {
      if (sm[[paste0(dt, "_n")]][i] == 0L) {
        return(c("-", "-"))
      }
      c(
        fmt_num(sm[[paste0(dt, "_worst")]][i]),
        if (sm[[paste0(dt, "_unexp")]][i] > 0) sprintf("%d", sm[[paste0(dt, "_unexp")]][i]) else "."
      )
    }
    c(one("f32"), one("f64"))
  })

  H <- c("worst rel", "unexplained")
  w <- max(nchar(sm$key), nchar(DRILL_LABEL[[axis]]))
  numw <- max(nchar(H), nchar(unlist(cells)))
  groupw <- 2L * numw + 1L
  dashes <- function(tag) {
    n <- groupw - nchar(tag) - 2L
    paste0(strrep("-", n %/% 2L), " ", tag, " ", strrep("-", n - n %/% 2L))
  }

  cat(sprintf("%s%-*s %s %s\n", indent, w, "", dashes("f32"), dashes("f64")))
  cat(sprintf(
    "%s%-*s %*s %*s %*s %*s\n",
    indent,
    w,
    DRILL_LABEL[[axis]],
    numw,
    H[1L],
    numw,
    H[2L],
    numw,
    H[1L],
    numw,
    H[2L]
  ))
  for (i in seq_len(nrow(sm))) {
    v <- cells[[i]]
    cat(sprintf(
      "%s%-*s %*s %*s %*s %*s\n",
      indent,
      w,
      if (nzchar(sm$key[i])) sm$key[i] else "-",
      numw,
      v[1L],
      numw,
      v[2L],
      numw,
      v[3L],
      numw,
      v[4L]
    ))
  }
  cat(sprintf(
    "\n%s\"unexplained\" counts results with a disagreement the spec cannot account for.\n",
    indent
  ))
  invisible(sm)
}


## ---- the factual view -----------------------------------------------------
##
## What the sweep measured, with nothing decided about it. This is the
## descendant of the old per-sweep .txt files, which got one thing right: you
## could read one and know exactly what had been computed. Their problem was
## that there were 217 and no index, not that they reported plainly.

human_int <- function(n) {
  if (is.na(n)) {
    return("NA")
  }
  if (n >= 1e15) sprintf("%.3g", n) else formatC(n, format = "d", big.mark = ",")
}

pct <- function(a, b) if (b == 0) "" else sprintf("%6.2f%%", 100 * a / b)

## A fixed-width bar, scaled to the largest count. Padded here rather than by
## sprintf's %-30s: the block character is multi-byte, so sprintf pads it to a
## width in *bytes* and the column after it comes out ragged.
bar <- function(n, max_n, width = 30L) {
  if (max_n <= 0) {
    return(strrep(" ", width))
  }
  k <- max(if (n > 0) 1L else 0L, round(width * n / max_n))
  paste0(strrep("█", k), strrep(" ", width - k))
}

## The shortest decimal that reads back as the same double. The exact input is
## the entire point of a bit-pattern sweep, so it is never abbreviated.
exact_num <- function(x) {
  if (is.na(x) || !is.finite(x)) fmt_num(x) else format(x, digits = 17, trim = TRUE)
}

## A compact one-line identifier for a cell: parameter set plus the flags that
## are actually on, e.g. "unit/upper,log".
short_cell <- function(row) {
  fl <- character(0)
  if (nzchar(row$flags) && row$flags != "-") {
    for (kv in strsplit(strsplit(row$flags, ",", fixed = TRUE)[[1L]], "=", fixed = TRUE)) {
      on <- isTRUE(as.logical(kv[2L]))
      fl <- c(
        fl,
        switch(
          kv[1L],
          lower_tail = if (on) "lower" else "upper",
          log_p = if (on) "log" else NULL,
          log = if (on) "log" else NULL,
          if (on) kv[1L] else NULL
        )
      )
    }
  }
  paste0(row$param_set, if (length(fl)) paste0("/", paste(fl, collapse = ",")) else "")
}

## Describe a cell in words rather than as a slash-separated key.
describe_cell <- function(spec, row) {
  params <- spec$params[[row$param_set]]
  bits <- character(0)
  if (length(params)) {
    bits <- c(
      bits,
      paste0(
        row$param_set,
        " (",
        paste(sprintf("%s = %s", names(params), vapply(params, exact_num, "")), collapse = ", "),
        ")"
      )
    )
  }
  fl <- row$flags
  if (nzchar(fl) && fl != "-") {
    for (kv in strsplit(strsplit(fl, ",", fixed = TRUE)[[1L]], "=", fixed = TRUE)) {
      on <- isTRUE(as.logical(kv[2L]))
      bits <- c(
        bits,
        switch(
          kv[1L],
          lower_tail = if (on) "lower tail" else "upper tail",
          log_p = if (on) "log scale" else "probability scale",
          log = if (on) "log scale" else "density scale",
          sprintf("%s = %s", kv[1L], kv[2L])
        )
      )
    }
  }
  paste(bits, collapse = " · ")
}

## The structural map: where along the number line the behaviour changes.
## Reported per binade and merged, so a long stretch that is bit-identical
## collapses to one line and the line where that stops is the finding.
print_bands <- function(bands) {
  cat("\nBEHAVIOUR ACROSS THE NUMBER LINE\n")
  if (is.null(bands) || !nrow(bands)) {
    cat("  (not recorded \u2014 re-run the sweep to populate this)\n")
    return(invisible(NULL))
  }
  b <- bands[order(bands$special, bands$x_from), , drop = FALSE]
  cat(sprintf(
    "  %-28s %-22s %6s %6s %6s  %s\n",
    "range",
    "behaviour",
    "ident.",
    "differ",
    "no fin.",
    "relative error within"
  ))
  for (i in seq_len(nrow(b))) {
    span <- if (isTRUE(b$special[i])) {
      sprintf("%sInf and NaN", if (b$sign[i] > 0) "+" else "-")
    } else {
      sprintf("%s .. %s", fmt_num(b$x_from[i]), fmt_num(b$x_to[i]))
    }
    tot <- b$n_identical[i] + b$n_differ[i] + b$n_nonfinite[i]
    ## A non-zero count must never print as 0%: that made a row labelled
    ## "mixed" show 0% / 0% / 100% and read as a contradiction.
    p <- function(n) {
      if (tot == 0) {
        return("-")
      }
      if (n == 0) {
        return("0%")
      }
      pc <- 100 * n / tot
      if (pc < 1) {
        "<1%"
      } else if (pc > 99 && n < tot) {
        ">99%"
      } else {
        sprintf("%.0f%%", pc)
      }
    }
    ## m is the decade of the error: 10^-(m+1) <= e < 10^-m. m_worst is the
    ## smallest m seen (the largest error), m_best the largest (the smallest),
    ## so the pair bounds every finite error taken across the whole range.
    env <- if (is.na(b$m_worst[i])) {
      "\u2014"
    } else {
      sprintf("%s .. %s", fmt_num(10^(-(b$m_best[i] + 1))), fmt_num(10^(-b$m_worst[i])))
    }
    cat(sprintf(
      "  %-28s %-22s %6s %6s %6s  %s\n",
      span,
      b$behaviour[i],
      p(b$n_identical[i]),
      p(b$n_differ[i]),
      p(b$n_nonfinite[i]),
      env
    ))
  }
  cat("  percentages are of the samples taken in that range; a range is a run of\n")
  cat("  binades sharing both a behaviour and an error bound, so a boundary is a\n")
  cat("  place where one of those genuinely changes.\n")
}

report_cell <- function(spec, row, res, detail, ranges, hist, bands = NULL) {
  what <- if (row$kind == "value") "value" else sprintf("gradient d/d%s", res$output)
  cat(strrep("\u2500", 72), "\n", sep = "")
  cat(sprintf("%s  \u00b7  %s\n", row$spec, spec$blurb))
  cat(sprintf(
    "%s %s over `%s` \u00b7 %s\n",
    row$dtype,
    what,
    spec$primary,
    describe_cell(spec, row)
  ))
  cat(strrep("\u2500", 72), "\n", sep = "")

  how <- if (row$dtype == "f32") {
    if (res$depth == "full") {
      "every float32 value, exhaustively"
    } else {
      sprintf("1 in every %s float32 values", human_int(DEPTHS[[res$depth]]$stride))
    }
  } else {
    sprintf(
      "stratified: 1 sample per %s-block of the float64 line",
      if (res$depth == "full") "2^32" else sprintf("%s x 2^32", human_int(DEPTHS[[res$depth]]$stride))
    )
  }
  cat(sprintf("swept   %s samples \u2014 %s\n", human_int(res$n_samples), how))
  cat(sprintf(
    "run     %s on %s \u00b7 anvl %s \u00b7 %.1f s\n\n",
    res$run_id,
    res$platform_key,
    substr(res$anvl_sha %||% "?", 1L, 7L),
    res$elapsed_sec
  ))

  n_diff <- res$n_samples - res$n_exact - res$n_inf
  cat("AGREEMENT WITH BASE R\n")
  cat(sprintf("  identical              %18s  %s\n", human_int(res$n_exact), pct(res$n_exact, res$n_samples)))
  cat(sprintf("  differ, finite error   %18s  %s\n", human_int(n_diff), pct(n_diff, res$n_samples)))
  cat(sprintf(
    "  differ, no finite error%18s  %s%s\n",
    human_int(res$n_inf),
    pct(res$n_inf, res$n_samples),
    if (res$n_inf > 0) "   (see REGIONS below)" else ""
  ))

  if (!is.null(hist) && nrow(hist) && sum(hist$count) > 0) {
    cat(sprintf("\nRELATIVE ERROR of the %s that differ\n", human_int(n_diff)))
    h <- hist[hist$count > 0, , drop = FALSE]
    h <- h[order(h$decade), , drop = FALSE]
    mx <- max(h$count)
    for (i in seq_len(nrow(h))) {
      lab <- if (h$decade[i] <= -20L) "      <1e-19" else sprintf("1e%-3d..1e%-3d", h$decade[i], h$decade[i] + 1L)
      cat(sprintf("  %-14s %s %s\n", lab, bar(h$count[i], mx), human_int(h$count[i])))
    }
  }
  if (!is.na(res$worst_x) && res$worst_rel_err > 0) {
    cat(sprintf(
      "\n  worst   %s relative  (%s ulp)  at x = %s\n",
      fmt_num(res$worst_rel_err),
      fmt_num(res$worst_ulp_err),
      exact_num(res$worst_x)
    ))
    cat(sprintf("          %s\n", res$worst_bits))
    cat(sprintf("          anvl    %.17g\n", res$worst_value))
    cat(sprintf("          base R  %.17g\n", res$worst_reference))
  }

  print_bands(bands)

  cat("\nREGIONS where they differ with no finite error\n")
  if (is.null(ranges) || !nrow(ranges)) {
    cat("  none\n")
  } else {
    lab <- c(
      nan = "reference is NaN",
      subnormal = "subnormal, flushed by the backend",
      below_support = "below the supported domain",
      above_support = "above the supported domain",
      unclassified = "UNEXPLAINED"
    )
    r <- ranges[order(ranges$class == "unclassified", decreasing = TRUE), , drop = FALSE]
    for (i in seq_len(nrow(r))) {
      lo <- min(r$x_from[i], r$x_to[i])
      hi <- max(r$x_from[i], r$x_to[i])
      span <- if (is.nan(r$x_from[i]) && is.nan(r$x_to[i])) "NaN" else sprintf("%s .. %s", fmt_num(lo), fmt_num(hi))
      ## n_patterns is how many representable values lie in the interval, not
      ## how many were sampled -- at anything short of `full` those differ by
      ## the stride.
      cat(sprintf(
        "  %-28s %-36s spans %s values\n",
        span,
        lab[[r$class[i]]],
        human_int(r$n_patterns[i])
      ))
    }
  }

  if (!is.null(detail) && nrow(detail) > 1L) {
    cat(sprintf("\n  next worst inputs (%d kept in the store)\n", nrow(detail)))
    d <- utils::head(detail[order(-detail$rel_err), , drop = FALSE], 5L)[-1L, , drop = FALSE]
    for (i in seq_len(nrow(d))) {
      cat(sprintf(
        "    x = %-24s %s  %s rel, %s ulp\n",
        exact_num(d$x[i]),
        d$bits[i],
        fmt_num(d$rel_err[i]),
        fmt_num(d$ulp_err[i])
      ))
    }
  }
  cat("\n")
}
