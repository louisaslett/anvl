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
## matter for deciding where to look next are the worst finite relative error
## (over the sweep and the exact points) and the count of results with a
## failure (a failure region or a failing exact point), so those are what a
## row carries.
summarise_by <- function(res, axis) {
  st <- result_state(res)
  res$worst_any <- st$worst_any
  res$failing <- st$failing
  keys <- sort(unique(res[[axis]]))
  do.call(
    rbind,
    lapply(keys, function(k) {
      g <- res[res[[axis]] == k, , drop = FALSE]
      row <- data.frame(key = k, n = nrow(g))
      for (dt in c("f32", "f64")) {
        d <- g[g$dtype == dt, , drop = FALSE]
        row[[paste0(dt, "_worst")]] <- if (nrow(d)) max(d$worst_any, na.rm = TRUE) else NA_real_
        row[[paste0(dt, "_unexp")]] <- if (nrow(d)) sum(d$failing) else NA_integer_
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
    cat(sprintf("%s%-10s %12s %12s %s\n", indent, "precision", "worst rel", "worst ulp", "failures"))
    for (i in seq_len(nrow(sm))) {
      d <- res[res$dtype == sm$key[i], , drop = FALSE][1L, , drop = FALSE]
      np <- d$n_points_failure %||% 0
      cat(sprintf(
        "%s%-10s %12s %12s %s\n",
        indent,
        sm$key[i],
        fmt_num(result_state(d)$worst_any),
        fmt_num(d$worst_ulp_err),
        if (d$n_runs_unclassified > 0 || np > 0) {
          sprintf("%d region(s), %d exact point(s)", d$n_runs_unclassified, np)
        } else {
          "."
        }
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

  H <- c("worst rel", "failing")
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
    "\n%s\"failing\" counts results with a failure region or a failing exact point;\n%s\"worst rel\" is the worst finite error over the sweep and the exact points.\n",
    indent,
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
## Collapse the per-binade profile into runs that behave alike.
##
## Adjacent binades merge when they share a sign, a behaviour and an *upper*
## error bound. The upper bound alone is the key because within one binade the
## finite errors routinely span a dozen decades -- a few inputs round almost
## exactly while most sit on the precision floor -- so a lower bound carries no
## information and keying on it fragments the table threefold. m_best is still
## reported for each merged run.
##
## This is a rendering step: the store keeps every binade, so a chart can use
## the full series and this compact view is derived from the same numbers.
merge_bands <- function(b) {
  if (is.null(b) || !nrow(b)) {
    return(b)
  }
  if (is.null(b$zero)) b$zero <- FALSE
  b <- b[order(b$special, b$sign, b$binade, !b$zero), , drop = FALSE]
  key <- paste(b$sign, b$special, b$behaviour, b$m_worst)
  ## the Inf/NaN field and each sign's zero are rows of their own, never
  ## merged into a neighbouring range
  alone <- b$special | b$zero
  contiguous <- c(TRUE, diff(b$binade) != 1L | key[-1L] != key[-length(key)] | alone[-1L] | alone[-nrow(b)])
  ## the zero row and binade 0's subnormals share a binade number
  contiguous[c(FALSE, diff(b$binade) == 0L)] <- TRUE
  grp <- cumsum(contiguous)

  agg <- function(f, col) vapply(split(b[[col]], grp), f, numeric(1))
  na_min <- function(v) if (all(is.na(v))) NA_real_ else min(v, na.rm = TRUE)
  na_max <- function(v) if (all(is.na(v))) NA_real_ else max(v, na.rm = TRUE)

  out <- data.frame(
    sign = agg(function(v) v[1L], "sign"),
    x_from = agg(na_min, "x_from"),
    x_to = agg(na_max, "x_to"),
    special = vapply(split(b$special, grp), function(v) v[1L], logical(1)),
    zero = vapply(split(b$zero, grp), function(v) v[1L], logical(1)),
    behaviour = vapply(split(b$behaviour, grp), function(v) v[1L], ""),
    n_identical = agg(sum, "n_identical"),
    n_differ = agg(sum, "n_differ"),
    n_nonfinite = agg(sum, "n_nonfinite"),
    m_worst = agg(na_min, "m_worst"),
    m_best = agg(na_max, "m_best"),
    worst_rel_err = agg(max, "worst_rel_err"),
    n_binades = as.integer(table(grp))
  )
  rownames(out) <- NULL
  ## x_from is NA on the Inf/NaN rows, so sign breaks the tie and their order
  ## does not drift between runs.
  out[order(out$special, out$x_from, ifelse(out$zero, out$sign, -out$sign)), , drop = FALSE]
}

print_bands <- function(bands) {
  cat("\nBEHAVIOUR ACROSS THE NUMBER LINE\n")
  if (is.null(bands) || !nrow(bands)) {
    cat("  (not recorded \u2014 re-run the sweep to populate this)\n")
    return(invisible(NULL))
  }
  b <- merge_bands(bands)
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
    } else if (isTRUE(b$zero[i])) {
      sprintf("%s0 exactly", if (b$sign[i] > 0) "+" else "-")
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

CATEGORY_LABEL <- c(
  failure = "FAILURE",
  backend_limitation = "backend limitation",
  boundary = "domain boundary",
  undefined_domain = "undefined-domain convention",
  reference_limitation = "verified base R limitation"
)

report_cell <- function(spec, row, res, detail, ranges, hist, bands = NULL, points = NULL, disputes = NULL,
                        validations = NULL) {
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
  ## Facts that sit across the three lines above, each kept visible.
  extra <- function(label, v, note) {
    if (!is.null(v) && !is.na(v) && v > 0) {
      cat(sprintf("  %-23s%18s  %s\n", label, human_int(v), note))
    }
  }
  extra("correctly rounded", res$n_rounded, "differ, but the best this precision can do")
  extra("signed zero differs", res$n_zero_sign, "counted identical above: +0 against -0")
  extra("flushed subnormal", res$n_flushed, "the result at the flushed zero, which is right")
  extra("flushed, zero wrong", res$n_flushed_zero_error, "the result at the flushed zero, which is itself wrong")

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
    ## the ulp error of *this* input; the worst ulp error over the sweep may
    ## belong to another input, and is shown on its own line
    here_ulp <- if (!is.null(detail) && nrow(detail)) detail$ulp_err[which.max(detail$rel_err)] else NA_real_
    cat(sprintf(
      "\n  worst   %s relative  (%s ulp)  at x = %s\n",
      fmt_num(res$worst_rel_err),
      fmt_num(here_ulp),
      exact_num(res$worst_x)
    ))
    cat(sprintf("          %s\n", res$worst_bits))
    cat(sprintf("          anvl    %.17g\n", res$worst_value))
    cat(sprintf("          base R  %.17g\n", res$worst_reference))
    cat(sprintf("  worst ulp error over every sample: %s\n", fmt_num(res$worst_ulp_err)))
  }

  print_disputes(res, disputes)
  print_validation(res, validations)

  print_bands(bands)

  cat("\nREGIONS where they differ with no finite error\n")
  if (is.null(ranges) || !nrow(ranges)) {
    cat("  none\n")
  } else {
    ## A store written before causes were recorded has only the old
    ## positional `class`; read it as the category it used to stand for.
    if (is.null(ranges$category)) {
      ranges$category <- ifelse(ranges$class == "unclassified", "failure", ranges$class)
      ranges$cause <- ranges$class
      ranges$pairs <- ""
    }
    lab <- CATEGORY_LABEL
    r <- ranges[order(ranges$category != "failure"), , drop = FALSE]
    rng <- function(a, b) {
      if (is.nan(a) && is.nan(b)) {
        return("NaN")
      }
      sprintf("%s .. %s", fmt_num(min(a, b)), fmt_num(max(a, b)))
    }
    for (i in seq_len(nrow(r))) {
      what <- sprintf("%s (%s)", lab[[r$category[i]]] %||% r$category[i], gsub("_", " ", r$cause[i]))
      ## For f32 the bounds are sampled inputs. For f64 they are the bounds of
      ## the 2^32-pattern blocks the failing samples fell in -- not inputs that
      ## were evaluated -- so they are labelled as such and the sampled
      ## evidence is shown beside them. n_patterns is how many representable
      ## values lie in the interval, not how many were sampled.
      bounds <- isTRUE(r$bounds_are_samples[i]) || is.null(r$bounds_are_samples)
      cat(sprintf(
        "  %-28s %-44s %s\n",
        rng(r$x_from[i], r$x_to[i]),
        what,
        if (bounds) {
          sprintf("spans %s values", human_int(r$n_patterns[i]))
        } else {
          sprintf("block bounds; %s values", human_int(r$n_patterns[i]))
        }
      ))
      if (!bounds && !is.null(r$sampled_from)) {
        cat(sprintf(
          "  %-28s sampled: %s failing, %s .. %s (%s .. %s)\n",
          "",
          human_int(r$n_failing[i]),
          exact_num(r$sampled_from[i]),
          exact_num(r$sampled_to[i]),
          r$sampled_bits_from[i],
          r$sampled_bits_to[i]
        ))
      }
      if (!is.null(r$pairs) && nzchar(r$pairs[i])) cat(sprintf("  %-28s returned: %s\n", "", r$pairs[i]))
      if (!is.null(r$rep_x) && !is.na(r$rep_x[i])) {
        cat(sprintf(
          "  %-28s e.g. x = %s (%s): %.17g vs base R %.17g\n",
          "",
          exact_num(r$rep_x[i]),
          r$rep_bits[i],
          r$rep_value[i],
          r$rep_reference[i]
        ))
      }
      if (!is.null(r$evidence) && !is.na(r$evidence[i])) cat(sprintf("  %-28s evidence: %s\n", "", r$evidence[i]))
      if (isTRUE(r$ref_candidate[i]) && !identical(r$category[i], "reference_limitation")) {
        cat(sprintf("  %-28s base R disputed by the stable reference (candidate: %s)\n", "", res$ref_stable_status %||% "not validated"))
      }
    }
  }

  print_points(points)

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

## The mandatory exact points: every one that is not bit-identical to base R,
## with its category, and a count of the rest. Never merged into the sweep's
## figures above.
print_points <- function(points) {
  cat("\nEXACT POINTS")
  if (is.null(points) || !nrow(points)) {
    cat("\n  (not recorded \u2014 re-run the sweep to populate this)\n")
    return(invisible(NULL))
  }
  same <- points$identical & !points$zero_sign
  cat(sprintf(" \u2014 %d of %d bit-identical to base R\n", sum(same), nrow(points)))
  p <- points[!same, , drop = FALSE]
  if (!nrow(p)) {
    return(invisible(NULL))
  }
  lab <- CATEGORY_LABEL
  p <- p[order(!p$failure, p$category != "failure", -p$rel_err), , drop = FALSE]
  what <- ifelse(
    p$failure,
    sprintf("%s (%s)", ifelse(is.na(lab[p$category]), p$category, lab[p$category]), gsub("_", " ", p$cause)),
    ifelse(p$zero_sign, "signed zero differs", ifelse(p$rounded, "correctly rounded", sprintf("rel %s", fmt_num(p$rel_err))))
  )
  for (i in seq_len(nrow(p))) {
    cat(sprintf(
      "  %-30s x = %-24s %s\n",
      substr(p$label[i], 1L, 30L),
      exact_num(p$x[i]),
      what[i]
    ))
    cat(sprintf("  %-30s %s  %.17g vs base R %.17g\n", "", p$bits[i], p$value[i], p$reference[i]))
  }
}

## Candidate base R disputes, where the spec has a stable reference. Shown as
## candidates: the figures without them sit beside the unfiltered ones, never
## in their place, and nothing is excluded until the stable reference passes
## validation.
print_disputes <- function(res, disputes) {
  if (is.null(res$n_ref_candidate) || is.na(res$n_ref_candidate)) {
    return(invisible(NULL))
  }
  status <- res$ref_stable_status %||% "not validated"
  verified <- identical(status, "validated")
  cat(if (verified) {
    "\nBASE R DISPUTES \u2014 verified reference limitations: the stable reference passed validation\n"
  } else {
    sprintf("\nBASE R DISPUTES \u2014 candidates, not exclusions: the stable reference is %s\n", status)
  })
  cat(sprintf("  why     %s\n", res$ref_stable_note))
  cat(sprintf(
    "  candidates %s  (%s with no finite error)   base R and anvl agree, both off: %s\n",
    human_int(res$n_ref_candidate),
    human_int(res$n_ref_candidate_nonfinite),
    human_int(res$n_ref_shared)
  ))
  without <- if (verified) "excluding verified limitations" else "without candidates"
  cat(sprintf(
    "  worst rel err         all %-10s %s %s\n",
    fmt_num(res$worst_rel_err),
    without,
    fmt_num(res$worst_rel_err_excl)
  ))
  cat(sprintf(
    "  normal outputs        all %-10s %s %s\n",
    fmt_num(res$worst_out_normal),
    without,
    fmt_num(res$worst_out_normal_excl)
  ))
  if (is.null(disputes) || !nrow(disputes)) {
    return(invisible(NULL))
  }
  for (kind in c("candidate", "shared")) {
    d <- disputes[disputes$kind == kind, , drop = FALSE]
    if (!nrow(d)) next
    d <- utils::head(d[order(-d$beyond_tolerance), , drop = FALSE], 3L)
    cat(sprintf("  e.g. (%s)\n", kind))
    for (i in seq_len(nrow(d))) {
      cat(sprintf(
        "    x = %-24s %s\n      anvl %.17g  base R %.17g  stable %.17g\n      |anvl - s| %s <= %s   |base R - s| %s vs %s\n",
        exact_num(d$x[i]), d$bits[i], d$value[i], d$reference[i], d$stable[i],
        fmt_num(d$d_anvl[i]), fmt_num(d$t_anvl[i]), fmt_num(d$d_base[i]), fmt_num(d$t_base[i])
      ))
    }
  }
}

## The latest validation of each reference this result was scored with or
## disputed by, as recorded: what was checked, how, and what it found.
print_validation <- function(res, validations) {
  ids <- c(stable = res$ref_stable_id %||% NA, gradient = res$ref_grad_id %||% NA)
  st <- c(stable = res$ref_stable_status %||% NA, gradient = res$ref_grad_status %||% NA)
  if (all(is.na(st))) {
    return(invisible(NULL))
  }
  cat("\nREFERENCE VALIDATION against high precision\n")
  for (k in names(st)[!is.na(st)]) {
    v <- if (is.null(validations)) NULL else validations[validations$reference == k & validations$ref_id %in% ids[[k]] &
      (k == "stable" | validations$output == res$output), , drop = FALSE]
    cat(sprintf("  %-8s %s\n", k, st[[k]]))
    if (is.null(v) || !nrow(v)) next
    last <- v[which.max(as.POSIXct(v$validated_at, format = "%Y-%m-%dT%H:%M:%S%z")), ]
    older <- nrow(v) - sum(v$truth_id == last$truth_id)
    cat(sprintf(
      "           %s at %d bits, %s samples: max %s ulp against a bound of %s%s\n",
      last$validated_at, last$precision, human_int(last$n_samples), fmt_num(last$max_err_ulp64),
      last$bound_ulp64, if (nzchar(last$reason %||% "")) paste0("\n           ", last$reason) else ""
    ))
    if (older > 0) cat(sprintf("           (%d earlier validation(s) under a superseded truth)\n", older))
  }
}
