## ---------------------------------------------------------------------------
## The sweep engine: enumerate -> score -> reduce.
##
## These three stages are deliberately separate. Enumeration decides *which*
## bit patterns are visited; scoring decides *how far apart* two answers are;
## reduction decides *what is remembered* about billions of samples. Adding a
## metric or a summary must not mean touching the enumerator, and the selftest
## needs to drive each stage on its own.
##
## No distribution, no reference mathematics and no I/O appear here.
## ---------------------------------------------------------------------------

## ---- 1. enumeration --------------------------------------------------------
##
## Both precisions are swept through one abstraction: a pattern index space of
## 2^31 values per sign, walked with a stride. What the index means differs.
##
##   f32  index = the 31 non-sign bits, so stride 1 enumerates every one of the
##        2^32 float32 bit patterns exactly once. One caveat, and only one:
##        widening an f32 to a double quiets a signalling NaN, so the
##        2 x (2^22 - 1) sNaN patterns (0.195% of the space) arrive as their
##        quiet counterparts. Both sides of the comparison receive the same
##        value, so nothing is mismeasured -- the sweep simply cannot tell two
##        NaN payloads apart, which is of no consequence for a distribution
##        function. Every finite value, both zeros and both infinities are
##        delivered exactly as their pattern encodes them.
##
##   f64  index = the high word (sign + 11 exponent bits + top 20 mantissa
##        bits = exactly 32 bits); the low 32 mantissa bits are drawn at random
##        from a fixed seed. Stride 1 therefore takes exactly one sample from
##        each of the 2^32 contiguous blocks of 2^32 patterns: every (sign,
##        exponent, top-20-mantissa) combination is visited, which is the most
##        that is reachable when 2^64 is out of the question.
##
## Depth sets the stride, so a smoke run is the same sweep at coarser spacing
## rather than a different, narrower sweep -- it covers the whole number line,
## Inf and NaN included, and can fail anywhere the full run can.

DEPTHS <- list(
  ## samples = 2 signs * (2^31 / stride)
  smoke = list(stride = 2^13), #       524,288 samples
  quick = list(stride = 2^7), #     33,554,432 samples
  full = list(stride = 1) #  4,294,967,296 samples
)

CHUNK_INDEX <- 2^20 # pattern indices per sign per chunk

## The low mantissa bits of an f64 sample are random; the seed makes a run on
## one machine reproduce on another, which is what lets HPC shards be compared.
SWEEP_SEED <- 1142212L # "anvl" as a=1 .. z=26

sweep_plan <- function(dtype, depth) {
  d <- DEPTHS[[depth]]
  if (is.null(d)) {
    stop("unknown depth '", depth, "'; expected one of ", paste(names(DEPTHS), collapse = ", "), call. = FALSE)
  }
  per_sign <- 2^31 / d$stride
  list(
    dtype = dtype,
    depth = depth,
    stride = d$stride,
    per_sign = per_sign,
    n_chunks = ceiling(per_sign / CHUNK_INDEX),
    n_samples = 2 * per_sign
  )
}

## The k-th chunk of pattern indices for one sign, and the values they encode.
## Returns NULL past the end.
sweep_chunk <- function(plan, k, sign) {
  from <- (k - 1) * CHUNK_INDEX
  if (from >= plan$per_sign) {
    return(NULL)
  }
  n <- min(CHUNK_INDEX, plan$per_sign - from)
  idx <- (from + seq_len(n) - 1) * plan$stride # pattern index, 0 .. 2^31-1

  x <- if (plan$dtype == "f32") {
    f32_from_bits(as.integer(idx))
  } else {
    f64_from_words(as.integer(idx), rand_word32(n))
  }
  ## An exact sign flip: negating the value is the same as setting the sign
  ## bit, for every finite value, both zeros, both infinities and every NaN.
  if (sign < 0) {
    x <- -x
  }

  list(idx = idx, x = x, n = n)
}

## ---- 2. scoring ------------------------------------------------------------
##
## Two metrics over the same comparison.
##
##   rel  |f - g| / |g|      -- scale-free, comparable across distributions
##   ulp  |f - g| / ulp(g)   -- the unit accuracy is actually argued in
##
## The degenerate cases are pinned down identically for both, so neither is
## ever NaN and a disagreement can never be scored as agreement:
##
##   f and g bit-identical (incl. both +-Inf, incl. both NaN)  -> 0
##   g == 0 but f != 0, g non-finite and f not identical to it,
##   f NaN where g is not                                      -> Inf
##
## An Inf score means "no meaningful denominator, and they disagree". Those are
## routed to the range tracker rather than the top-K list, because they arrive
## in huge contiguous blocks (every NaN pattern, everything off the support)
## and would otherwise bury every real finding.

score_pair <- function(fx, gx, dtype) {
  ok <- (fx == gx) | (is.nan(fx) & is.nan(gx))
  ok[is.na(ok)] <- FALSE
  bad <- !ok & (gx == 0 | !is.finite(gx) | is.nan(fx))

  d <- abs(fx - gx)
  rel <- d / abs(gx)
  ulp <- d / ulp_size(gx, dtype)
  rel[ok] <- 0
  ulp[ok] <- 0
  rel[bad] <- Inf
  ulp[bad] <- Inf
  list(rel = rel, ulp = ulp, bad = bad)
}

## ---- 3. reducers -----------------------------------------------------------
##
## Each is a stateful accumulator fed one chunk at a time and drained once at
## the end. A sweep visits billions of samples and can keep only what these
## remember, so anything not reduced here is gone: that is why the histogram
## exists alongside the top-K. The top-K bounds the worst case; the histogram
## says whether the worst case is one pathological input or a systemic floor,
## and there is no way to recover it after the fact.

## -- 3a. the worst inputs, K per binade --
##
## Per binade rather than a single global list. A global top-K clusters: for
## nv_qunif f32 every one of the worst 1000 sat around x = 1/3, so the worst
## input anywhere in the tail had been computed and thrown away. Keeping K in
## each binade guarantees coverage across the whole number line and makes the
## behaviour bands drillable -- a band says how bad, these say exactly where.
##
## K is small (10) because the cost is multiplied by the number of occupied
## binades, not divided by it: per-binade top-100 would be roughly twelve times
## today's store, per-binade top-10 is about the same size.
##
## The overall worst input is still simply the first row, because the retained
## entries are ranked globally on the way out.
reducer_topk <- function(dtype, k = 10L) {
  span <- if (dtype == "f32") 2^23 else 2^20
  ## one slot per (sign, binade); an environment keyed by name keeps the
  ## occupied ones only, which is most of the saving on a well-behaved cell
  store <- new.env(parent = emptyenv())
  cut <- new.env(parent = emptyenv())

  list(
    add = function(ch, s, fx, gx, sgn) {
      i <- which(is.finite(s$rel) & s$rel > 0)
      if (!length(i)) {
        return(invisible(NULL))
      }
      b <- floor(ch$idx[i] / span)
      tag <- if (sgn > 0) "p" else "n"
      for (bb in unique(b)) {
        key <- paste0(tag, bb)
        j <- i[b == bb]
        ## Drop anything that cannot displace the current k-th worst before
        ## sorting: at full depth a chunk holds a million candidates and a full
        ## sort of every one of them, every chunk, is the whole cost.
        c0 <- cut[[key]]
        if (!is.null(c0)) {
          j <- j[s$rel[j] > c0]
        }
        if (!length(j)) {
          next
        }
        if (length(j) > k) {
          j <- j[order(s$rel[j], decreasing = TRUE)[seq_len(k)]]
        }
        cand <- data.frame(
          binade = bb,
          sign = sgn,
          x = ch$x[j],
          rel_err = s$rel[j],
          ulp_err = s$ulp[j],
          value = fx[j],
          reference = gx[j]
        )
        old <- store[[key]]
        all <- if (is.null(old)) cand else rbind(old, cand)
        all <- all[order(all$rel_err, decreasing = TRUE), , drop = FALSE]
        all <- utils::head(all, k)
        store[[key]] <- all
        if (nrow(all) == k) cut[[key]] <- all$rel_err[k]
      }
      invisible(NULL)
    },
    get = function() {
      keys <- ls(store)
      if (!length(keys)) {
        return(data.frame(
          rank = integer(0),
          binade = numeric(0),
          sign = numeric(0),
          bits = character(0),
          x = numeric(0),
          value = numeric(0),
          reference = numeric(0),
          rel_err = numeric(0),
          ulp_err = numeric(0)
        ))
      }
      d <- do.call(rbind, mget(keys, envir = store))
      d <- d[order(d$rel_err, decreasing = TRUE), , drop = FALSE]
      data.frame(
        rank = seq_len(nrow(d)),
        binade = d$binade,
        sign = d$sign,
        bits = bits_of(d$x, dtype),
        x = d$x,
        value = d$value,
        reference = d$reference,
        rel_err = d$rel_err,
        ulp_err = d$ulp_err
      )
    }
  )
}

## -- 3b. Inf-score patterns, collapsed to contiguous [lo, hi] index runs --
## Indices are walked in order, so a run of failures is a maximal stretch of
## consecutive indices; a run touching the previous chunk's tail extends it
## rather than starting a new one. One tracker per sign, so a run never
## straddles the +/- halves of the number line.
reducer_runs <- function() {
  lo <- hi <- numeric(0)
  list(
    add = function(idx, bad) {
      i <- which(bad)
      if (!length(i)) {
        return(invisible(NULL))
      }
      st <- which(c(TRUE, diff(i) != 1))
      s <- idx[i[st]]
      e <- idx[i[c(st[-1] - 1, length(i))]]
      if (length(lo) && s[1] == hi[length(hi)] + 1) {
        hi[length(hi)] <<- e[1]
        s <- s[-1]
        e <- e[-1]
      }
      lo <<- c(lo, s)
      hi <<- c(hi, e)
    },
    get = function() data.frame(lo = lo, hi = hi)
  )
}

## -- 3c. behaviour across the number line, by binade --
##
## Where does the function stop reproducing base R exactly, and where does it
## stop producing a finite answer at all? Those transitions happen in ranges,
## not at scattered points, and a histogram of error magnitudes cannot show
## them.
##
## Accumulated per binade rather than per bit pattern. Tracking runs of
## individual patterns would explode: at f32 epsilon, exact and differing
## values interleave constantly, so a well-behaved cell would produce hundreds
## of thousands of alternating one-element runs. A binade is the scale at which
## the behaviour actually changes, and there are only 256 of them in f32 and
## 2048 in f64, per sign.
reducer_bands <- function(dtype) {
  span <- if (dtype == "f32") 2^23 else 2^20 # pattern indices per binade
  nexp <- if (dtype == "f32") 256L else 2048L
  ## `m` is the decade of the relative error: floor(-log10(e)), so an error e
  ## satisfies 10^-(m+1) <= e < 10^-m. A *larger* error has a *smaller* m, so
  ## m_worst is the minimum over a binade and m_best the maximum. Both stay NA
  ## until the binade sees a finite non-zero error: a wholly exact binade has
  ## no envelope and must not pretend to one.
  mk <- function() {
    list(
      n = matrix(0, nexp, 3L),
      worst = rep(0, nexp),
      m_worst = rep(NA_real_, nexp),
      m_best = rep(NA_real_, nexp)
    )
  }
  acc <- list(mk(), mk()) # [[1]] positive, [[2]] negative

  list(
    add = function(idx, s, sign) {
      k <- if (sign > 0) 1L else 2L
      e <- floor(idx / span) + 1L
      ## 1 = identical, 2 = finite difference, 3 = no finite error
      cls <- ifelse(s$rel == 0, 1L, ifelse(is.finite(s$rel), 2L, 3L))
      for (cl in seq_len(ncol(acc[[k]]$n))) {
        i <- cls == cl
        if (any(i)) acc[[k]]$n[, cl] <- acc[[k]]$n[, cl] + tabulate(e[i], nbins = nexp)
      }
      fin <- is.finite(s$rel) & s$rel > 0
      if (any(fin)) {
        ef <- e[fin]
        rf <- s$rel[fin]
        w <- vapply(split(rf, ef), max, 0)
        j <- as.integer(names(w))
        acc[[k]]$worst[j] <- pmax(acc[[k]]$worst[j], w)

        m <- floor(-log10(rf))
        acc[[k]]$m_worst[j] <- pmin(acc[[k]]$m_worst[j], vapply(split(m, ef), min, 0), na.rm = TRUE)
        acc[[k]]$m_best[j] <- pmax(acc[[k]]$m_best[j], vapply(split(m, ef), max, 0), na.rm = TRUE)
      }
      acc <<- acc
    },
    get = function() acc,
    span = function() span
  )
}

## Merge adjacent binades that behave the same way into one reported range.
## A band's label is what makes two binades "the same": a run of binades that
## are all bit-identical collapses to a single line, and the line where that
## stops is the finding.
band_label <- function(n) {
  tot <- sum(n)
  if (tot == 0) {
    return(NA_character_)
  }
  if (n[3L] == tot) {
    return("no finite error")
  }
  if (n[1L] == tot) {
    return("all identical")
  }
  if (n[2L] == tot) {
    return("all differ")
  }
  if (n[3L] == 0) "identical + differ" else "mixed, some non-finite"
}

## The per-binade profile, unmerged: one row for every binade that was
## sampled, carrying its behaviour, its error envelope and its worst error.
##
## Merging happens at *render* time, not here. The terminal wants a dozen rows
## and a chart wants all 256 (f32) or 2048 (f64) per sign, and deriving the
## compact view from the full one keeps them consistent; storing only the
## merged form would make the chart impossible without re-running the sweep.
binade_profile <- function(bands, dtype) {
  span <- bands$span()
  acc <- bands$get()
  nexp <- nrow(acc[[1L]]$n)

  out <- lapply(seq_along(acc), function(k) {
    a <- acc[[k]]
    sgn <- if (k == 1L) 1 else -1
    lab <- vapply(seq_len(nexp), function(i) band_label(a$n[i, ]), "")
    ix <- which(!is.na(lab))
    if (!length(ix)) {
      return(NULL)
    }

    ## The highest exponent field is not an interval: it holds +-Inf (zero
    ## mantissa) and every NaN (non-zero mantissa) side by side, so decoding
    ## its upper edge yields NaN. It is flagged and given no numeric range.
    special <- ix == nexp
    lo <- (ix - 1L) * span
    hi <- ix * span - 1
    ends <- if (dtype == "f32") {
      cbind(sgn * f32_from_bits(as.integer(lo)), sgn * f32_from_bits(as.integer(hi)))
    } else {
      cbind(
        sgn * f64_from_words(as.integer(lo), 0L),
        sgn * f64_from_words(as.integer(hi), -1L)
      )
    }
    x_from <- pmin(ends[, 1L], ends[, 2L])
    x_to <- pmax(ends[, 1L], ends[, 2L])
    x_from[special] <- NA_real_
    x_to[special] <- NA_real_

    data.frame(
      sign = sgn,
      binade = ix - 1L,
      x_from = x_from,
      x_to = x_to,
      special = special,
      behaviour = lab[ix],
      n_identical = a$n[ix, 1L],
      n_differ = a$n[ix, 2L],
      n_nonfinite = a$n[ix, 3L],
      m_worst = a$m_worst[ix],
      m_best = a$m_best[ix],
      worst_rel_err = a$worst[ix]
    )
  })

  out <- do.call(rbind, out)
  if (is.null(out)) {
    return(data.frame(
      sign = numeric(0),
      binade = integer(0),
      x_from = numeric(0),
      x_to = numeric(0),
      special = logical(0),
      behaviour = character(0),
      n_identical = numeric(0),
      n_differ = numeric(0),
      n_nonfinite = numeric(0),
      m_worst = numeric(0),
      m_best = numeric(0),
      worst_rel_err = numeric(0)
    ))
  }
  out[order(out$special, out$sign, out$binade), , drop = FALSE]
}

## -- 3d. error distribution: counts per decade of relative error --
## Bin j holds the samples with rel_err in [10^j, 10^(j+1)). Exact agreement
## and Inf are counted separately, so the three totals always reconstruct the
## sample count.
HIST_LO <- -20L
HIST_HI <- 4L

reducer_hist <- function() {
  counts <- integer(HIST_HI - HIST_LO + 1L)
  n_exact <- 0
  n_inf <- 0
  list(
    add = function(s) {
      n_exact <<- n_exact + sum(s$rel == 0)
      n_inf <<- n_inf + sum(is.infinite(s$rel))
      e <- s$rel[s$rel > 0 & is.finite(s$rel)]
      if (!length(e)) {
        return(invisible(NULL))
      }
      b <- pmin(pmax(floor(log10(e)), HIST_LO), HIST_HI) - HIST_LO + 1L
      counts <<- counts + tabulate(b, nbins = length(counts))
    },
    get = function() {
      data.frame(
        decade = HIST_LO:HIST_HI,
        count = counts
      )
    },
    totals = function() list(n_exact = n_exact, n_inf = n_inf)
  )
}

## ---- the sweep -------------------------------------------------------------
##
## `fun` and `ref` each take the chunk's values and return a *named list* of
## outputs. That plural is the point: a reverse-mode pass already computes
## d/dq, d/dmin and d/dmax together, so scoring them from one sweep rather than
## re-sweeping once per argument is a straight 3x saving on what is by far the
## largest part of the grid. One sweep, one set of reducers per output.

run_sweep <- function(fun, ref, dtype, depth, outputs, progress = TRUE, topk = 10L) {
  plan <- sweep_plan(dtype, depth)
  set.seed(SWEEP_SEED)

  acc <- lapply(outputs, function(o) {
    list(
      topk = reducer_topk(dtype, topk),
      hist = reducer_hist(),
      bands = reducer_bands(dtype),
      runs = list(reducer_runs(), reducer_runs())
    )
  })
  names(acc) <- outputs

  if (progress) {
    cli::cli_progress_bar(
      format = "{cli::pb_extra$tag} {cli::pb_bar} {cli::pb_percent} | ETA {cli::pb_eta}",
      total = plan$n_chunks * 2L,
      extra = list(tag = sprintf("%s/%s", dtype, depth))
    )
  }

  t0 <- proc.time()[["elapsed"]]
  for (k in seq_len(plan$n_chunks)) {
    for (sgn in c(1, -1)) {
      ch <- sweep_chunk(plan, k, sgn)
      if (is.null(ch)) {
        next
      }
      fx <- fun(ch$x)
      gx <- ref(ch$x)
      for (o in outputs) {
        s <- score_pair(fx[[o]], gx[[o]], dtype)
        a <- acc[[o]]
        a$topk$add(ch, s, fx[[o]], gx[[o]], sgn)
        a$hist$add(s)
        a$bands$add(ch$idx, s, sgn)
        a$runs[[if (sgn > 0) 1L else 2L]]$add(ch$idx, s$bad)
      }
      if (progress) cli::cli_progress_update()
    }
  }
  elapsed <- proc.time()[["elapsed"]] - t0
  if (progress) {
    cli::cli_progress_done()
  }

  lapply(acc, function(a) {
    tot <- a$hist$totals()
    ranges <- decode_runs(a$runs, dtype)
    detail <- a$topk$get()
    have <- nrow(detail) > 0L

    list(
      detail = detail,
      hist = a$hist$get(),
      bands = binade_profile(a$bands, dtype),
      ranges = ranges,
      summary = data.frame(
        n_samples = plan$n_samples,
        n_exact = tot$n_exact,
        n_inf = tot$n_inf,
        n_inf_runs = nrow(ranges),
        worst_rel_err = if (have) detail$rel_err[1] else 0,
        worst_ulp_err = if (have) max(detail$ulp_err) else 0,
        ## The input that produced the worst error, carried on the summary row
        ## so that "how bad is it" and "at what input" can be read together.
        ## Without it every headline number needs a second lookup to mean
        ## anything, which is how a status screen stops being read.
        worst_x = if (have) detail$x[1] else NA_real_,
        worst_bits = if (have) detail$bits[1] else NA_character_,
        worst_value = if (have) detail$value[1] else NA_real_,
        worst_reference = if (have) detail$reference[1] else NA_real_,
        elapsed_sec = elapsed
      )
    )
  })
}

## Turn index runs back into the interval of the number line they cover.
## For f64 an index is a whole block of 2^32 patterns, so the endpoints are the
## exact bounds of the affected interval: low word 0x00000000 at one end and
## 0xFFFFFFFF at the other.
decode_runs <- function(runs, dtype) {
  out <- lapply(seq_along(runs), function(k) {
    r <- runs[[k]]$get()
    if (!nrow(r)) {
      return(NULL)
    }
    sgn <- if (k == 1L) 1 else -1
    if (dtype == "f32") {
      data.frame(
        bits_from = bits_of(sgn * f32_from_bits(as.integer(r$lo)), dtype),
        bits_to = bits_of(sgn * f32_from_bits(as.integer(r$hi)), dtype),
        x_from = sgn * f32_from_bits(as.integer(r$lo)),
        x_to = sgn * f32_from_bits(as.integer(r$hi)),
        n_patterns = r$hi - r$lo + 1
      )
    } else {
      lo_x <- sgn * f64_from_words(as.integer(r$lo), 0L)
      hi_x <- sgn * f64_from_words(as.integer(r$hi), -1L)
      data.frame(
        bits_from = bits_of(lo_x, dtype),
        bits_to = bits_of(hi_x, dtype),
        x_from = lo_x,
        x_to = hi_x,
        n_patterns = (r$hi - r$lo + 1) * 2^32
      )
    }
  })
  out <- do.call(rbind, out)
  if (is.null(out)) {
    data.frame(
      bits_from = character(0),
      bits_to = character(0),
      x_from = numeric(0),
      x_to = numeric(0),
      n_patterns = numeric(0)
    )
  } else {
    out
  }
}

## ---- classifying the no-finite-error regions --------------------------------
##
## When the two sides disagree with no meaningful denominator -- one is NaN, or
## the reference is zero or infinite -- the sample carries no relative error,
## and these arrive in huge contiguous blocks. The sweep collapses them into
## intervals; this labels the ones the spec can account for, so that what is
## left over is the part worth reading.
##
## This is measurement, not judgement: it describes what the numbers are, and
## nothing here decides whether a result is acceptable.

SMALLEST_NORMAL <- c(f32 = 2^-126, f64 = 2^-1022)

classify_ranges <- function(ranges, support, dtype) {
  if (is.null(ranges) || !nrow(ranges)) {
    return(character(0))
  }
  lo <- pmin(ranges$x_from, ranges$x_to)
  hi <- pmax(ranges$x_from, ranges$x_to)

  cls <- rep("unclassified", nrow(ranges))

  ## NaN sorts as neither, so test it first and on the raw endpoints.
  ##
  ## A run is NaN-only when at least one endpoint is NaN and neither is finite.
  ## The "neither is finite" half matters: the highest exponent field holds
  ## +-Inf (zero mantissa) and every NaN (non-zero mantissa) side by side, and
  ## the f64 sweep reports a block by its bounds, so a block whose sampled
  ## points are all NaN still *decodes* its lower bound as Inf. Requiring one
  ## NaN endpoint and no finite one names exactly the runs confined to that
  ## field, and no others -- a genuine disagreement at Inf alone would have two
  ## infinite endpoints and stays unclassified.
  nan <- (is.nan(ranges$x_from) | is.nan(ranges$x_to)) &
    !is.finite(ranges$x_from) &
    !is.finite(ranges$x_to)
  cls[nan] <- "nan"

  ## Every arithmetic and comparison operation on this PJRT/XLA CPU backend
  ## flushes subnormals to zero, in both precisions -- storage round-trips, but
  ## `x >= 0` is TRUE for a negative subnormal because the comparison sees -0.
  ## Nothing measured entirely inside that band describes the function under
  ## test, so a range that lies wholly below the smallest normal is explained
  ## by the platform. It is still recorded and countable: `ranges` keeps the
  ## class, so "show me everything the platform is flushing" stays one query
  ## away, and the day the backend stops flushing these turn into PASSes.
  sn <- SMALLEST_NORMAL[[dtype]]
  cls[!nan & pmax(abs(ranges$x_from), abs(ranges$x_to)) < sn] <- "subnormal"

  if (!is.null(support)) {
    s <- support
    cls[!nan & cls != "subnormal" & hi < s[1L]] <- "below_support"
    cls[!nan & cls != "subnormal" & lo > s[2L]] <- "above_support"
    ## An interval that reaches an infinity but stays outside the support is
    ## still explained by the support; one that straddles a boundary is not.
    cls[!nan & is.infinite(lo) & hi < s[1L]] <- "below_support"
    cls[!nan & is.infinite(hi) & lo > s[2L]] <- "above_support"
  }
  cls
}

## ---- expectations ----------------------------------------------------------

## The relative error above which a result is never accepted automatically.
## `baseline` refuses to write a bound for anything worse, and `status` lists
## it as needing a look, so the two screens cannot disagree about what counts
## as bad.
