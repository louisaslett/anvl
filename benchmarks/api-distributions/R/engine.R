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
## Two metrics over the same comparison, and one fact beside them.
##
##   rel      |f - g| / |g|      -- scale-free, comparable across distributions
##   ulp      |f - g| / ulp(g)   -- the unit accuracy is actually argued in
##   rounded  f is g correctly rounded to the result's precision, but not g
##
## The reference g is base R's double and is never rounded before scoring, so
## rel and ulp measure numerical error against it. `rounded` answers a
## different question: could this precision have done better? For an f32
## result the two part company at the edges of the range. Where |g| reaches
## the f32 overflow threshold (2^128 - 2^103), the rounded answer is +-Inf, and its relative
## error is still infinite; where g is below half the smallest subnormal, the
## correctly rounded answer is 0, and its relative error is still exactly 1.
## Both statements are true and both are kept: rel is not adjusted, and
## `rounded` is recorded alongside it. For f64, g is already a double, so a
## rounded result is an identical one and `rounded` is never set.
##
## The degenerate cases are pinned down identically for rel and ulp, so
## neither is ever NaN and a disagreement can never be scored as agreement:
##
##   f and g bit-identical (incl. both +-Inf, incl. both NaN)  -> 0
##   anything else without a finite score                     -> Inf
##
## "Anything else" is deliberately not a list of cases. An earlier version
## enumerated them -- g zero, g non-finite, f NaN -- and every case it missed
## scored Inf without being flagged: f = +-Inf against a finite g, an f64
## difference overflowing (-1e308 against 1e308), a finite difference over a
## tiny g overflowing. Those samples reached neither the top-K list nor the
## range tracker and could never make a result unexplained. A difference that
## overflows only because both operands are huge is not left at Inf either:
## it is recomputed on halved operands, which is exact at that magnitude.
##
## An Inf score means "no finite error, and they disagree". Those are routed to
## the range tracker rather than the top-K list, because they arrive in huge
## contiguous blocks (every NaN pattern, everything off the support) and would
## otherwise bury every real finding. A correctly rounded result is never
## routed there: it is the best the precision allows.

score_pair <- function(fx, gx, dtype) {
  ok <- (fx == gx) | (is.nan(fx) & is.nan(gx))
  ok[is.na(ok)] <- FALSE

  gr <- if (dtype == "f32") as_f32(gx) else gx
  rounded <- !ok & (fx == gr)
  rounded[is.na(rounded)] <- FALSE

  d <- abs(fx - gx)
  over <- is.infinite(d) & is.finite(fx) & is.finite(gx)
  rel <- d / abs(gx)
  ulp <- d / ulp_size(gx, dtype)
  if (any(over)) {
    dh <- abs(0.5 * fx[over] - 0.5 * gx[over])
    rel[over] <- 2 * (dh / abs(gx[over]))
    ulp[over] <- 2 * (dh / ulp_size(gx[over], dtype))
  }

  bad <- !ok & !rounded & !is.finite(rel)
  rel[ok] <- 0
  ulp[ok] <- 0
  rel[bad] <- Inf
  ulp[bad] <- Inf
  list(rel = rel, ulp = ulp, bad = bad, rounded = rounded)
}

## ---- 2b. what each side returned, and why a failure happened ---------------
##
## A relative error says how far apart two values are, not what they were, and
## a failure region says where the two sides disagreed, not why. Both are
## recorded here, and neither is inferred from the other.
##
## *Kinds.* Each side's value is one of seven kinds. "subnormal" is judged at
## the result's precision, so a double reference below the f32 range is a
## subnormal reference for an f32 cell -- which is exactly the case where an
## f32 result cannot follow it. Kinds are tallied per binade, in and out of the
## valid input domain separately, and per failure region.

KINDS <- c("nan", "+inf", "-inf", "+0", "-0", "subnormal", "normal")
N_KINDS <- length(KINDS)

value_kind <- function(v, dtype) {
  k <- rep.int(7L, length(v))
  a <- abs(v)
  ## normal is by far the common case: screen it out in one comparison and
  ## classify only the rest (NaN fails `>=`, so it lands in the rest)
  r <- which(!((a >= SMALLEST_NORMAL[[dtype]] & a < Inf) %in% TRUE))
  if (length(r)) {
    vr <- v[r]
    kr <- rep.int(6L, length(r))
    kr[vr == 0 & 1 / vr > 0] <- 4L
    kr[vr == 0 & 1 / vr < 0] <- 5L
    kr[vr == Inf] <- 2L
    kr[vr == -Inf] <- 3L
    kr[is.na(vr)] <- 1L
    k[r] <- kr
  }
  k
}

## Agreement down to the sign of zero, or both NaN (a NaN's payload and sign
## are not compared). Stricter than `==`, which cannot tell -0 from +0.
same_value <- function(a, b) {
  s <- (a == b & (a != 0 | 1 / a == 1 / b)) | (is.na(a) & is.na(b))
  s[is.na(s)] <- FALSE
  s
}

## *Causes.* Every failing sample is given exactly one cause, tested rather
## than assumed from where the input lies. In order:
##
##   nan_input                  the input is NaN
##   input_flushing             a subnormal input the backend flushed to zero:
##                              the result is bit-identical to the function's
##                              own result at the same-signed zero, AND that
##                              zero result is validated -- identical to, or
##                              the correctly rounded, reference at zero
##   flush_inherits_zero_error  as above, but the zero result is not validated
##                              (any error at all, finite or not); the
##                              subnormals inherit an error at zero
##   domain_boundary            an endpoint of the valid input domain (e.g.
##                              p = 0 or 1), or a subnormal inheriting the
##                              behaviour of a zero that is one
##   outside_domain             wholly outside the valid input domain
##   zero_input                 the input is +-0 and not a domain boundary
##   inf_input                  the input is +-Inf, inside the domain
##   unidentified               none of the above
##
## Zero is never a subnormal here: it is the value a subnormal is flushed TO,
## and it is checked, not exempted. A cause is a description; whether it counts
## against an accuracy verdict is the category, below.

CAUSES <- c(
  "nan_input", "input_flushing", "flush_inherits_zero_error", "domain_boundary",
  "outside_domain", "zero_input", "inf_input", "unidentified"
)
CAUSE_CATEGORY <- c(
  nan_input = "failure",
  input_flushing = "backend_limitation",
  flush_inherits_zero_error = "failure",
  domain_boundary = "boundary",
  outside_domain = "failure",
  zero_input = "failure",
  inf_input = "failure",
  unidentified = "failure"
)
## outside_domain is a failure where it is established: a value that should
## be NaN and is not. For a *gradient* whose forward values are both NaN it is
## an undefined-domain convention instead -- but a gradient cell never computes
## forward values, so that is settled at export from the matching value cell,
## which swept the identical inputs (see resolve_domain_conventions()).

## Per-cell context for the above: the domain, its finite endpoints at the
## cell's precision, and the function's own behaviour at +-0.
sweep_context <- function(fun, ref, dtype, outputs, domain = c(-Inf, Inf)) {
  z <- c(0, NEG_ZERO) # not the literal -0: see NEG_ZERO in util.R
  context_from(fun(z), ref(z), dtype, outputs, domain)
}

## The same context from results at c(+0, -0) already in hand.
##
## A zero result is *validated* only if it is what the precision allows: equal
## to the reference (a signed-zero disagreement is recorded, not failed) or the
## reference correctly rounded. Nothing weaker will do. Merely "has a finite
## error" -- the first version of this test -- accepted f(0) = 2 against
## g(0) = 1, and every subnormal flushed onto that zero was then excused as a
## backend limitation although it inherited a 100% error. A 1-ulp error at
## zero also fails validation: the subnormals then carry that error as well as
## the flush, and are not the flush's alone.
context_from <- function(f0, g0, dtype, outputs, domain) {
  bounds <- domain[is.finite(domain)]
  if (dtype == "f32") bounds <- as_f32(bounds)
  list(
    dtype = dtype,
    domain = domain,
    boundaries = bounds,
    zero_is_boundary = any(bounds == 0),
    zero = lapply(stats::setNames(outputs, outputs), function(o) {
      s <- score_pair(f0[[o]], g0[[o]], dtype)
      list(
        value = f0[[o]],
        reference = g0[[o]],
        rel_err = s$rel,
        validated = s$rel == 0 | s$rounded
      )
    })
  )
}

## The facts recorded for every sample of one output: kinds, domain
## membership, signed-zero disagreement, flush consistency, and, for failures,
## the cause.
sample_facts <- function(x, fx, gx, s, ctx, o) {
  dtype <- ctx$dtype
  z <- ctx$zero[[o]]
  n <- length(x)
  nanx <- is.na(x)
  ## Most of this concerns small subsets -- subnormal inputs, zero results,
  ## failures -- and is computed on those alone: the whole-chunk passes are the
  ## cost that scales with the sweep.
  ax <- abs(x)
  sub <- which(ax < SMALLEST_NORMAL[[dtype]] & ax > 0) # NaN compares FALSE
  ## which signed zero a subnormal flushes to: +0 for a positive one. Always an
  ## integer index -- ifelse() on a NaN input would yield a *logical* NA, and a
  ## logical NA index recycles, returning both zeros' results instead of one.
  zsub <- ifelse(x[sub] > 0, 1L, 2L)
  flush_same <- logical(n)
  if (length(sub)) flush_same[sub] <- same_value(fx[sub], z$value[zsub])
  in_domain <- if (ctx$domain[1L] == -Inf && ctx$domain[2L] == Inf) {
    !nanx
  } else {
    !nanx & x >= ctx$domain[1L] & x <= ctx$domain[2L]
  }
  kf <- value_kind(fx, dtype)
  kg <- value_kind(gx, dtype)

  cause <- integer(n)
  b <- which(s$bad)
  if (length(b)) {
    cb <- rep.int(8L, length(b))
    xb <- x[b]
    nb <- nanx[b]
    cb[nb] <- 1L
    fl <- flush_same[b] # FALSE unless a subnormal input matched its signed zero
    zf <- !z$validated[ifelse(!nb & xb > 0, 1L, 2L)]
    cb[fl & !zf] <- 2L
    cb[fl & zf] <- if (ctx$zero_is_boundary) 4L else 3L
    rest <- cb == 8L & !nb
    out <- rest & !in_domain[b]
    cb[out] <- 5L
    rest <- rest & !out
    bnd <- rest & xb %in% ctx$boundaries
    cb[bnd] <- 4L
    rest <- rest & !bnd
    cb[rest & xb == 0] <- 6L
    cb[rest & is.infinite(xb)] <- 7L
    cause[b] <- cb
  }

  zero_sign <- logical(n)
  zf0 <- which(fx == 0)
  if (length(zf0)) {
    zz <- zf0[gx[zf0] %in% 0]
    zero_sign[zz] <- 1 / fx[zz] != 1 / gx[zz]
  }
  ## A subnormal whose result is its signed zero's and differs from base R,
  ## split by whether that zero result was validated: only against a validated
  ## zero is the difference the flush's alone.
  flushed <- flushed_zero_error <- logical(n)
  if (length(sub)) {
    fs <- flush_same[sub] & s$rel[sub] != 0 & !s$rounded[sub]
    zv <- z$validated[zsub]
    flushed[sub] <- fs & zv
    flushed_zero_error[sub] <- fs & !zv
  }

  list(
    kf = kf,
    kg = kg,
    pair = (kf - 1L) * N_KINDS + kg,
    in_domain = in_domain,
    zero_sign = zero_sign,
    ## differs from base R, and the difference is the backend's flush: the
    ## result is what the function gives at the flushed input, and that is
    ## the correct result there
    flushed = flushed,
    ## the same flush onto a zero whose own result is not validated
    flushed_zero_error = flushed_zero_error,
    cause = cause
  )
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
      ## Zero has a shortlist of its own, as it has a band of its own: sharing
      ## binade 0's would let ten subnormals displace it. Its slot is keyed -1
      ## and stored as binade 0; `x` itself says which it is.
      b <- floor(ch$idx[i] / span)
      b[ch$x[i] %in% 0] <- -1
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
          binade = max(bb, 0),
          sign = sgn,
          x = ch$x[j],
          rel_err = s$rel[j],
          ulp_err = s$ulp[j],
          value = fx[j],
          reference = gx[j],
          rounded = s$rounded[j],
          flushed = s$flushed[j]
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
          ulp_err = numeric(0),
          rounded = logical(0),
          flushed = logical(0)
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
        ulp_err = d$ulp_err,
        rounded = d$rounded,
        flushed = d$flushed
      )
    }
  )
}

## -- 3b. Inf-score patterns, collapsed to contiguous [lo, hi] index runs --
## Indices are walked in order, so a run of failures is a maximal stretch of
## consecutive indices; a run touching the previous chunk's tail extends it
## rather than starting a new one. One tracker per sign, so a run never
## straddles the +/- halves of the number line.
reducer_runs <- function(stride) {
  ## One entry per region: its index bounds, cause, sample count, the first and
  ## last failing inputs actually evaluated, one representative sample, and a
  ## tally of what the two sides returned.
  lo <- hi <- n <- x_first <- x_last <- rep_x <- rep_f <- rep_g <- numeric(0)
  cause <- integer(0)
  pairs <- list()
  list(
    add = function(idx, bad, cause_code, x, fx, gx, pair) {
      i <- which(bad)
      if (!length(i)) {
        return(invisible(NULL))
      }
      ## A region is a maximal stretch of consecutive samples with one cause.
      ## Causes follow the input's location and the flush test, which is
      ## deterministic per sign, so this cannot fragment the way splitting on
      ## returned values could.
      cc <- cause_code[i]
      st <- which(c(TRUE, diff(i) != 1 | diff(cc) != 0))
      en <- c(st[-1] - 1L, length(i))
      for (q in seq_along(st)) {
        a <- i[st[q]]
        b <- i[en[q]]
        seg <- i[st[q]:en[q]]
        tally <- tabulate(pair[seg], nbins = N_KINDS^2)
        m <- length(lo)
        ## contiguous with the previous chunk's last region, with the same cause
        if (m && idx[a] == hi[m] + stride && cc[st[q]] == cause[m]) {
          hi[m] <<- idx[b]
          n[m] <<- n[m] + length(seg)
          x_last[m] <<- x[b]
          pairs[[m]] <<- pairs[[m]] + tally
          next
        }
        lo <<- c(lo, idx[a])
        hi <<- c(hi, idx[b])
        n <<- c(n, length(seg))
        cause <<- c(cause, cc[st[q]])
        x_first <<- c(x_first, x[a])
        x_last <<- c(x_last, x[b])
        rep_x <<- c(rep_x, x[a])
        rep_f <<- c(rep_f, fx[a])
        rep_g <<- c(rep_g, gx[a])
        pairs[[m + 1L]] <<- tally
      }
    },
    get = function() {
      describe <- function(t) {
        j <- which(t > 0)
        j <- j[order(-t[j])]
        paste(sprintf(
          "%s vs %s: %s",
          KINDS[(j - 1L) %/% N_KINDS + 1L],
          KINDS[(j - 1L) %% N_KINDS + 1L],
          format(t[j], big.mark = ",", scientific = FALSE, trim = TRUE)
        ), collapse = "; ")
      }
      top <- function(t) which.max(t)
      data.frame(
        lo = lo,
        hi = hi,
        cause = CAUSES[cause],
        n_failing = n,
        x_first = x_first,
        x_last = x_last,
        rep_x = rep_x,
        rep_value = rep_f,
        rep_reference = rep_g,
        value_kind = KINDS[(vapply(pairs, top, 1L) - 1L) %/% N_KINDS + 1L],
        reference_kind = KINDS[(vapply(pairs, top, 1L) - 1L) %% N_KINDS + 1L],
        pairs = vapply(pairs, describe, ""),
        stringsAsFactors = FALSE
      )
    }
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
## Contiguous groups of equal values in `key` (already in order), as slices.
group_slices <- function(key) {
  r <- rle(key)
  to <- cumsum(r$lengths)
  list(key = r$values, from = to - r$lengths + 1L, to = to)
}

## The position of each group's extreme value (which.max / which.min).
group_extreme <- function(g, v, pick) {
  vapply(seq_along(g$key), function(q) g$from[q] - 1L + pick(v[g$from[q]:g$to[q]]), 1L)
}

## Exact zero inputs are not a subnormal binade's samples: zero is the value a
## subnormal is flushed *to*, and its result is checked, not excused. Binade 0
## therefore holds the subnormals alone, and each sign's zero samples are kept
## in one extra slot (`nexp + 1`), emitted as a band row with `zero = TRUE`.
## The bands still partition the samples, so every sum over them is unchanged.
reducer_bands <- function(dtype) {
  span <- if (dtype == "f32") 2^23 else 2^20 # pattern indices per binade
  nexp <- if (dtype == "f32") 256L else 2048L
  nslot <- nexp + 1L
  ## `m` is the decade of the relative error: floor(-log10(e)), so an error e
  ## satisfies 10^-(m+1) <= e < 10^-m. A *larger* error has a *smaller* m, so
  ## m_worst is the minimum over a binade and m_best the maximum. Both stay NA
  ## until the binade sees a finite non-zero error: a wholly exact binade has
  ## no envelope and must not pretend to one.
  mk <- function() {
    list(
      n = matrix(0, nslot, 3L),
      ## correctly rounded but not identical; overlaps columns 2 and 3 of `n`
      rounded = rep(0, nslot),
      ## both sides zero, of opposite sign: counted as identical by `n`, which
      ## compares with ==, and recorded here so it stays visible
      zero_sign = rep(0, nslot),
      ## differs from base R because the backend flushed a subnormal input to a
      ## zero whose result is validated
      flushed = rep(0, nslot),
      ## the same, onto a zero whose result is not validated
      flushed_zero_error = rep(0, nslot),
      ## samples whose reference is a normal float at the result's precision:
      ## the ones where a small relative error is the right expectation
      out_normal = rep(0, nslot),
      out_normal_identical = rep(0, nslot),
      out_normal_worst = rep(0, nslot),
      out_normal_x = rep(NA_real_, nslot),
      out_normal_value = rep(NA_real_, nslot),
      out_normal_reference = rep(NA_real_, nslot),
      ## (value kind, reference kind) pairs, in and out of the domain, by the
      ## input's true binade (zero inputs in binade 0):
      ## nexp x 2 x N_KINDS^2, flattened
      kinds = numeric(nexp * 2L * N_KINDS^2),
      worst = rep(0, nslot),
      m_worst = rep(NA_real_, nslot),
      m_best = rep(NA_real_, nslot)
    )
  }
  acc <- list(mk(), mk()) # [[1]] positive, [[2]] negative

  list(
    add = function(idx, s, sign, x, fx, gx) {
      k <- if (sign > 0) 1L else 2L
      a <- acc[[k]]
      eb <- floor(idx / span) + 1L # the true binade, for the kinds
      e <- eb
      ## Zero is the lowest pattern index of its sign, so its slot is a
      ## contiguous group of its own at the start of a chunk.
      e[which(x == 0)] <- nslot
      ## A chunk's pattern indices increase, so its samples arrive grouped by
      ## binade and in order: every per-binade reduction below is a pass over
      ## contiguous slices, not a split() of the whole chunk. At quick depth an
      ## f64 chunk spans ~129 binades; the old split()-based version was the
      ## single most expensive step of a sweep.
      tab <- function(i) tabulate(e[i], nbins = nslot)
      rel <- s$rel
      ## 1 = identical, 2 = finite difference, 3 = no finite error
      fin <- is.finite(rel)
      same <- rel == 0
      a$n[, 1L] <- a$n[, 1L] + tab(same)
      a$n[, 2L] <- a$n[, 2L] + tab(fin & !same)
      a$n[, 3L] <- a$n[, 3L] + tab(!fin)
      if (any(s$rounded)) a$rounded <- a$rounded + tab(s$rounded)
      if (any(s$zero_sign)) a$zero_sign <- a$zero_sign + tab(s$zero_sign)
      if (any(s$flushed)) a$flushed <- a$flushed + tab(s$flushed)
      if (any(s$flushed_zero_error)) a$flushed_zero_error <- a$flushed_zero_error + tab(s$flushed_zero_error)

      ## Kinds: the bulk -- normal against normal inside the domain -- is never
      ## reported, so only the rest is tallied.
      rest <- which(s$pair != N_KINDS^2 | !s$in_domain)
      if (length(rest)) {
        code <- (eb[rest] - 1L) * (2L * N_KINDS^2) + (!s$in_domain[rest]) * N_KINDS^2 + s$pair[rest]
        a$kinds <- a$kinds + tabulate(code, nbins = nexp * 2L * N_KINDS^2)
      }

      ## Worst and best error per binade, and the decade envelope, which
      ## follows from them: m = floor(-log10(e)) is monotone in e, so the
      ## smallest m comes from the largest error and the largest m from the
      ## smallest -- no log10 over every sample.
      pos <- which(fin & !same)
      if (length(pos)) {
        g <- group_slices(e[pos])
        rp <- rel[pos]
        hi <- group_extreme(g, rp, which.max)
        lo <- group_extreme(g, rp, which.min)
        j <- g$key
        a$worst[j] <- pmax(a$worst[j], rp[hi])
        a$m_worst[j] <- pmin(a$m_worst[j], floor(-log10(rp[hi])), na.rm = TRUE)
        a$m_best[j] <- pmax(a$m_best[j], floor(-log10(rp[lo])), na.rm = TRUE)
      }

      ## Samples whose reference is a normal float: counts, and the worst
      ## with the sample that produced it.
      on <- s$kg == 7L
      if (any(on)) {
        a$out_normal <- a$out_normal + tab(on)
        a$out_normal_identical <- a$out_normal_identical + tab(on & same)
        fo <- which(on & fin & !same)
        if (length(fo)) {
          g <- group_slices(e[fo])
          w <- fo[group_extreme(g, rel[fo], which.max)]
          j <- g$key
          better <- rel[w] > a$out_normal_worst[j]
          if (any(better)) {
            w <- w[better]
            j <- j[better]
            a$out_normal_worst[j] <- rel[w]
            a$out_normal_x[j] <- x[w]
            a$out_normal_value[j] <- fx[w]
            a$out_normal_reference[j] <- gx[w]
          }
        }
      }
      acc[[k]] <<- a
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
  nslot <- nrow(acc[[1L]]$n)
  nexp <- nslot - 1L

  out <- lapply(seq_along(acc), function(k) {
    a <- acc[[k]]
    sgn <- if (k == 1L) 1 else -1
    lab <- vapply(seq_len(nslot), function(i) band_label(a$n[i, ]), "")
    ix <- which(!is.na(lab))
    if (!length(ix)) {
      return(NULL)
    }

    ## The highest exponent field is not an interval: it holds +-Inf (zero
    ## mantissa) and every NaN (non-zero mantissa) side by side, so decoding
    ## its upper edge yields NaN. It is flagged and given no numeric range.
    special <- ix == nexp
    ## The zero slot is exponent field 0 too, and covers the one value +-0.
    zero <- ix == nslot
    bin <- ifelse(zero, 1L, ix)
    lo <- (bin - 1L) * span
    hi <- bin * span - 1
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
    x_from[zero] <- x_to[zero] <- if (sgn > 0) 0 else NEG_ZERO
    ## binade 0 holds only the subnormals, so its edge nearest zero is the
    ## smallest subnormal, not zero
    sub <- !zero & bin == 1L
    if (sgn > 0) x_from[sub] <- SUBNORMAL_MIN[[dtype]] else x_to[sub] <- -SUBNORMAL_MIN[[dtype]]

    data.frame(
      sign = sgn,
      binade = bin - 1L,
      x_from = x_from,
      x_to = x_to,
      special = special,
      zero = zero,
      behaviour = lab[ix],
      n_identical = a$n[ix, 1L],
      n_differ = a$n[ix, 2L],
      n_nonfinite = a$n[ix, 3L],
      n_rounded = a$rounded[ix],
      n_zero_sign = a$zero_sign[ix],
      n_flushed = a$flushed[ix],
      n_flushed_zero_error = a$flushed_zero_error[ix],
      n_out_normal = a$out_normal[ix],
      n_out_normal_identical = a$out_normal_identical[ix],
      worst_out_normal = a$out_normal_worst[ix],
      worst_out_normal_x = a$out_normal_x[ix],
      worst_out_normal_value = a$out_normal_value[ix],
      worst_out_normal_reference = a$out_normal_reference[ix],
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
      zero = logical(0),
      behaviour = character(0),
      n_identical = numeric(0),
      n_differ = numeric(0),
      n_nonfinite = numeric(0),
      n_rounded = numeric(0),
      n_zero_sign = numeric(0),
      n_flushed = numeric(0),
      n_flushed_zero_error = numeric(0),
      n_out_normal = numeric(0),
      n_out_normal_identical = numeric(0),
      worst_out_normal = numeric(0),
      worst_out_normal_x = numeric(0),
      worst_out_normal_value = numeric(0),
      worst_out_normal_reference = numeric(0),
      m_worst = numeric(0),
      m_best = numeric(0),
      worst_rel_err = numeric(0)
    ))
  }
  out[order(out$special, out$sign, out$binade, !out$zero), , drop = FALSE]
}

## The (value kind, reference kind) tallies as a sparse table: one row per
## binade, domain side and pair that occurred. The bulk -- normal against
## normal inside the domain -- is left out; it is the complement of the rest.
kinds_table <- function(bands) {
  acc <- bands$get()
  nexp <- nrow(acc[[1L]]$n) - 1L # less the zero slot
  per <- 2L * N_KINDS^2
  out <- lapply(seq_along(acc), function(k) {
    v <- acc[[k]]$kinds
    i <- which(v > 0)
    if (!length(i)) {
      return(NULL)
    }
    r <- (i - 1L) %% per
    d <- data.frame(
      sign = if (k == 1L) 1 else -1,
      binade = (i - 1L) %/% per,
      in_domain = r < N_KINDS^2,
      value_kind = KINDS[((r %% N_KINDS^2) %/% N_KINDS) + 1L],
      reference_kind = KINDS[(r %% N_KINDS) + 1L],
      n = v[i]
    )
    d[!(d$in_domain & d$value_kind == "normal" & d$reference_kind == "normal"), , drop = FALSE]
  })
  out <- do.call(rbind, out)
  if (is.null(out)) {
    out <- data.frame(sign = numeric(0), binade = numeric(0), in_domain = logical(0),
      value_kind = character(0), reference_kind = character(0), n = numeric(0))
  }
  out
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
  n_rounded <- 0
  list(
    add = function(s) {
      n_exact <<- n_exact + sum(s$rel == 0)
      n_inf <<- n_inf + sum(is.infinite(s$rel))
      n_rounded <<- n_rounded + sum(s$rounded)
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
    totals = function() list(n_exact = n_exact, n_inf = n_inf, n_rounded = n_rounded)
  )
}

## ---- exact points ----------------------------------------------------------
##
## A mandatory check beside every sweep, at the inputs a sweep can miss: the
## f64 sweep draws its low 32 bits at random, so it essentially never lands on
## +-0, +-Inf, or an exact point such as p = 1. Each cell evaluates:
##
##   universal         +-0, +-Inf, NaN, the smallest and largest subnormal,
##                     the smallest normal, the largest finite value, +-0.5, +-1
##   domain_boundary   the finite endpoints of the valid input domain
##   support_edge      the finite edges of the distribution's support
##   branch            where anvl's implementation switches algorithm
##
## Every point is taken at the cell's precision -- a boundary of an f32 cell is
## the boundary after conversion to f32 -- and every finite one comes with its
## two representable neighbours, universal points included: an off-by-one-ulp
## threshold is as likely at 1/2 or 1 as at a declared edge. The results are kept in their own table and are
## never added to the sweep's counts, histograms or bands, so a point the sweep
## also visited is not counted twice.

universal_points <- function(dtype) {
  smin <- SUBNORMAL_MIN[[dtype]]
  nmin <- SMALLEST_NORMAL[[dtype]]
  big <- if (dtype == "f32") (2 - 2^-23) * 2^127 else .Machine$double.xmax
  v <- c(zero = 0, inf = Inf, subnormal_min = smin, subnormal_max = nmin - smin,
    normal_min = nmin, finite_max = big, half = 0.5, one = 1)
  c(stats::setNames(v, paste0("+", names(v))), stats::setNames(-v, paste0("-", names(v))), nan = NaN)
}

exact_points <- function(dtype, domain = c(-Inf, Inf), support = NULL, branch = NULL) {
  at <- function(v) if (dtype == "f32") as_f32(v) else v
  u <- universal_points(dtype)
  pts <- data.frame(label = names(u), role = "universal", x = unname(u), stringsAsFactors = FALSE)
  decl <- rbind(
    data.frame(label = c("domain_lo", "domain_hi"), role = "domain_boundary", x = domain),
    if (length(support)) data.frame(label = c("support_lo", "support_hi"), role = "support_edge", x = support),
    if (length(branch)) data.frame(label = paste0("branch_", names(branch)), role = "branch", x = unname(branch))
  )
  decl <- decl[is.finite(decl$x), , drop = FALSE]
  decl$x <- at(decl$x)
  pts <- rbind(pts, decl)
  f <- pts[is.finite(pts$x), , drop = FALSE]
  nb <- float_neighbours(f$x, dtype)
  pts <- rbind(
    pts,
    data.frame(label = paste0(f$label, "_below"), role = paste0(f$role, "_neighbour"), x = nb[, 1L]),
    data.frame(label = paste0(f$label, "_above"), role = paste0(f$role, "_neighbour"), x = nb[, 2L])
  )
  pts <- pts[!is.na(pts$x) | pts$label == "nan", , drop = FALSE]
  ## One row per bit pattern, keeping every label it arrived under.
  pts$bits <- bits_of(pts$x, dtype)
  key <- factor(pts$bits, levels = unique(pts$bits))
  data.frame(
    label = vapply(split(pts$label, key), function(v) paste(unique(v), collapse = "+"), ""),
    role = vapply(split(pts$role, key), function(v) paste(unique(v), collapse = "+"), ""),
    x = vapply(split(pts$x, key), `[`, 0, 1L),
    bits = levels(key),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

## A points table with no rows, for results that have none (a failed cell).
NO_POINTS <- data.frame(
  label = character(0), x = numeric(0), rel_err = numeric(0), identical = logical(0),
  zero_sign = logical(0), rounded = logical(0), failure = logical(0), category = character(0)
)

## Returns the points' results with, as attribute "context", the context the
## sweep needs: the points include +-0, so one evaluation serves both, and a
## backend that compiles per input shape compiles one extra shape, not two.
run_points <- function(fun, ref, dtype, outputs, pts, domain = c(-Inf, Inf)) {
  fx <- fun(pts$x)
  gx <- ref(pts$x)
  pz <- which(pts$x == 0 & 1 / pts$x > 0)[1L]
  nz <- which(pts$x == 0 & 1 / pts$x < 0)[1L]
  ctx <- context_from(
    lapply(fx, `[`, c(pz, nz)), lapply(gx, `[`, c(pz, nz)), dtype, outputs, domain
  )
  out <- lapply(outputs, function(o) {
    s <- score_pair(fx[[o]], gx[[o]], dtype)
    f <- sample_facts(pts$x, fx[[o]], gx[[o]], s, ctx, o)
    cause <- ifelse(s$bad, CAUSES[pmax(f$cause, 1L)], NA_character_)
    data.frame(
      output = o,
      label = pts$label,
      role = pts$role,
      x = pts$x,
      bits = pts$bits,
      value = fx[[o]],
      reference = gx[[o]],
      rel_err = s$rel,
      ulp_err = s$ulp,
      identical = s$rel == 0 & !s$bad,
      rounded = s$rounded,
      zero_sign = f$zero_sign,
      flushed = f$flushed,
      flushed_zero_error = f$flushed_zero_error,
      in_domain = f$in_domain,
      value_kind = KINDS[f$kf],
      reference_kind = KINDS[f$kg],
      failure = s$bad,
      cause = cause,
      category = unname(CAUSE_CATEGORY[cause]),
      stringsAsFactors = FALSE
    )
  })
  structure(do.call(rbind, out), context = ctx)
}

## ---- the sweep -------------------------------------------------------------
##
## `fun` and `ref` each take the chunk's values and return a *named list* of
## outputs. That plural is the point: a reverse-mode pass already computes
## d/dq, d/dmin and d/dmax together, so scoring them from one sweep rather than
## re-sweeping once per argument is a straight 3x saving on what is by far the
## largest part of the grid. One sweep, one set of reducers per output.

run_sweep <- function(fun, ref, dtype, depth, outputs, progress = TRUE, topk = 10L,
                      domain = c(-Inf, Inf), ctx = NULL) {
  plan <- sweep_plan(dtype, depth)
  ## The behaviour at +-0 is taken before seeding, so it cannot shift the
  ## random stream that the f64 samples are drawn from. A caller that has
  ## already evaluated the exact points passes it in (see run_points()).
  if (is.null(ctx)) ctx <- sweep_context(fun, ref, dtype, outputs, domain)
  set.seed(SWEEP_SEED)

  acc <- lapply(outputs, function(o) {
    list(
      topk = reducer_topk(dtype, topk),
      hist = reducer_hist(),
      bands = reducer_bands(dtype),
      runs = list(reducer_runs(plan$stride), reducer_runs(plan$stride))
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
        s <- c(s, sample_facts(ch$x, fx[[o]], gx[[o]], s, ctx, o))
        a <- acc[[o]]
        a$topk$add(ch, s, fx[[o]], gx[[o]], sgn)
        a$hist$add(s)
        a$bands$add(ch$idx, s, sgn, ch$x, fx[[o]], gx[[o]])
        a$runs[[if (sgn > 0) 1L else 2L]]$add(ch$idx, s$bad, s$cause, ch$x, fx[[o]], gx[[o]], s$pair)
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
    bands <- binade_profile(a$bands, dtype)

    list(
      detail = detail,
      hist = a$hist$get(),
      bands = bands,
      kinds = kinds_table(a$bands),
      ranges = ranges,
      summary = data.frame(
        n_samples = plan$n_samples,
        n_exact = tot$n_exact,
        ## correctly rounded to the result's precision without being identical;
        ## counted apart from n_exact, since its relative error is not zero
        n_rounded = tot$n_rounded,
        n_inf = tot$n_inf,
        n_zero_sign = sum(bands$n_zero_sign),
        n_flushed = sum(bands$n_flushed),
        n_flushed_zero_error = sum(bands$n_flushed_zero_error),
        ## normal reference outputs, over every input: the population where a
        ## small relative error is the right expectation
        n_out_normal = sum(bands$n_out_normal),
        n_out_normal_identical = sum(bands$n_out_normal_identical),
        worst_out_normal = if (nrow(bands)) max(bands$worst_out_normal) else 0,
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
  span <- if (dtype == "f32") 2^23 else 2^20
  out <- lapply(seq_along(runs), function(k) {
    r <- runs[[k]]$get()
    if (!nrow(r)) {
      return(NULL)
    }
    sgn <- if (k == 1L) 1 else -1
    ## For f32 every pattern in a region's bounds that the sweep visited was
    ## evaluated, so bounds and sampled points coincide. For f64 the bounds
    ## are those of the 2^32-pattern blocks the failing samples fell in -- not
    ## evidence that the endpoints themselves were evaluated. The sampled
    ## points are carried separately, and are the ones to cite.
    if (dtype == "f32") {
      lo_x <- sgn * f32_from_bits(as.integer(r$lo))
      hi_x <- sgn * f32_from_bits(as.integer(r$hi))
      n_pat <- r$hi - r$lo + 1
    } else {
      lo_x <- sgn * f64_from_words(as.integer(r$lo), 0L)
      hi_x <- sgn * f64_from_words(as.integer(r$hi), -1L)
      n_pat <- (r$hi - r$lo + 1) * 2^32
    }
    data.frame(
      bits_from = bits_of(lo_x, dtype),
      bits_to = bits_of(hi_x, dtype),
      x_from = lo_x,
      x_to = hi_x,
      n_patterns = n_pat,
      bounds_are_samples = dtype == "f32",
      sampled_from = r$x_first,
      sampled_to = r$x_last,
      sampled_bits_from = bits_of(r$x_first, dtype),
      sampled_bits_to = bits_of(r$x_last, dtype),
      n_failing = r$n_failing,
      sign = sgn,
      binade_from = floor(r$lo / span),
      binade_to = floor(r$hi / span),
      cause = r$cause,
      category = unname(CAUSE_CATEGORY[r$cause]),
      value_kind = r$value_kind,
      reference_kind = r$reference_kind,
      pairs = r$pairs,
      rep_x = r$rep_x,
      rep_bits = bits_of(r$rep_x, dtype),
      rep_value = r$rep_value,
      rep_reference = r$rep_reference,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, out)
  if (is.null(out)) {
    out <- data.frame(
      bits_from = character(0), bits_to = character(0), x_from = numeric(0), x_to = numeric(0),
      n_patterns = numeric(0), bounds_are_samples = logical(0), sampled_from = numeric(0),
      sampled_to = numeric(0), sampled_bits_from = character(0), sampled_bits_to = character(0),
      n_failing = numeric(0), sign = numeric(0), binade_from = numeric(0), binade_to = numeric(0),
      cause = character(0), category = character(0), value_kind = character(0),
      reference_kind = character(0), pairs = character(0), rep_x = numeric(0),
      rep_bits = character(0), rep_value = numeric(0), rep_reference = numeric(0),
      stringsAsFactors = FALSE
    )
  }
  out
}

## ---- what the failure regions add up to ------------------------------------
##
## Each region carries one cause (see sample_facts()), and each cause one
## category. The categories are what a reader weighs:
##
##   failure             numerical or behavioural failure on valid inputs --
##                       what "unexplained" counts
##   backend_limitation  the platform, not the function: input flushing
##   boundary            behaviour at an endpoint of the valid input domain,
##                       which needs an explicit convention or limiting value
##   undefined_domain    a gradient outside the valid input domain, where the
##                       forward values on both sides are NaN, so no derivative
##                       is defined and the two sides only differ in convention
##
## All four stay visible. Only the last is set aside from an accuracy verdict,
## and setting it aside validates nothing.

SMALLEST_NORMAL <- c(f32 = 2^-126, f64 = 2^-1022)
SUBNORMAL_MIN <- c(f32 = 2^-149, f64 = 2^-1074)

## One (cell, output)'s regions, summarised for its results row: how many in
## each category, and where the first failure is.
region_summary <- function(ranges) {
  cat_n <- function(k) sum(ranges$category == k)
  ## failing samples, as opposed to regions: what lets a reader check that
  ## every non-identical sample is accounted for by one category or another
  cat_s <- function(k) sum(ranges$n_failing[ranges$category == k])
  f <- which(ranges$category == "failure")
  list(
    n_runs_unclassified = cat_n("failure"),
    n_regions_backend = cat_n("backend_limitation"),
    n_regions_boundary = cat_n("boundary"),
    n_regions_domain = cat_n("undefined_domain"),
    n_failing_failure = cat_s("failure"),
    n_failing_backend = cat_s("backend_limitation"),
    n_failing_boundary = cat_s("boundary"),
    n_failing_domain = cat_s("undefined_domain"),
    unexplained_from = if (length(f)) ranges$x_from[f[1L]] else NA_real_,
    unexplained_to = if (length(f)) ranges$x_to[f[1L]] else NA_real_
  )
}

## One (cell, output)'s exact points, summarised for its results row. Kept
## apart from every sweep count: a point the sweep also visited is not counted
## twice, and a point the sweep never visits -- most of them, in f64 -- still
## reaches the summary and every screen built on it.
point_summary <- function(points) {
  n <- nrow(points)
  cat_n <- function(k) sum(points$failure & points$category %in% k)
  fin <- is.finite(points$rel_err) & points$rel_err > 0
  w <- if (any(fin)) which(fin)[which.max(points$rel_err[fin])] else NA_integer_
  f <- which(points$failure & points$category %in% "failure")
  list(
    n_points = n,
    ## identical down to the sign of zero
    n_points_identical = sum(points$identical & !points$zero_sign),
    n_points_zero_sign = sum(points$zero_sign),
    n_points_rounded = sum(points$rounded),
    n_points_finite_error = sum(fin & !points$rounded),
    n_points_failure = cat_n("failure"),
    n_points_backend = cat_n("backend_limitation"),
    n_points_boundary = cat_n("boundary"),
    n_points_domain = cat_n("undefined_domain"),
    worst_point_rel_err = if (is.na(w)) 0 else points$rel_err[w],
    worst_point_label = if (is.na(w)) NA_character_ else points$label[w],
    worst_point_x = if (is.na(w)) NA_real_ else points$x[w],
    first_point_failure = if (length(f)) points$label[f[1L]] else NA_character_,
    first_point_failure_x = if (length(f)) points$x[f[1L]] else NA_real_
  )
}

## Settle outside-domain gradient regions against their value cell. A gradient
## region outside the valid input domain is an undefined-domain convention only
## where the matching value cell found BOTH forward values NaN for every
## out-of-domain sample in every binade the region touches.
##
## The compatibility rule for that evidence is: **the same run**. The value
## cell must have been swept in the gradient region's own run (same run_id),
## which fixes the platform, the anvl build, the harness, the depth and the
## seed together -- so "the identical inputs" is true by construction rather
## than by assumption. Evidence from any other run, however similar, is not
## used; a gradient cell re-run on its own simply stays a failure, and says why.
##
## `ranges` and `kinds` must be the whole store (or everything for the runs in
## question), never a presentation subset: every reader goes through
## resolved_ranges(), so the terminal and the export see the same evidence.
##
## Each candidate gets an `evidence` note; every other region gets NA.
resolve_domain_conventions <- function(ranges, kinds) {
  if (is.null(ranges) || !nrow(ranges)) {
    return(ranges)
  }
  ranges$evidence <- NA_character_
  seg <- strsplit(ranges$cell_id, "/", fixed = TRUE)
  is_grad <- vapply(seg, function(p) p[4L] == "grad", TRUE)
  cand <- which(ranges$cause == "outside_domain" & is_grad)
  if (!length(cand)) {
    return(ranges)
  }
  ranges$evidence[cand] <- "no value cell swept in this run"
  if (is.null(kinds) || !nrow(kinds)) {
    return(ranges)
  }
  out <- kinds[!kinds$in_domain & kinds$output == "value", , drop = FALSE]
  key <- paste(out$run_id, out$cell_id, out$sign, out$binade, sep = "\r")
  total <- tapply(out$n, key, sum)
  both_nan <- tapply(out$n * (out$value_kind == "nan" & out$reference_kind == "nan"), key, sum)
  ## a value cell of the run with any out-of-domain tally at all
  swept <- unique(paste(out$run_id, out$cell_id, sep = "\r"))
  for (i in cand) {
    p <- seg[[i]]
    p[4L] <- "value"
    vid <- paste(p, collapse = "/")
    if (!paste(ranges$run_id[i], vid, sep = "\r") %in% swept) {
      next
    }
    b <- seq(ranges$binade_from[i], ranges$binade_to[i])
    k <- paste(ranges$run_id[i], vid, ranges$sign[i], b, sep = "\r")
    tt <- total[k]
    nn <- both_nan[k]
    if (all(!is.na(tt)) && all(tt > 0) && all(nn == tt)) {
      ranges$category[i] <- "undefined_domain"
      ranges$evidence[i] <- "value cell, same run: both forward values NaN throughout"
    } else {
      ranges$evidence[i] <- "value cell, same run: forward values not NaN/NaN throughout"
    }
  }
  ranges
}

## ---- input categories ------------------------------------------------------
##
## Relative error is only a meaningful measure for ordinary finite inputs inside
## the valid input domain. Everywhere else the right question is whether the
## result matches base R exactly, and folding both into one "worst relative
## error" let an output flushed to zero, or a NaN, stand in for how accurate a
## function is. So every band is assigned one input class, by its exponent
## field:
##
##   normal          binades 1 .. top-1, inside the domain
##   zero            the sweep's +-0 samples, from their own band rows. Zero is
##                   the value a subnormal is flushed to, and is checked like
##                   any other input; it is also in the `points` table.
##   subnormal       binade 0 without its zeros: flushed to zero on entry by
##                   this backend
##   outside_domain  a finite band lying wholly outside the valid input domain,
##                   where the value is NaN by specification
##   inf_nan         the top field: both infinities and every NaN
##
## The distribution's support plays no part: a CDF below its support is an
## ordinary input with an ordinary answer. Within each class the figures for
## samples whose reference is a *normal* float are kept apart (n_out_normal,
## worst_out_normal): normal input and normal output is where a small relative
## error is the right expectation.
##
## Classification is per binade, so a binade straddling a domain edge counts as
## inside it. `x_from`/`x_to` are the smallest and largest values in the band on
## either sign, so "wholly outside" is exact.

input_class <- function(bands, lo, hi) {
  zero <- if (is.null(bands$zero)) rep(FALSE, nrow(bands)) else bands$zero
  cls <- rep("normal", nrow(bands))
  cls[bands$binade == 0L] <- "subnormal"
  cls[zero] <- "zero"
  outside <- !bands$special & bands$binade != 0L & (bands$x_to < lo | bands$x_from > hi)
  cls[outside %in% TRUE] <- "outside_domain"
  cls[bands$special] <- "inf_nan"
  cls
}

## One row per (run, cell, output, input class): sample counts and the worst
## error, both from the bands, whose accumulators saw every sample; and the
## input that produced that worst from `detail`. The detail table keeps the
## top-K per binade and one for each sign's zero, so it holds every band's
## worst -- and where it somehow does not, the sample is left NA rather than
## replaced by a lesser one.
category_table <- function(bands, detail, bounds) {
  lo <- bounds$domain_lo[match(bands$cell_id, bounds$cell_id)]
  hi <- bounds$domain_hi[match(bands$cell_id, bounds$cell_id)]
  bands$input_class <- input_class(bands, lo, hi)
  ## NA for a band from a store written before a column existed: unknown, and
  ## summed as unknown rather than as zero
  col <- function(nm) if (is.null(bands[[nm]])) NA_real_ else bands[[nm]]

  grp <- paste(bands$run_id, bands$cell_id, bands$output, bands$input_class, sep = "\r")
  counts <- rowsum(
    cbind(
      n = bands$n_identical + bands$n_differ + bands$n_nonfinite,
      n_identical = bands$n_identical,
      n_rounded = col("n_rounded"),
      n_differ = bands$n_differ,
      n_nonfinite = bands$n_nonfinite,
      n_zero_sign = col("n_zero_sign"),
      n_flushed = col("n_flushed"),
      n_flushed_zero_error = col("n_flushed_zero_error"),
      n_out_normal = col("n_out_normal"),
      n_out_normal_identical = col("n_out_normal_identical")
    ),
    grp,
    reorder = FALSE
  )
  first <- !duplicated(grp)
  out <- bands[first, c("run_id", "cell_id", "output", "input_class")]
  out <- cbind(out, counts[grp[first], , drop = FALSE])

  ## a detail row belongs to its band, and a zero input to the zero band
  bzero <- if (is.null(bands$zero)) rep(FALSE, nrow(bands)) else bands$zero
  bkey <- paste(bands$run_id, bands$cell_id, bands$output, bands$sign, bands$binade, bzero, sep = "\r")
  dkey <- paste(detail$run_id, detail$cell_id, detail$output, detail$sign, detail$binade,
    !is.null(bands$zero) & detail$x %in% 0, sep = "\r")
  dgrp <- paste(detail$run_id, detail$cell_id, detail$output,
    bands$input_class[match(dkey, bkey)], sep = "\r")
  o <- order(dgrp, -detail$rel_err)
  top <- o[!duplicated(dgrp[o])]
  m <- match(grp[first], dgrp[top])
  w <- detail[top[m], , drop = FALSE]
  ## The class's worst error is the maximum over its bands' own accumulators,
  ## which saw every sample; `detail` only supplies the input that produced
  ## it. 0 means no finite non-zero error in the class -- every sample was
  ## identical or had no finite error at all; the counts say which.
  out$worst_rel_err <- unname(tapply(bands$worst_rel_err, grp, max)[grp[first]])
  ## a retained sample that is not the class's worst would mislead: drop it
  stale <- is.na(m) | w$rel_err != out$worst_rel_err
  w[stale, c("x", "value", "reference")] <- NA_real_
  w$bits[stale] <- NA_character_
  out$worst_x <- w$x
  out$worst_bits <- w$bits
  out$worst_value <- w$value
  out$worst_reference <- w$reference

  ## The worst among samples with a normal reference, and the sample itself,
  ## from whichever band holds it.
  if (!is.null(bands$worst_out_normal)) {
    ob <- order(grp, -bands$worst_out_normal)
    tb <- ob[!duplicated(grp[ob])]
    mb <- match(grp[first], grp[tb])
    wb <- bands[tb[mb], , drop = FALSE]
    out$worst_out_normal <- wb$worst_out_normal
    out$worst_out_normal_x <- wb$worst_out_normal_x
    out$worst_out_normal_value <- wb$worst_out_normal_value
    out$worst_out_normal_reference <- wb$worst_out_normal_reference
  }
  rownames(out) <- NULL
  out[order(out$cell_id, out$output, out$input_class), , drop = FALSE]
}

## The exact points' counterpart of resolve_domain_conventions(), under the
## same rule: a gradient point outside the valid input domain is an
## undefined-domain convention only if the value cell *of the same run*
## evaluated the same bit pattern and found both values NaN.
resolve_point_conventions <- function(points) {
  if (is.null(points) || !nrow(points)) {
    return(points)
  }
  points$evidence <- NA_character_
  seg <- strsplit(points$cell_id, "/", fixed = TRUE)
  is_grad <- vapply(seg, function(p) p[4L] == "grad", TRUE)
  cand <- which(points$failure & points$cause %in% "outside_domain" & is_grad)
  if (!length(cand)) {
    return(points)
  }
  vid <- vapply(seg[cand], function(p) {
    p[4L] <- "value"
    paste(p, collapse = "/")
  }, "")
  vals <- points[points$output == "value", , drop = FALSE]
  j <- match(
    paste(points$run_id[cand], vid, points$bits[cand], sep = "\r"),
    paste(vals$run_id, vals$cell_id, vals$bits, sep = "\r")
  )
  ok <- !is.na(j) & vals$value_kind[j] %in% "nan" & vals$reference_kind[j] %in% "nan"
  points$category[cand[ok]] <- "undefined_domain"
  points$evidence[cand] <- ifelse(
    is.na(j),
    "no value cell evaluated this point in this run",
    ifelse(ok, "value cell, same run: both values NaN", "value cell, same run: values not NaN/NaN")
  )
  points
}
