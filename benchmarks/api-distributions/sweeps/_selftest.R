## ---------------------------------------------------------------------------
## A synthetic family that exercises the contract, with no anvl in sight.
##
## It exists because a sweep that silently sweeps nothing is indistinguishable
## from a sweep that found nothing, and at full depth a cell costs minutes --
## far too expensive a way to discover that a filter was misspelled or a
## reducer stopped reducing. Every mechanism the real specs rely on is used
## here, and the errors are injected at known places so the engine has to find
## them.
##
##   value/clean    reference reproduced exactly            -> expect 0
##   value/nudged   one ulp added on a known interval       -> expect ~1 ulp
##   grad           two outputs from one pass, one broken   -> expect a FAIL
##   ranges         a deliberate NaN-only disagreement      -> expect "nan"
##
## Run it with:  Rscript run.R selftest
## ---------------------------------------------------------------------------

nudge <- function(x, lo, hi) {
  ## add one ulp, but only on [lo, hi], so the sweep has to locate the interval
  i <- !is.na(x) & x >= lo & x <= hi
  x[i] <- x[i] + ulp_size(x[i], "f64")
  x
}

sweep_spec(
  name = "selftest",
  family = "_selftest",
  blurb = "synthetic family exercising the harness contract",
  primary = "x",
  dtypes = c("f32", "f64"),
  params = list(clean = list(err = 0), nudged = list(err = 1)),
  flags = list(broken = c(FALSE, TRUE)),

  ## Support is the whole line, so only NaN should ever be classified -- which
  ## is exactly the assertion: an unclassified range here means the classifier
  ## has drifted.
  support = function(params, flags) c(-Inf, Inf),
  value = function(x, dtype, params, flags) {
    y <- abs(x)
    if (params$err > 0) {
      y <- nudge(y, 1, 2)
    }
    if (isTRUE(flags$broken)) {
      y[!is.na(x) & x > 1e300] <- 0
    }
    if (dtype == "f32") as_f32(y) else y
  },
  ref_value = function(x, params, flags) {
    abs(x)
  },

  ## Two outputs from one call, which is the mechanism the gradient cells use.
  grad_wrt = c("x", "scale"),
  grad = function(x, dtype, params, flags) {
    g <- list(x = sign(x), scale = abs(x))
    if (isTRUE(flags$broken)) {
      g$scale <- g$scale * (1 + 1e-6)
    }
    if (dtype == "f32") lapply(g, as_f32) else g
  },
  ref_grad = function(x, params, flags) {
    list(x = sign(x), scale = abs(x))
  }
)
