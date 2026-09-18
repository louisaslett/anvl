## ---------------------------------------------------------------------------
## nv_qunif against base R qunif().
##
## The log_p variants are the interesting ones: lower_tail goes through exp(p)
## and upper_tail through -expm1(p), which behave very differently as p
## approaches 0 from below.
##
## Note that `unit` is a weak variant here in a way it is not for the density:
## with min = 0 and width 1 the whole computation collapses to the identity.
## It is kept because a regression that breaks the identity is worth catching
## cheaply, but `wide` is what actually exercises the affine map.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_uniform.R"), local = TRUE)

## x = min + u * w, with u the fraction of the way up the interval. Hence
##   d/dp  = w * du/dp      d/dmin = 1 - u      d/dmax = u
## and du/dp is 1 on the probability scale, exp(p) on the log scale, negated
## in both cases for the upper tail. Outside the valid range of p the result is
## constant and every derivative is 0.
qunif_grad_ref <- function(pr, p, f) {
  a <- p$min
  b <- p$max
  w <- b - a
  lower <- isTRUE(f$lower_tail)
  if (isTRUE(f$log_p)) {
    in_range <- !is.nan(pr) & pr <= 0
    u <- if (lower) exp(pr) else -expm1(pr)
    u1 <- if (lower) -expm1(pr) else exp(pr)
    dp <- if (lower) w * exp(pr) else -w * exp(pr)
  } else {
    in_range <- !is.nan(pr) & pr >= 0 & pr <= 1
    u <- if (lower) pr else 1 - pr
    u1 <- if (lower) 1 - pr else pr
    dp <- if (lower) w else -w
  }
  z <- list(p = dp, min = u1, max = u)
  lapply(z, function(v) ifelse(is.nan(pr), NaN, ifelse(in_range, v, 0)))
}

grad_qunif <- anvl::jit(
  anvl::gradient(
    \(p, min, max, lower_tail = TRUE, log_p = FALSE) {
      sum(anvl::nv_qunif(p, min, max, lower_tail = lower_tail, log_p = log_p))
    },
    wrt = c("p", "min", "max")
  ),
  static = c("lower_tail", "log_p")
)

sweep_spec(
  name = "nv_qunif",
  family = "uniform",
  blurb = "uniform quantile function against base R qunif()",
  primary = "p",
  params = UNIF_INTERVALS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),

  ## A quantile function is only defined on its probability scale; base R
  ## returns NaN with a warning off it, so those regions are explained rather
  ## than reported. This is the one place the support really bites, and it is
  ## why classification beats counting: without it every qunif cell reports
  ## billions of "failures" that are simply p outside [0, 1].
  support = function(p, f) if (isTRUE(f$log_p)) c(-Inf, 0) else c(0, 1),
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_qunif(
      anvl::nv_array(x, dtype = dtype),
      p$min,
      p$max,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    suppressWarnings(qunif(x, min = p$min, max = p$max, lower.tail = f$lower_tail, log.p = f$log_p))
  },

  grad_wrt = c("p", "min", "max"),
  grad = function(x, dtype, p, f) {
    n <- length(x)
    d <- grad_qunif(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$min, n), dtype = dtype),
      anvl::nv_array(rep(p$max, n), dtype = dtype),
      lower_tail = f$lower_tail,
      log_p = f$log_p
    )
    lapply(list(p = d$p, min = d$min, max = d$max), as.double)
  },
  ref_grad = qunif_grad_ref,

  ## jax.scipy.stats.uniform has ppf on the probability scale only.
  jax_covers = function(f, kind) isTRUE(f$lower_tail) && !isTRUE(f$log_p),
  jax_value = function(x, dtype, p, f) {
    jax_init()
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    js <- reticulate::import("jax.scipy.stats", convert = FALSE)
    np <- reticulate::import("numpy")
    as.double(np$asarray(js$uniform$ppf(
      jnp$asarray(x, dtype = jax_dtype(dtype)),
      p$min,
      p$max - p$min
    )))
  },
  jax_grad = function(x, dtype, p, f) {
    jax_init()
    reticulate::py_run_string(
      "
import jax
from jax.scipy.stats import uniform as _u
def _qunif_grad(i):
    g = jax.grad(lambda p, mn, mx: _u.ppf(p, mn, mx - mn), argnums=i)
    return jax.jit(jax.vmap(g, in_axes=(0, None, None)))
"
    )
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    np <- reticulate::import("numpy")
    xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
    g <- function(i) as.double(np$asarray(reticulate::py$`_qunif_grad`(i)(xx, p$min, p$max)))
    list(p = g(0L), min = g(1L), max = g(2L))
  }
)
