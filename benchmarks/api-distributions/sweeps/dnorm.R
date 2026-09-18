## ---------------------------------------------------------------------------
## nv_dnorm against base R dnorm().
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_normal.R"), local = TRUE)

## With z = (x - mean)/sd and phi the density itself,
##
##   d/dx    -z/sd * phi        log:  -z/sd
##   d/dmean  z/sd * phi        log:   z/sd
##   d/dsd   (z^2 - 1)/sd * phi log:  (z^2 - 1)/sd
##
## The log forms are the derivative of  -log(sd) - log(2 pi)/2 - z^2/2, and are
## the accurate ones to compare against: on the probability scale the far tail
## multiplies a large bracket by a density that has already underflowed, so the
## reference itself goes to zero long before the formula does.
dnorm_grad_ref <- function(x, p, f) {
  z <- (x - p$mean) / p$sd
  core <- list(x = -z / p$sd, mean = z / p$sd, sd = (z^2 - 1) / p$sd)
  if (isTRUE(f$log)) core else lapply(core, function(v) v * dnorm(x, p$mean, p$sd))
}

grad_dnorm <- anvl::jit(
  anvl::gradient(
    \(x, mean, sd, log = FALSE) sum(anvl::nv_dnorm(x, mean, sd, log = log)),
    wrt = c("x", "mean", "sd")
  ),
  static = "log"
)

sweep_spec(
  name = "nv_dnorm",
  family = "normal",
  blurb = "normal density against base R dnorm()",
  primary = "x",
  params = NORM_PARAMS,
  flags = list(log = c(FALSE, TRUE)),
  support = function(p, f) c(-Inf, Inf),
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_dnorm(anvl::nv_array(x, dtype = dtype), p$mean, p$sd, log = f$log))
  },
  ref_value = function(x, p, f) dnorm(x, mean = p$mean, sd = p$sd, log = f$log),

  grad_wrt = c("x", "mean", "sd"),
  grad = function(x, dtype, p, f) {
    n <- length(x)
    d <- grad_dnorm(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$mean, n), dtype = dtype),
      anvl::nv_array(rep(p$sd, n), dtype = dtype),
      log = f$log
    )
    lapply(list(x = d$x, mean = d$mean, sd = d$sd), as.double)
  },
  ref_grad = dnorm_grad_ref,

  ## jax.scipy.stats.norm covers both pdf and logpdf, so every variant twins.
  jax_value = function(x, dtype, p, f) {
    jax_init()
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    js <- reticulate::import("jax.scipy.stats", convert = FALSE)
    np <- reticulate::import("numpy")
    fn <- if (isTRUE(f$log)) js$norm$logpdf else js$norm$pdf
    as.double(np$asarray(fn(jnp$asarray(x, dtype = jax_dtype(dtype)), p$mean, p$sd)))
  },
  jax_grad = function(x, dtype, p, f) {
    jax_init()
    reticulate::py_run_string(
      "
import jax
from jax.scipy.stats import norm as _n
def _dnorm_grad(log, i):
    f = _n.logpdf if log else _n.pdf
    return jax.jit(jax.vmap(jax.grad(f, argnums=i), in_axes=(0, None, None)))
"
    )
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    np <- reticulate::import("numpy")
    xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
    g <- function(i) as.double(np$asarray(reticulate::py$`_dnorm_grad`(f$log, i)(xx, p$mean, p$sd)))
    list(x = g(0L), mean = g(1L), sd = g(2L))
  }
)
