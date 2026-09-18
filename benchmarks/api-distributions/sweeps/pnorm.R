## ---------------------------------------------------------------------------
## nv_pnorm against base R pnorm().
##
## nv_pnorm carries two symmetric tail branches -- an asymptotic Mills-ratio
## series in the far lower tail and its mirror image near 1 -- and the sweep
## crosses both seams from both sides. That symmetry is the point: every
## accuracy grid this function was originally developed against was one-sided
## (deeply negative z only), which left the upper-tail branch unexercised in
## the artifact even though unit tests happened to cover it. A bit-pattern
## sweep has no sidedness to get wrong.
## ---------------------------------------------------------------------------

source(file.path(here(), "sweeps", "_normal.R"), local = TRUE)

## With z = (q - mean)/sd and phi the standard density, the lower-tail CDF has
##
##   d/dq  phi(z)/sd     d/dmean  -phi(z)/sd     d/dsd  -z phi(z)/sd
##
## and the upper tail flips every sign. On the log scale each is divided by the
## probability itself; that ratio is formed in log space -- exp(log phi - log P)
## rather than phi/P -- because in the far tail both underflow while their
## ratio stays perfectly ordinary, which is the whole reason the log variants
## exist.
pnorm_grad_ref <- function(q, p, f) {
  z <- (q - p$mean) / p$sd
  lower <- isTRUE(f$lower_tail)
  s <- if (lower) 1 else -1
  ## On the log scale the ratio is the inverse Mills ratio, taken at z for the
  ## lower tail and at -z for the upper by symmetry. See inv_mills() in
  ## _normal.R for why it is not exp(log phi - log Phi).
  r <- if (isTRUE(f$log_p)) {
    inv_mills(if (lower) z else -z) / p$sd
  } else {
    dnorm(z) / p$sd
  }
  list(q = s * r, mean = -s * r, sd = -s * z * r)
}

grad_pnorm <- anvl::jit(
  anvl::gradient(
    \(q, mean, sd, lower_tail = TRUE, log_p = FALSE) {
      sum(anvl::nv_pnorm(q, mean, sd, lower_tail = lower_tail, log_p = log_p))
    },
    wrt = c("q", "mean", "sd")
  ),
  static = c("lower_tail", "log_p")
)

sweep_spec(
  name = "nv_pnorm",
  family = "normal",
  blurb = "normal CDF against base R pnorm()",
  primary = "q",
  params = NORM_PARAMS,
  flags = list(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE)),
  support = function(p, f) c(-Inf, Inf),
  value = function(x, dtype, p, f) {
    as.double(anvl::nv_pnorm(
      anvl::nv_array(x, dtype = dtype),
      p$mean,
      p$sd,
      lower_tail = f$lower_tail,
      log_p = f$log_p
    ))
  },
  ref_value = function(x, p, f) {
    pnorm(x, mean = p$mean, sd = p$sd, lower.tail = f$lower_tail, log.p = f$log_p)
  },

  grad_wrt = c("q", "mean", "sd"),
  grad = function(x, dtype, p, f) {
    n <- length(x)
    d <- grad_pnorm(
      anvl::nv_array(x, dtype = dtype),
      anvl::nv_array(rep(p$mean, n), dtype = dtype),
      anvl::nv_array(rep(p$sd, n), dtype = dtype),
      lower_tail = f$lower_tail,
      log_p = f$log_p
    )
    lapply(list(q = d$q, mean = d$mean, sd = d$sd), as.double)
  },
  ref_grad = pnorm_grad_ref,

  ## jax.scipy.stats.norm has cdf, logcdf, sf and logsf, so all four variants
  ## have a twin -- the only distribution here where that is true.
  jax_value = function(x, dtype, p, f) {
    jax_init()
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    js <- reticulate::import("jax.scipy.stats", convert = FALSE)
    np <- reticulate::import("numpy")
    fn <- if (isTRUE(f$lower_tail)) {
      if (isTRUE(f$log_p)) js$norm$logcdf else js$norm$cdf
    } else {
      if (isTRUE(f$log_p)) js$norm$logsf else js$norm$sf
    }
    as.double(np$asarray(fn(jnp$asarray(x, dtype = jax_dtype(dtype)), p$mean, p$sd)))
  },
  jax_grad = function(x, dtype, p, f) {
    jax_init()
    reticulate::py_run_string(
      "
import jax
from jax.scipy.stats import norm as _n
def _pnorm_grad(lower, logp, i):
    f = (_n.logcdf if logp else _n.cdf) if lower else (_n.logsf if logp else _n.sf)
    return jax.jit(jax.vmap(jax.grad(f, argnums=i), in_axes=(0, None, None)))
"
    )
    jnp <- reticulate::import("jax.numpy", convert = FALSE)
    np <- reticulate::import("numpy")
    xx <- jnp$asarray(x, dtype = jax_dtype(dtype))
    g <- function(i) {
      as.double(np$asarray(
        reticulate::py$`_pnorm_grad`(f$lower_tail, f$log_p, i)(xx, p$mean, p$sd)
      ))
    }
    list(q = g(0L), mean = g(1L), sd = g(2L))
  }
)
