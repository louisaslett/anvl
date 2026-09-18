## ---------------------------------------------------------------------------
## Shared by the three normal specs. Not a spec itself (leading underscore).
##
##   standard  (mean, sd) = (0, 1), the default. Every affine step is exact, so
##             it isolates the distribution's own mathematics -- and says
##             nothing about the standardisation.
##   shifted   (mean, sd) = (-pi, 2*pi). Neither constant is representable in
##             either precision, so (x - mean)/sd is exercised the way a real
##             caller exercises it. This is also the route that the old
##             literal-precision bug destroyed: `mean` and `sd` arrive as jit
##             arguments, which no anvl-side patch could reach. These cells are
##             what hold that fix in place.
## ---------------------------------------------------------------------------

NORM_PARAMS <- list(
  standard = list(mean = 0, sd = 1),
  shifted = list(mean = -pi, sd = 2 * pi)
)

jax_init <- local({
  done <- FALSE
  function() {
    if (done) {
      return(invisible(TRUE))
    }
    library(reticulate)
    venv <- normalizePath(file.path(here(), "..", "..", "..", "py-benchmarks", ".venv"), mustWork = FALSE)
    if (!nzchar(Sys.getenv("RETICULATE_PYTHON")) && dir.exists(venv)) {
      use_virtualenv(venv, required = TRUE)
    }
    import("jax")$config$update("jax_enable_x64", TRUE)
    done <<- TRUE
    invisible(TRUE)
  }
})

jax_dtype <- function(dtype) {
  jnp <- reticulate::import("jax.numpy", convert = FALSE)
  if (dtype == "f32") jnp$float32 else jnp$float64
}

## The inverse Mills ratio, phi(z) / Phi(z) -- the derivative of log Phi, and
## the quantity every log-scale normal gradient reduces to.
##
## The obvious reference, exp(dnorm(z, log = TRUE) - pnorm(z, log.p = TRUE)),
## is wrong in the far tail and wrong silently. Both logs there are about
## -z^2/2; at z = -1e150 that is -5e299, whose ulp is ~1e284, while the
## difference being sought is only log(-z) ~ 345. The subtraction loses every
## digit and returns exp(0) = 1 where the true value is 1e150, and past
## z = -1e154 both logs overflow to -Inf and it returns NaN outright.
##
## This was not a hypothetical. The first version of these specs used that
## formula, and the sweep reported nv_pnorm's log-scale gradients as wrong by a
## factor of 1e231 across the whole deep tail. anvl was right and the reference
## was wrong -- which is the failure mode a reference implementation must be
## built to avoid, because it is indistinguishable from a real finding until
## someone checks.
##
## So: the asymptotic expansion where the logs are too large to subtract, and
## the log-space ratio everywhere else. At the z = -100 crossover the series
## has converged to f64 (the first dropped term is ~1e-17) and the logs are
## only ~5e3, so both routes are accurate and they agree.
inv_mills <- function(z) {
  out <- numeric(length(z))
  far <- !is.na(z) & z < -100
  if (any(far)) {
    u <- 1 / z[far]^2
    ## Phi(z) = phi(z)/(-z) * (1 - u + 3u^2 - 15u^3 + 105u^4 - 945u^5), so the
    ## ratio is (-z) divided by that bracket.
    out[far] <- -z[far] / (1 - u * (1 - u * (3 - u * (15 - u * (105 - 945 * u)))))
  }
  near <- !far & !is.na(z)
  if (any(near)) {
    out[near] <- exp(dnorm(z[near], log = TRUE) - pnorm(z[near], log.p = TRUE))
  }
  out[is.na(z)] <- z[is.na(z)] # keep NaN as NaN
  out
}
