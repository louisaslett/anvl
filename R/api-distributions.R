#' @include jit.R
NULL

## Probability distributions ---------------------------------------------------

# The operand -- `x`/`q`/`p`, always the first argument -- governs the data type
# the distribution is evaluated at; the parameters follow it. The float check
# runs before the promotion, so a non-float operand is reported as itself rather
# than as a failure to bring a parameter to its data type.
promote_distribution_args <- function(...) {
  args <- list(...)
  operand <- names(args)[[1L]]
  assert_float_dtype(
    peek_dtype(args[[1L]]),
    arg = operand,
    hint = "Convert it with `nv_convert()`."
  )
  do.call(as_anvl_arrays, c(args, list(.promote = promotion_like(operand))))
}

#' @title The Normal Distribution
#' @name nv_normal
#' @description
#' Density (`nv_dnorm`), distribution function (`nv_pnorm`),
#' quantile function (`nv_qnorm`), and random
#' generation (`nv_rnorm`) for the Normal distribution with mean `mean` and
#' standard deviation `sd`.
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`).
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`.
#' @param mean ([`arrayish`])\cr
#'   Mean of the distribution. Either a scalar, or an array of exactly the
#'   shape of `x`/`q`/`p` (or the sample, for `nv_rnorm`).
#' @param sd ([`arrayish`])\cr
#'   Standard deviation of the distribution, shaped like `mean`. Must be
#'   positive, otherwise results are invalid.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the densities/probabilities are given as logarithms. For
#'   `nv_qnorm` this describes the input `p`.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le x)}; otherwise,
#'   \eqn{P(X > x)}.
#' @details
#' The Normal distribution has probability density function:
#' \deqn{f(x) = \frac{1}{\sigma\sqrt{2\pi}}
#'   \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}
#' where \eqn{\mu} is the mean and \eqn{\sigma} is the standard deviation.
#'
#' `nv_pnorm` uses the asymptotic expansion from
#' `r cite_bib("abramowitz1964handbook")`, equation 26.2.12, in the
#' left tail when `log_p = TRUE` to maintain accuracy.
#'
#' `nv_qnorm` uses the same minimax rational approximation as
#' `r cite_bib("moshier1989methods")` (this is `ndtri` in the Cephes
#' library as used by JAX) for `f64`, and uses a new lower degree Remez minimax
#' rational approximation on the same intervals for `f32`.
#'
#' @templateVar dist norm
#' @templateVar params `mean` or `sd`
#' @template section_distribution_dtype
#' @references
#' `r format_bib("abramowitz1964handbook", "moshier1989methods")`
#' @seealso [nv_rnorm()] for sampling from a normal distribution.
#' @return ([`arrayish`] | named `list` of two [`arrayish`])\cr
#' `nv_dnorm()`, `nv_pnorm()` and `nv_qnorm()` return an [`arrayish`] with the
#' shape and data type of `x`/`q`/`p`.
#'
#' `nv_rnorm()` returns a named `list` of two [`arrayish`]: `state`, the updated
#' RNG state with the input `state`'s data type and shape, and `values`, the
#' sample of shape `shape` and the data type described under `dtype`.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-1, 0, 1))
#' nv_dnorm(x)
#' nv_dnorm(x, mean = 1, sd = 2)
#' nv_dnorm(x, log = TRUE)
#'
#' nv_pnorm(x)
#' nv_pnorm(x, mean = 1, sd = 2)
#' nv_pnorm(x, lower_tail = FALSE)
#' nv_pnorm(x, log_p = TRUE)
#'
#' p <- nv_array(c(0.025, 0.5, 0.975))
#' nv_qnorm(p)
#' nv_qnorm(p, mean = 1, sd = 2)
#' nv_qnorm(p, lower_tail = FALSE)
#' nv_qnorm(nv_array(c(-700, -2, -0.1), dtype = "f64"), log_p = TRUE)
NULL

#' @rdname nv_normal
#' @export
nv_dnorm <- jit(
  function(x, mean = 0, sd = 1, log = FALSE) {
    assert_flag(log)
    args <- promote_distribution_args(x = x, mean = mean, sd = sd)
    x <- args$x
    mean <- args$mean
    sd <- args$sd

    z <- (x - mean) / sd
    log_density <- -0.5 * (z * z) - nv_log(sd) - 0.5 * base::log(2 * pi)

    if (log) {
      return(log_density)
    }
    nv_exp(log_density)
  },
  static = "log"
)

#' @rdname nv_normal
#' @export
nv_pnorm <- jit(
  function(q, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(q = q, mean = mean, sd = sd)
    q <- args$q
    mean <- args$mean
    sd <- args$sd
    # One threshold set per width, so a narrower float has none: it would
    # silently take the `f64` set.
    op_dtype <- assert_rng_float_dtype(dtype(q), arg = "q")

    # Standardise, flipping sign if computing upper tail
    d <- if (lower_tail) (q - mean) / sd else (mean - q) / sd

    if (!log_p) {
      # When not computing log cdf we're done as no accuracy concerns with erfc
      return(0.5 * nv_erfc(-d / sqrt(2)))
    }

    # Here computing log cdf: care required to ensure accuracy deep in the tails,
    # since there is no log version of erfc in XLA.
    # R handles this with a near-minimax approximation due to Cody
    # <doi:10.1090/S0025-5718-1969-0247736-4>, but this algorithm does not perform
    # well with XLA due to complicated rational-polynomial expression.
    # Instead use the classic successive integration by parts asymptotic expansion
    # from Abramowitz & Stegun, eq 26.2.12 p.932 <isbn:0-486-61272-4> (originally
    # due to Laplace? Also used by JAX) if the argument is in a region where
    # direct evaluation of log(erfc) would be inaccurate.

    # Thresholds between direct computation of erfc and the asymptotic expansion,
    # Q, for f32 and f64. These differ from JAX for accuracy.
    is_f32 <- op_dtype == "f32"
    lower_threshold <- if (is_f32) -11.9 else -20
    upper_threshold <- 0

    # Computation regime:
    #   d <= lower_threshold ... then we compute log Q(-d) using asymptotic
    #                            expansion
    #   d > upper_threshold  ... then we compute log(1-erfc(d/sqrt(2))). Note the
    #                            approximation -erfc(d/sqrt(2)) has catastrophic
    #                            loss of accuracy
    #   d in between         ... accuracy of log(erfc(-d/sqrt(2))) is fine

    # Compute Q(-d) for the asymptotic region, first clamping the value to protect
    # gradient from poisoning later
    d_asymp <- nv_pmax(-d, 1)
    d2_asymp <- d_asymp * d_asymp
    w <- 1 / d2_asymp
    # Compute just what is required for precision (confirmed if statement compiles
    # away during tracing)
    series_minus_1 <- if (is_f32) {
      w * (-1 + w * 3)
    } else {
      w * (-1 + w * (3 + w * (-15 + w * (105 + w * (-945 + w * (10395 + w * (-135135)))))))
    }
    log_pdf_term <- -0.5 * d2_asymp - 0.5 * base::log(2 * pi)

    # Check which regime (asymptotic, direct, upper tail)
    use_non_asymp <- d > lower_threshold
    use_direct <- use_non_asymp & d <= upper_threshold
    # Compute correct erfc(-d/sqrt(2)) or erfc(d/sqrt(2)), selecting on arg to
    # avoid multiple erfc evaluations
    erfc_arg <- nv_ifelse(use_direct, -d, d)
    erfc_res <- 0.5 * nv_erfc(erfc_arg / sqrt(2))
    # Clamp result to a safe value on other branches so gradient not poisoned on
    # log/log1p calls
    erfc_res_direct <- nv_ifelse(use_direct, erfc_res, 1)
    erfc_res_upper <- nv_ifelse(use_non_asymp, erfc_res, 0)
    # Compute final answer down all branches, returning correct branch for each
    # element
    nv_ifelse(
      use_direct,
      nv_log(erfc_res_direct),
      nv_ifelse(
        use_non_asymp,
        nv_log1p(-erfc_res_upper),
        log_pdf_term -
          nv_log(d_asymp) +
          if (is_f32) series_minus_1 else nv_log1p(series_minus_1)
      )
    )
  },
  static = c("lower_tail", "log_p")
)

# Horner's method for polynomials, coefficients in decreasing power order.
# x can be vector, say length n.
# coefs can be:
#   - a vector length d for a single polynomial; or
#   - a list of d vectors, each length n, for a different polynomial for each x.
#     Note layout is by power, so `coef[[1L]]` holds all n highest power coefs,
#     `coef[[2L]]` holds all n second highest power coefs etc.
#     Hence *only* suitable if all polynomials of the same degree.
horner <- function(x, coefs) {
  Reduce(function(acc, coef) acc * x + coef, coefs[-1L], init = coefs[[1L]])
}

# P/Q rational polynomial coefficients (P = numerator, Q = denominator), highest
# power first.
# - central region covers p \in (e^{-2}, 1-e{-2}] and is poly in w^2 where
#           w = p - 1/2;
# - tail region covers all other p. Separates into `tail` for z < 8 and
#        `far_tail` for z >= 8. Both are poly in 1/z where z = sqrt(-2 log t)
#        and t = min(p, 1 - p)
#
# NOTE: efficient use of Map in `select_far()` inside `nv_qnorm` assumes that
#       `p_tail` and `p_far_tail`, as well as `q_tail` and `q_far_tail` are the
#       same length, so any future Remez refit must ensure this or change
#       `select_far()` (applies to f32 and f64)
#
# First f64 precision: these are the coefficients from Cephes, as used also by
# JAX
qnorm_f64_coefs <- list(
  p_central = c(
    -5.99633501014107895267e1,
    9.80010754185999661536e1,
    -5.66762857469070293439e1,
    1.39312609387279679503e1,
    -1.23916583867381258016
  ),
  q_central = c(
    1.0,
    1.95448858338141759834,
    4.67627912898881538453,
    8.63602421390890590575e1,
    -2.25462687854119370527e2,
    2.00260212380060660359e2,
    -8.20372256168333339912e1,
    1.59056225126211695515e1,
    -1.18331621121330003142
  ),
  p_tail = c(
    4.05544892305962419923,
    3.15251094599893866154e1,
    5.71628192246421288162e1,
    4.40805073893200834700e1,
    1.46849561928858024014e1,
    2.18663306850790267539,
    -1.40256079171354495875e-1,
    -3.50424626827848203418e-2,
    -8.57456785154685413611e-4
  ),
  q_tail = c(
    1.0,
    1.57799883256466749731e1,
    4.53907635128879210584e1,
    4.13172038254672030440e1,
    1.50425385692907503408e1,
    2.50464946208309415979,
    -1.42182922854787788574e-1,
    -3.80806407691578277194e-2,
    -9.33259480895457427372e-4
  ),
  p_far_tail = c(
    3.23774891776946035970,
    6.91522889068984211695,
    3.93881025292474443415,
    1.33303460815807542389,
    2.01485389549179081538e-1,
    1.23716634817820021358e-2,
    3.01581553508235416007e-4,
    2.65806974686737550832e-6,
    6.23974539184983293730e-9
  ),
  q_far_tail = c(
    1.0,
    6.02427039364742014255,
    3.67983563856160859403,
    1.37702099489081330271,
    2.16236993594496635890e-1,
    1.34204006088543189037e-2,
    3.28014464682127739104e-4,
    2.89247864745380683936e-6,
    6.79019408009981274425e-9
  )
)

# Then we specialise to f32: the above polynomials are overkill at f32 so below
# is an independent Remez fit for anvl using the same thresholds between
# central/tail/far tail and the same poly argument (w^2 or 1/z)
qnorm_f32_coefs <- list(
  p_central = c(-6.691131842723991e-1, 7.5626636219604695, -5.770283790138877, 1.047197585894062),
  q_central = c(-1.257612301180524e1, 1.820651998768941e1, -7.70932529281657, 1.0),
  p_tail = c(-1.1703880518959358, 9.77404924657488, 2.8949524675071373e1, 9.415665982832321, 9.171604050864e-1),
  q_tail = c(2.662973999005499e1, 1.0071629918518465e1, 1.0),
  p_far_tail = c(
    -1.3985698840384828e2,
    4.453781244880416e2,
    8.742497845723311e2,
    8.258251477108792e1,
    9.189365211474885e-1
  ),
  q_far_tail = c(9.349271395441176e2, 8.984706461134404e1, 1.0)
)

#' @rdname nv_normal
#' @export
nv_qnorm <- jit(
  function(p, mean = 0, sd = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(p = p, mean = mean, sd = sd)
    p <- args$p
    mean <- args$mean
    sd <- args$sd
    # One coefficient set per width -- see `nv_pnorm()`.
    op_dtype <- assert_rng_float_dtype(dtype(p), arg = "p")

    is_f32 <- op_dtype == "f32"

    cf <- if (is_f32) qnorm_f32_coefs else qnorm_f64_coefs
    lp <- if (log_p) p else nv_log(p)

    # As described above for rational polynomial coefficients, we divide into
    # regions.
    # upper tail if p > 1-e^-2         poly in 1/z where z = sqrt(-2 log p)
    # central    if e^-2 < p <= 1-e^-2 poly in w^2, w = p-0.5
    # lower tail if p <= e^-2          poly in 1/z where z = sqrt(-2 log (1-p))
    # Will actually handle lower tail by folding into upper via z = sqrt(-2 log t)
    # for t = min(p, 1-p).
    # The tail approximation is split between a near (z < 8) and far (z >= 8).

    # First, identify flags for upper and central region.
    # Then,
    #          use_upper  use_central
    # upper     TRUE       FALSE
    # central   FALSE      TRUE
    # lower     FALSE      FALSE
    if (log_p) {
      upper_threshold <- base::log1p(-exp(-2))
      use_upper <- lp > upper_threshold
      use_central <- (lp > -2) & (lp <= upper_threshold)
    } else {
      upper_threshold <- 1 - exp(-2)
      use_upper <- p > upper_threshold
      use_central <- (p > exp(-2)) & (p <= upper_threshold)
    }

    # Tail approximation
    # Compute log(1-p), with guards for derivatives ...
    log_t <- if (log_p) {
      nv_log(-nv_expm1(nv_ifelse(use_upper, lp, -1)))
    } else {
      nv_log1p(-nv_ifelse(use_upper, p, 0))
    }
    # ... and then log t = min(log p, log(1-p)) by selection
    log_t <- nv_ifelse(use_upper, log_t, lp)
    is_boundary <- log_t == -Inf
    # Safely clamp central region and boundary elements onto the branch boundary,
    # where tail is well behaved for gradients
    log_t <- nv_ifelse(is_boundary | use_central, -2, log_t)

    # Compute the near or far tail rational polynomial approximation
    # Poly is in 1/z for z = sqrt(-2 log t)
    z <- nv_sqrt(-2 * log_t)
    inv_z <- 1 / z
    use_far_tail <- z >= 8
    # See important "NOTE" preceding coefficients above regarding this helper func
    select_far <- function(far, near) {
      # The coefficients are plain R numbers, so a bare `nv_ifelse(pred, x, y)`
      # would have nothing to yield to and materialize at the default float,
      # dragging the whole result up with it. They are built at `p`'s data type
      # instead.
      Map(
        function(x, y) nv_ifelse(use_far_tail, nv_scalar_like(p, x), nv_scalar_like(p, y)),
        far,
        near
      )
    }
    ratio <- horner(inv_z, select_far(cf$p_far_tail, cf$p_tail)) /
      horner(inv_z, select_far(cf$q_far_tail, cf$q_tail))
    res_tail <- z - nv_log(z) * inv_z - ratio * inv_z

    # Central approximation
    # Poly is in w^2 for w = p-0.5 (accounting for if arg was log_p)
    w <- if (log_p) 0.5 * nv_expm1(lp + base::log(2)) else p - 0.5
    w2 <- w * w
    res_central <- base::sqrt(2 * pi) *
      (w + w * w2 * (horner(w2, cf$p_central) / horner(w2, cf$q_central)))

    # Final standardised Normal result
    # Distinguish central region from a tail, then resolve left/right tail
    res_std <- nv_ifelse(
      use_central,
      res_central,
      nv_ifelse(use_upper, res_tail, -res_tail)
    )
    # The infinities are built at `p`'s data type: two bare R doubles here would
    # have nothing to yield to, materialize at the default float, and drag the
    # result up with them.
    res_std <- nv_ifelse(
      is_boundary,
      nv_ifelse(use_upper, nv_scalar_like(p, Inf), nv_scalar_like(p, -Inf)),
      res_std
    )
    # Handle tail switch
    if (!lower_tail) {
      res_std <- -res_std
    }
    # Unstandardise as necessary
    mean + sd * res_std
  },
  static = c("lower_tail", "log_p")
)

#' @title The Uniform Distribution
#' @name nv_uniform
#' @description
#' Density (`nv_dunif`), distribution function (`nv_punif`), quantile
#' function (`nv_qunif`), and random generation (`nv_runif`) for the Uniform
#' distribution on the interval from `min` to `max`.
#' @param x,q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the density (`x`) or the distribution
#'   function (`q`).
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`.
#' @param min,max ([`arrayish`])\cr
#'   Lower and upper limits of the distribution. Either scalars, or arrays of
#'   exactly the same shape as `x`/`q`/`p` (or the sample, for `nv_runif`), in
#'   which case the interval varies elementwise and each element of `x`/`q`/`p`
#'   is evaluated against, or each draw made from, its own `min`/`max`.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the densities/probabilities are given as logarithms. For
#'   `nv_qunif` this describes the input `p`.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le x)}; otherwise,
#'   \eqn{P(X > x)}.
#' @details
#' The Uniform distribution has probability density function:
#' \deqn{f(x) = \frac{1}{b - a}, \quad a \le x \le b}
#' and zero elsewhere, where \eqn{a} is `min` and \eqn{b} is `max`.
#'
#' All four are univariate functions evaluated elementwise, returning one
#' value per element of `x`/`q`/`p` (or of the sample). Non-scalar `min`/`max` therefore give a
#' separate univariate Uniform per element, *not* a multivariate Uniform over
#' the hyper-rectangle \eqn{\prod_i [a_i, b_i]}. For that, reduce over the
#' result: `nv_prod(nv_dunif(x, min, max))`, or
#' `nv_sum(nv_dunif(x, min, max, log = TRUE))` on the log scale.
#'
#' @templateVar dist unif
#' @templateVar params `min` or `max`
#' @template section_distribution_dtype
#' @return ([`arrayish`] | named `list` of two [`arrayish`])\cr
#' `nv_dunif()`, `nv_punif()`, and `nv_qunif()` return an [`arrayish`] with the
#' same shape and data type as `x`/`q`/`p`.
#'
#' `nv_runif()` returns a named `list` of two [`arrayish`]: `state`, the updated
#' RNG state with the input `state`'s data type and shape, and `values`, the
#' sample of shape `shape` and the data type described under `dtype`.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(-0.5, 0, 0.25, 1, 1.5))
#' nv_dunif(x)
#' nv_dunif(x, min = -1, max = 2)
#' nv_dunif(x, log = TRUE)
#'
#' # `min`/`max` may vary elementwise, giving one univariate Uniform per
#' # element rather than a single distribution over a hyper-rectangle
#' lower <- nv_array(c(-1, -1, 0, 0, 1))
#' upper <- nv_array(c(0, 1, 1, 2, 2))
#' nv_dunif(x, min = lower, max = upper)
#'
#' nv_punif(x)
#' nv_punif(x, min = -1, max = 2)
#' nv_punif(x, lower_tail = FALSE)
#' nv_punif(x, log_p = TRUE)
#'
#' p <- nv_array(c(0.025, 0.5, 0.975))
#' nv_qunif(p)
#' nv_qunif(p, min = -1, max = 2)
#' nv_qunif(p, lower_tail = FALSE)
#' nv_qunif(nv_array(c(-700, -2, -0.1), dtype = "f64"), log_p = TRUE)
#'
#' # `state` is the updated RNG state, `values` the sample
#' state <- nv_rng_state(42L)
#' result <- nv_runif(c(2, 3), state)
#' result$values
#'
#' # `min`/`max` may also be arrays of the same shape as the sample
#' lower <- nv_array(matrix(c(0, 10, 20, 30, 40, 50), nrow = 2))
#' nv_runif(c(2, 3), state, min = lower, max = lower + 1)$values
NULL

#' @rdname nv_uniform
#' @export
nv_dunif <- jit(
  function(x, min = 0, max = 1, log = FALSE) {
    assert_flag(log)
    args <- promote_distribution_args(x = x, min = min, max = max)
    x <- args$x
    min <- args$min
    max <- args$max

    # Density constant on support, just need support indicator
    in_support <- (x >= min) & (x <= max)
    width <- max - min

    density <- if (log) {
      nv_ifelse(in_support, -nv_log(width), -Inf)
    } else {
      nv_ifelse(in_support, 1 / width, 0)
    }
    # NOTE: `in_support` will eval to FALSE when x is NaN, so need to restore a
    #       NaN result there. Similarly, the `max > min` check ensures NaN is
    #       restored for same reason if either is NaN while also rejecting
    #       reversed interval ends
    nv_ifelse(!nv_is_nan(x) & (max > min), density, NaN)
  },
  static = "log"
)

#' @rdname nv_uniform
#' @export
nv_punif <- jit(
  function(q, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(q = q, min = min, max = max)
    q <- args$q
    min <- args$min
    max <- args$max

    width <- max - min
    # Resolve q against the endpoints before dividing, to match base R behaviour.
    # Also avoids degenerate 0 / 0 for edge case q == min == max.
    at_or_above <- q >= max
    at_or_below <- q <= min
    resolve_ends <- function(above_val, below_val, interior_val) {
      nv_ifelse(at_or_above, above_val, nv_ifelse(at_or_below, below_val, interior_val))
    }

    # Ensure all branches have safe value for gradients
    q_int <- nv_ifelse(at_or_above | at_or_below, min, q)

    u <- if (lower_tail) {
      resolve_ends(1, 0, (q_int - min) / width)
    } else {
      resolve_ends(0, 1, (max - q_int) / width)
    }

    # Reversed/non-finite interval is NaN to match base R: `valid` flag to track
    valid <- nv_is_finite(min) & nv_is_finite(max) & (max >= min)

    if (!log_p) {
      return(nv_ifelse(valid, u, NaN))
    }

    # To maintain accuracy of log near 1, switch to log1p in opposite tail mid way
    v <- if (lower_tail) {
      resolve_ends(0, 1, (max - q_int) / width)
    } else {
      resolve_ends(1, 0, (q_int - min) / width)
    }
    # So flag if can use log, else switch to log1p() of the opposite tail
    use_log <- u <= 0.5
    # Include inner clamp of a safe input on branch not taken for gradient calcs
    res <- nv_ifelse(
      use_log,
      nv_log(nv_ifelse(use_log, u, 1)),
      nv_log1p(-nv_ifelse(use_log, 0, v))
    )
    nv_ifelse(valid, res, NaN)
  },
  static = c("lower_tail", "log_p")
)

#' @rdname nv_uniform
#' @export
nv_qunif <- jit(
  function(p, min = 0, max = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(p = p, min = min, max = max)
    p <- args$p
    min <- args$min
    max <- args$max

    # Out-of-range `p` is resolved to NaN by `valid`, but also need `p_safe` to
    # avoid poisoning gradients
    if (log_p) {
      in_range <- p <= 0
      p_safe <- nv_ifelse(in_range, p, 0)
      u <- if (lower_tail) nv_exp(p_safe) else -nv_expm1(p_safe)
    } else {
      in_range <- (p >= 0) & (p <= 1)
      p_safe <- nv_ifelse(in_range, p, 0)
      u <- if (lower_tail) p_safe else 1 - p_safe
    }

    # Conditions to match NaN behaviour of base R
    valid <- in_range & nv_is_finite(min) & nv_is_finite(max) & (max >= min)
    nv_ifelse(valid, min + u * (max - min), NaN)
  },
  static = c("lower_tail", "log_p")
)

## Poisson distribution helpers -------------------------------------------------

# `x` with no gradient flowing back through it. anvl has no stop-gradient
# primitive, but the reverse rule of a bitcast is zero, and a round trip through
# the integer data type of the same width leaves the bits unchanged.
stop_gradient <- function(x) {
  float_dtype <- dtype(x)
  int_dtype <- if (float_dtype == "f32") "i32" else "i64"
  nv_bitcast_convert(nv_bitcast_convert(x, int_dtype), float_dtype)
}

# log(1 - exp(x)) for x <= 0, switching at -log(2) between the two forms that
# keep full accuracy either side of it (Maechler, "Accurately Computing
# log(1 - exp(-|a|))"; R's `R_Log1_Exp()`). Each form is fed a safe stand-in on
# the elements where it is not selected, so it cannot poison the gradient.
#
# The CPU plugin's f64 `log1p()` loses up to ~120 ulp on [1 - sqrt(2), -0.36],
# where log(1 + y) is accurate (1 + y is nearly exact there), so log1p(-exp(x))
# is taken as log(1 - exp(x)) for exp(x) in (1/3, 1/2]. XLA applied its rational
# approximation for |y| < sqrt(2) - 1 rather than 1/sqrt(2) <= 1 + y <= sqrt(2);
# fixed in openxla/xla#46765, which the CPU plugin predates. Drop the third
# branch once the plugin is built from XLA at or after that fix.
log1mexp <- function(x) {
  use_expm1 <- x > -base::log(2)
  e <- nv_exp(nv_ifelse(use_expm1, -1, x))
  nv_ifelse(
    use_expm1,
    nv_log(-nv_expm1(nv_ifelse(use_expm1, x, -1))),
    nv_ifelse(e > 1 / 3, nv_log(1 - e), nv_log1p(-e))
  )
}

# Error of Stirling's approximation to log(n!),
#   stirlerr(n) = log(n!) - log(sqrt(2 * pi * n) * (n / e)^n),
# for whole n >= 1, as R's `stirlerr()`: exact values up to n = 15, and past it
# the asymptotic series 1/(12 n) - 1/(360 n^3) + ..., truncated where it is
# accurate to working precision from n = 16. The series coefficients are in
# powers of 1/n^2, highest first.
stirlerr_table <- c(
  0.0810614667953272582196702,
  0.0413406959554092940938221,
  0.02767792568499833914878929,
  0.02079067210376509311152277,
  0.01664469118982119216319487,
  0.01387612882307074799874573,
  0.01189670994589177009505572,
  0.010411265261972096497478567,
  0.009255462182712732917728637,
  0.008330563433362871256469318,
  0.007573675487951840794972024,
  0.006942840107209529865664152,
  0.006408994188004207068439631,
  0.005951370112758847735624416,
  0.005554733551962801371038690
)
stirlerr_f64_coefs <- c(1 / 156, -691 / 360360, 1 / 1188, -1 / 1680, 1 / 1260, -1 / 360, 1 / 12)
stirlerr_f32_coefs <- c(-1 / 1680, 1 / 1260, -1 / 360, 1 / 12)

stirlerr <- function(n, is_f32) {
  # Gather only takes a vector of indices, so flatten and restore the shape
  table <- nv_array(stirlerr_table, dtype = dtype(n))
  idx <- nv_reshape(nv_convert(nv_clamp(n, 1, 15), "i32"), -1L)
  tabled <- nv_reshape(table[idx], shape(n))
  n_series <- nv_pmax(n, 16)
  series <- horner(1 / (n_series * n_series), if (is_f32) stirlerr_f32_coefs else stirlerr_f64_coefs) /
    n_series
  nv_ifelse(n <= 15, tabled, series)
}

# log(x / y) for x >= 1 and y > 0, kept accurate where the ratio over- or
# underflows by rescaling y by a power of two first. Rescaling y rather than x
# matters: XLA divides by multiplying with the reciprocal, and the reciprocal of
# a large y would itself flush to zero.
log_ratio <- function(x, y, is_f32) {
  k <- if (is_f32) 126 else 1022
  r <- x / y
  over <- r == Inf
  under <- r < 2^-k
  y_scaled <- nv_ifelse(over, y * 2^k, nv_ifelse(under, y * 2^-k, y))
  # Built at `x`'s data type: with only R values to choose between, the shift
  # would materialize at the default float
  shift <- nv_scalar_like(x, k * base::log(2))
  nv_log(x / y_scaled) + nv_ifelse(over, shift, nv_ifelse(under, -shift, 0))
}

# Deviance term bd0(x, np) = x log(x / np) + np - x of Loader's saddle point
# expansion, for x >= 1 and np > 0 (Loader, 2000; R's `bd0()`). Near x = np the
# direct form cancels catastrophically, so it is replaced there by the series
#   (x - np) v + 2 x v (v^2 / 3 + v^4 / 5 + ...),  v = (x - np) / (x + np),
# truncated where it is accurate to working precision for |v| < 0.1. The
# coefficients 1 / (2j + 3) are highest power of v^2 first.
bd0_f64_coefs <- 1 / c(17, 15, 13, 11, 9, 7, 5, 3)
bd0_f32_coefs <- 1 / c(9, 7, 5, 3)

bd0 <- function(x, np, is_f32) {
  d <- x - np
  # v and the test for |v| < 0.1 are scale-free, so near the top of the range
  # x and np are scaled down first: x + np could overflow, and the reciprocal
  # XLA divides by could be subnormal. The scaling is selected rather than
  # factored out, which XLA would undo by rewriting c x + c np as c (x + np).
  huge <- nv_pmax(x, np) > (if (is_f32) 2^120 else 2^1016)
  xs <- nv_ifelse(huge, x * 2^-4, x)
  nps <- nv_ifelse(huge, np * 2^-4, np)
  ds <- xs - nps
  ss <- xs + nps
  near <- nv_abs(ds) < 0.1 * ss
  v <- nv_ifelse(near, ds / ss, 0)
  w <- v * v
  series <- d * v + x * v * w * (2 * horner(w, if (is_f32) bd0_f32_coefs else bd0_f64_coefs))
  # np - x first: near the top of the range x log(x / np) + np can overflow
  # where the whole cannot
  direct <- x * log_ratio(x, np, is_f32) + (np - x)
  nv_ifelse(near, series, direct)
}

# Log of the Poisson probability of a whole x >= 0 at a finite lambda > 0, with
# lambda = 0 admitted at x = 0. By Loader's saddle point expansion
#   log p(x) = -stirlerr(x) - bd0(x, lambda) - log(2 pi x) / 2
# for x >= 1, and exactly -lambda at x = 0.
dpois_log_raw <- function(x, lambda, is_f32) {
  at_zero <- x == 0
  x_pos <- nv_ifelse(at_zero, 1, x)
  lambda_pos <- nv_ifelse(at_zero, 1, lambda)
  saddle <- stirlerr(x_pos, is_f32) + bd0(x_pos, lambda_pos, is_f32)
  nv_ifelse(at_zero, -lambda, -saddle - 0.5 * nv_log(x_pos) - 0.5 * base::log(2 * pi))
}

# 2^-e for a normal x = f 2^e with 1 <= f < 2, from x's exponent bits: an exact
# rescaling, and no division. x must be below the largest power of two, or the
# result is not normal.
inv_pow2 <- function(x) {
  is_f32 <- dtype(x) == "f32"
  bits_dtype <- if (is_f32) "ui32" else "ui64"
  n_mantissa <- if (is_f32) 23L else 52L
  exponent_max <- if (is_f32) 254L else 2046L
  e <- nv_shift_right_logical(nv_bitcast_convert(x, bits_dtype), n_mantissa)
  nv_bitcast_convert(nv_shift_left(exponent_max - e, n_mantissa), dtype(x))
}

# `compute()` where `pending`, else `init`. XLA's CPU backend turns a
# conditional of elementwise branches into a select, which evaluates both; a
# loop of at most one trip is not converted, so this does skip the work.
skip_unless <- function(pending, init, compute) {
  nv_while(
    init = list(pending = pending, value = init),
    cond = function(pending, value) pending,
    body = function(pending, value) list(pending = nv_scalar(FALSE), value = compute())
  )$value
}

# Sum of the series
#   upper: lambda / (k + 1) + lambda^2 / ((k + 1)(k + 2)) + ...
#   lower: k / lambda + k (k - 1) / lambda^2 + ...  (finite: stops at k terms)
# which multiply the probability of k to give P(X > k) and P(X <= k) - p(k)
# respectively (R's `pd_upper_series()` and `pd_lower_series()`).
#
# Each term is the one before times u_n / a_n, with u_n = lambda and
# a_n = k + 1 + n (upper), or u_n = k - n and a_n = lambda (lower). Divisions
# are by far the most expensive operation here, so the sum is carried as a
# fraction num / den, with term / den the current term:
#   term <- term u_n,  num <- num a_n + term,  den <- den a_n,
# and divided out once at the end. Every u_n and a_n is first scaled by the
# same power of two c, which leaves both fractions alone and makes the a_n
# close to 1 + n / (k + 1); every 8 steps num, den and term are rescaled by a
# power of two that puts den in [1, 2), and (a_n c)^8 < 2^80 keeps them in
# range for the 1000 steps allowed. Both rescalings are exact, so the
# arithmetic is that of the plain fractions.
#
# The ratio of successive terms is below one and decreasing, so once a term is
# below (1 - ratio) * sum * eps / 2 the rest of the series cannot move the sum
# by half an ulp; without dividing, term u_(n+1) <= (a_(n+1) - u_(n+1)) num
# eps / 2. That takes at most ~170 terms in f64 and ~90 in f32 within the
# regions `ppois_raw()` uses the series in, but far fewer for most inputs, so
# the first steps are unrolled for every element, and a loop finishes the
# rest for those that need it, eight steps a trip. In the loop each element is
# frozen once converged (u = 0 and a = 1 leave num and den exactly as they
# are), so it does not depend on how long the others take. The unrolled steps
# are the same for every element anyway, and are not frozen: an element that
# has converged only takes on terms below half an ulp, or a rounding of num and
# den alike, and freezing each step doubled the cost of the function for no
# measurable change in accuracy. A while loop has no gradient, so the inputs
# must not carry one.
poisson_series <- function(k, lambda, upper, is_f32) {
  tol <- if (is_f32) 2^-24 else 2^-53
  # Tuned on the CPU backend for lambda from 1 to 150 (a trip costs about as
  # much as several unrolled steps; frozen steps cost more than plain ones)
  n_unroll <- if (is_f32) 8L else 24L
  steps_per_trip <- 8L
  # u_n = u0 - n du and a_n = a0 + n da, scaled by c so that a0 is in [1, 2)
  # (or a little more where it is beyond the largest power of two). The lower
  # series is finite: its u_n reach zero at n = k, after which the steps only
  # multiply a zero term. With k >= 1 it has lambda > k >= 1, so c <= 1 and
  # n du stays small; but at k = 0 (the only count below a mean under 1) c is
  # about 1 / lambda, and n c overflows for a small enough lambda, making the
  # zero term NaN. The series has no terms there at all, so du is zero.
  a0 <- nv_ifelse(upper, k + 1, lambda)
  c <- inv_pow2(nv_pmin(a0, if (is_f32) 2^126 else 2^1022))
  u0 <- nv_ifelse(upper, lambda, k) * c
  du <- nv_ifelse(upper | (k == 0), 0, c)
  a0 <- a0 * c
  da <- nv_ifelse(upper, c, 0)
  normalise <- function(f) {
    s <- inv_pow2(f$den)
    list(term = f$term * s, num = f$num * s, den = f$den * s)
  }
  step <- function(f, n, live = NULL) {
    u <- u0 - n * du
    a <- a0 + n * da
    if (!is.null(live)) {
      u <- nv_ifelse(live, u, 0)
      a <- nv_ifelse(live, a, 1)
    }
    term <- f$term * u
    list(term = term, num = f$num * a + term, den = f$den * a)
  }
  # Whether the series has converged after term n; if so, term is set to zero
  settle <- function(f, n) {
    u <- u0 - (n + 1) * du
    a <- a0 + (n + 1) * da
    f$term <- nv_ifelse(f$term * u <= (a - u) * f$num * tol, 0, f$term)
    f
  }

  f <- normalise(list(term = u0, num = u0, den = a0))
  for (n in seq_len(n_unroll)) {
    f <- step(f, n)
    if (n %% steps_per_trip == 0L || n == n_unroll) {
      f <- normalise(f)
    }
  }
  f <- settle(f, n_unroll)
  out <- nv_while(
    init = list(n = nv_scalar_like(k, n_unroll), term = f$term, num = f$num, den = f$den),
    cond = function(n, term, num, den) (n < 1000) & nv_any(term != 0),
    body = function(n, term, num, den) {
      f <- list(term = term, num = num, den = den)
      for (j in seq_len(steps_per_trip)) {
        f <- settle(step(f, n + j, live = f$term != 0), n + j)
      }
      f <- normalise(f)
      list(n = n + steps_per_trip, term = f$term, num = f$num, den = f$den)
    }
  )
  out$num / out$den
}

# log(1 + t) - t for |t| <= 1/4, by the series in r = t / (2 + t) of R's
# `log1pmx()`: r (2 r^2 S(r^2) - t), S(y) = 1/3 + y/5 + y^2/7 + ..., truncated
# where accurate to working precision for |t| <= 1/4. The coefficients
# 1 / (2j + 3) are highest power first.
log1pmx_f64_coefs <- 1 / c(19, 17, 15, 13, 11, 9, 7, 5, 3)
log1pmx_f32_coefs <- 1 / c(9, 7, 5, 3)

log1pmx_small <- function(t, is_f32) {
  r <- t / (2 + t)
  y <- r * r
  r * (2 * y * horner(y, if (is_f32) log1pmx_f32_coefs else log1pmx_f64_coefs) - t)
}

# Coefficients of the asymptotic expansion of the Poisson distribution function
# for large k with lambda close to k, as R's `ppois_asymp()`.
ppois_asymp_a <- c(
  2 / 3,
  -4 / 135,
  8 / 2835,
  16 / 8505,
  -8992 / 12629925,
  -334144 / 492567075,
  698752 / 1477701225
)
ppois_asymp_b <- c(
  1 / 12,
  1 / 288,
  -139 / 51840,
  -571 / 2488320,
  163879 / 209018880,
  5246819 / 75246796800,
  -534703531 / 902961561600
)

# P(X <= k) (or P(X > k)) for whole k with lambda close to k, both large, as R's
# `ppois_asymp()`: a normal approximation in the signed root of the deviance,
# corrected by an expansion in 1/k.
ppois_asymp <- function(k, lambda, lower_tail, log_p, is_f32) {
  dfm <- lambda - k
  # Near the top of the range the reciprocal of k that XLA divides by would be
  # subnormal, so both are scaled down first (see `bd0()`)
  huge <- k > (if (is_f32) 2^120 else 2^1016)
  t <- nv_ifelse(huge, dfm * 2^-4, dfm) / nv_ifelse(huge, k * 2^-4, k)
  pt <- -log1pmx_small(t, is_f32)
  # As two square roots, as 2 * k can overflow
  s2pt <- nv_sqrt(2 * pt) * nv_sqrt(k)
  s2pt <- nv_ifelse(dfm < 0, -s2pt, s2pt)

  res12 <- 0
  res1_term <- res1_ig <- nv_sqrt(k)
  res2_term <- res2_ig <- s2pt
  for (i in seq_along(ppois_asymp_a)) {
    res12 <- res12 + res1_ig * ppois_asymp_a[[i]] + res2_ig * ppois_asymp_b[[i]]
    res1_term <- res1_term * pt / i
    res2_term <- res2_term * 2 * pt / (2 * i + 1)
    res1_ig <- res1_ig / k + res1_term
    res2_ig <- res2_ig / k + res2_term
  }
  elfb <- k + horner(1 / k, rev(ppois_asymp_b))
  if (!lower_tail) {
    elfb <- -elfb
  }
  f <- res12 / elfb

  np <- nv_pnorm(s2pt, lower_tail = !lower_tail, log_p = log_p)
  log_dnorm <- -0.5 * s2pt * s2pt - 0.5 * base::log(2 * pi)
  if (log_p) {
    # `nv_pnorm(log_p = TRUE)` takes log1p(-erfc) above one half, which the
    # CPU plugin's f64 `log1p()` gets wrong by up to ~120 ulp for probabilities
    # in about (0.586, 0.64) (openxla/xla#46765, see `log1mexp()`). There the
    # plain log is accurate. Drop this once the plugin includes that fix.
    np_plain <- nv_pnorm(s2pt, lower_tail = !lower_tail)
    np <- nv_ifelse((np_plain >= 0.5) & (np_plain <= 2 / 3), nv_log(np_plain), np)
    np + nv_log1p(f * nv_exp(log_dnorm - np))
  } else {
    np + f * nv_exp(log_dnorm)
  }
}

# The Poisson distribution function at a whole k >= 0 and finite lambda > 0,
# with the regions of R's `pgamma_raw()` (for its shape k + 1 and x = lambda):
#   - lambda well below k: P(X > k) = p(k) * (upper series);
#   - lambda well above k: P(X <= k) = p(k) * (1 + lower series);
#   - otherwise k and lambda are close and large: the asymptotic expansion.
# Each series gives one tail directly and the other as its complement, which is
# taken on the log scale to keep full accuracy either side. Returns the
# probability, the log of p(k), from which the gradient follows, and which
# elements took a series (whose error `nv_qpois()` allows for). A while
# loop has no gradient, so the inputs must not carry one.
ppois_raw <- function(k, lambda, lower_tail, log_p, is_f32) {
  upper <- (lambda <= k) & (lambda < 0.8 * (k + 51))
  lower <- (k < lambda) & (k + 1 < 0.8 * (lambda + 50))
  use_series <- upper | lower

  log_density <- dpois_log_raw(k, lambda, is_f32)

  # The asymptotic expansion is computed only if some element needs it (see
  # `skip_unless()`). The series is not gated the same way, though it costs
  # about a third of the time of a batch that is all asymptotic (large lambda
  # with counts near it): its result then has to cross a loop boundary, which
  # on the CPU backend cost up to ~18% (f64) on batches that need only the
  # series (lambda below ~200, the common case for count data) at 1e7
  # elements, ~3-5% at 1e6.
  series_branch <- function() {
    # Elements outside both series are given k = 0, whose lower series is 0
    sum <- poisson_series(nv_ifelse(use_series, k, 0), nv_ifelse(use_series, lambda, 1), upper, is_f32)
    # The upper series starts at lambda / (k + 1), which underflows for a tiny
    # enough lambda (below the smallest normal times k + 1), leaving the sum zero
    # and its log -Inf. The later terms are then below eps of it, so the series
    # is lambda / (k + 1) to working precision, and its log is taken directly.
    # Only computed if some element needs it.
    smallest_normal <- if (is_f32) 2^-126 else 2^-1022
    underflowed <- upper & (sum < smallest_normal)
    log_upper_sum <- nv_log(nv_ifelse(upper, sum, 1))
    log_upper_sum <- skip_unless(nv_any(underflowed), log_upper_sum, function() {
      nv_ifelse(
        underflowed,
        -log_ratio(nv_ifelse(underflowed, k + 1, 1), nv_ifelse(underflowed, lambda, 1), is_f32),
        log_upper_sum
      )
    })
    log_tail <- log_density + nv_ifelse(upper, log_upper_sum, nv_log1p(sum))
    # `log_tail` is the log of P(X > k) for the upper series, P(X <= k) for the lower
    direct <- if (lower_tail) lower else upper
    from_series <- if (log_p) {
      nv_ifelse(direct, log_tail, log1mexp(log_tail))
    } else {
      nv_ifelse(direct, nv_exp(log_tail), -nv_expm1(log_tail))
    }
    from_series
  }
  from_series <- series_branch()
  asymp <- skip_unless(nv_any(!use_series), nv_fill_like(k, 0), function() {
    ppois_asymp(
      nv_ifelse(use_series, 200, k),
      nv_ifelse(use_series, 200, lambda),
      lower_tail,
      log_p,
      is_f32
    )
  })
  list(value = nv_ifelse(use_series, from_series, asymp), log_density = log_density, use_series = use_series)
}

#' @title The Poisson Distribution
#' @name nv_poisson
#' @description
#' Probability mass function (`nv_dpois`), distribution function (`nv_ppois`),
#' and quantile function (`nv_qpois`) for the Poisson distribution with mean
#' `lambda`.
#' @param x ([`arrayish`])\cr
#'   Values at which to evaluate the probability mass function. Values that are
#'   not whole numbers (up to a relative tolerance of `1e-9`, as in base R) have
#'   probability zero.
#' @param q ([`arrayish`])\cr
#'   Quantiles at which to evaluate the distribution function. Values that are
#'   not whole numbers are rounded down, except that, as in base R, one within
#'   `1e-7` below a whole number counts as that number.
#' @param p ([`arrayish`])\cr
#'   Probabilities at which to evaluate the quantile function. Values outside
#'   \eqn{[0, 1]} give `NaN`. With `log_p = TRUE`, supply log-probabilities
#'   in \eqn{[-\infty, 0]} instead.
#' @param lambda ([`arrayish`])\cr
#'   Mean of the distribution. Scalars broadcast; otherwise `lambda` and
#'   `x`/`q`/`p` must have the same shape. Negative values give `NaN`.
#' @param log,log_p (`logical(1)`)\cr
#'   If `TRUE`, the probabilities are given as logarithms. For `nv_qpois` this
#'   describes the input `p`.
#' @param lower_tail (`logical(1)`)\cr
#'   If `TRUE` (default), probabilities are \eqn{P(X \le x)}; otherwise,
#'   \eqn{P(X > x)}.
#' @details
#' The Poisson distribution has probability mass function:
#' \deqn{p(x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0, 1, 2, \ldots}
#' where \eqn{\lambda} is `lambda`.
#'
#' `nv_dpois` uses the saddle point expansion of
#' `r cite_bib("loader2000fast")`, as base R does. `nv_ppois` follows base R's
#' `pgamma()`: a series in the tails and an asymptotic expansion where `q` and
#' `lambda` are large and close. `nv_qpois` searches from a Cornish-Fisher
#' approximation for the smallest whole number \eqn{x} with
#' \eqn{P(X \le x) \ge p}, or with \eqn{P(X > x) < p} for
#' `lower_tail = FALSE`; with `log_p = TRUE`, `p` is the log of that
#' probability.
#'
#' The comparison with `p` allows for rounding, with the intention that a `p`
#' computed by the distribution function at \eqn{x} returns \eqn{x}; this is
#' not guaranteed. The allowance has two parts: base R's, a few ulp of `p` (or
#' of `log p`), and one for the error of the distribution function, relative
#' to the smaller of the two tail probabilities (in `f64`, from about `1e-14`
#' of it near `lambda` to about `1e-12` far into the tails, and more where a
#' series is summed far from `lambda`). A `p` within the
#' allowance of a jump of the distribution function cannot be told apart from
#' one exactly at it, and its quantile can be off in either direction. On the
#' probability scale near `p = 1`, base R's part is proportional to `p` rather
#' than to the small upper tail, so it can span several jumps at large
#' `lambda` and the quantile can then be off by more than one. Supplying the
#' probability as the upper tail with `lower_tail = FALSE`, or as a log with
#' `log_p = TRUE`, avoids this only if it is computed accurately in that form:
#' converting a probability already rounded near one (e.g. `1 - p`, `log(p)`)
#' cannot recover what the rounding lost.
#' In `f32` the rounding of `p` itself dominates. The allowance is empirical:
#' outside it, no quantile errors were found in the tested ranges (`lambda` up
#' to `1e14`), which is not a guarantee beyond them.
#'
#' `lambda = 0` is the point mass at zero, and a `lambda` of `Inf` is admitted
#' as in base R: the probabilities are those of an infinitely distant mass,
#' and quantiles are `NaN`.
#'
#' @section Numerical limits:
#' On the XLA CPU backend these functions were tested on, subnormal numbers are
#' treated as zero (other backends have not been checked). There, a `lambda`
#' below the smallest normal float (`2^-126` in `f32`, `2^-1022` in `f64`) acts
#' as `lambda = 0`, and a probability or density that would be subnormal is
#' returned as zero: `nv_dpois(1, 2^-126)` in `f32` is `0`, where base R gives
#' `2^-126`. On the log scale such values stay finite.
#'
#' Above `2^24` in `f32` and `2^53` in `f64` not every whole number is a
#' float. `nv_qpois` then returns the smallest float \eqn{x} with
#' \eqn{P(X \le x) \ge p}, which can be one float above base R's answer (base
#' R's result can correspond to rounding the mathematical quantile to the
#' available precision). Where no finite float qualifies, which can happen
#' only at `lambda` equal to the largest float, `nv_qpois` returns the largest
#' float, as base R does: a saturating convention, and an exception to that
#' rule.
#'
#' @section Gradients:
#' `nv_dpois` and `nv_ppois` are differentiable in `lambda`; for `nv_ppois`
#' only first derivatives are supported. At `lambda = 0`, where only a
#' right derivative exists, the implementation returns it for
#' `nv_dpois(0, lambda)` and, by convention, zero for the other probabilities
#' resolved there directly. With respect to `x`/`q`, which are used only as whole
#' numbers, autodiff returns zero, as it does for all arguments of `nv_qpois`,
#' a step function.
#'
#' @templateVar dist pois
#' @templateVar params `lambda`
#' @template section_distribution_dtype
#' @references
#' `r format_bib("loader2000fast")`
#' @return ([`arrayish`])\cr
#' `nv_dpois()`, `nv_ppois()`, and `nv_qpois()` return an [`arrayish`] with the
#' data type of `x`/`q`/`p` and the shape after scalar broadcasting with
#' `lambda`.
#'
#' @examplesIf pjrt::plugins_downloaded()
#' x <- nv_array(c(0, 1, 2, 5, 10))
#' nv_dpois(x)
#' nv_dpois(x, lambda = 4)
#' nv_dpois(x, lambda = 4, log = TRUE)
#'
#' nv_ppois(x, lambda = 4)
#' nv_ppois(x, lambda = 4, lower_tail = FALSE)
#' nv_ppois(x, lambda = 4, log_p = TRUE)
#'
#' p <- nv_array(c(0.025, 0.5, 0.975))
#' nv_qpois(p, lambda = 4)
#' nv_qpois(p, lambda = 4, lower_tail = FALSE)
#' nv_qpois(nv_array(c(-700, -2, -0.1), dtype = "f64"), lambda = 1000, log_p = TRUE)
NULL

#' @rdname nv_poisson
#' @export
nv_dpois <- jit(
  function(x, lambda = 1, log = FALSE) {
    assert_flag(log)
    args <- promote_distribution_args(x = x, lambda = lambda)
    args <- nv_broadcast_scalars(x = args$x, lambda = args$lambda)
    x <- args$x
    lambda <- args$lambda
    op_dtype <- assert_rng_float_dtype(dtype(x), arg = "x")
    is_f32 <- op_dtype == "f32"

    # As base R, `x` is rounded to the nearest whole number, once anything not
    # within a relative 1e-9 of one has been given probability zero. Infinite
    # `x` fails the test too (x - round(x) is NaN). Only the rounded value is
    # used, so the gradient wrt `x` is zero; it is detached so that the NaN of
    # an infinite `x` cannot reach it through the test.
    x <- stop_gradient(x)
    x_whole <- nv_round(x)
    is_whole <- nv_abs(x - x_whole) <= 1e-9 * nv_pmax(nv_abs(x), 1)
    # The sign is tested before rounding, as base R does: a tiny negative `x`
    # rounds to zero but is outside the support
    zero <- !is_whole | (x < 0) | (lambda == Inf) | ((lambda == 0) & (x_whole != 0))
    valid <- !nv_is_nan(x) & (lambda >= 0)

    # Each branch is fed a safe stand-in where it is not selected, so it cannot
    # poison the gradient
    at_zero <- x_whole == 0
    pos <- valid & !zero & !at_zero
    x_pos <- nv_ifelse(pos, x_whole, 1)
    lambda_pos <- nv_ifelse(pos, lambda, 1)
    lambda_zero <- nv_ifelse(valid & !zero & at_zero, lambda, 0)

    saddle <- stirlerr(x_pos, is_f32) + bd0(x_pos, lambda_pos, is_f32)
    density <- if (log) {
      nv_ifelse(at_zero, -lambda_zero, -saddle - 0.5 * nv_log(x_pos) - 0.5 * base::log(2 * pi))
    } else {
      # sqrt(2 pi x) as two factors, which cannot overflow
      nv_ifelse(at_zero, nv_exp(-lambda_zero), nv_exp(-saddle) / (base::sqrt(2 * pi) * nv_sqrt(x_pos)))
    }
    nv_ifelse(valid, nv_ifelse(zero, if (log) -Inf else 0, density), NaN)
  },
  static = "log"
)

#' @rdname nv_poisson
#' @export
nv_ppois <- jit(
  function(q, lambda = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(q = q, lambda = lambda)
    args <- nv_broadcast_scalars(q = args$q, lambda = args$lambda)
    q <- args$q
    lambda <- args$lambda
    op_dtype <- assert_rng_float_dtype(dtype(q), arg = "q")
    is_f32 <- op_dtype == "f32"

    # Resolved directly, in base R's order: P(X <= q) is 0 below the support,
    # 1 for `lambda = 0` or `q = Inf`, and 0 for `lambda = Inf`.
    below <- q < 0
    at_one <- (lambda == 0) | (q == Inf)
    valid <- !nv_is_nan(q) & (lambda >= 0)
    regular <- valid & !below & !at_one & (lambda != Inf)
    dt0 <- if (lower_tail) (if (log_p) -Inf else 0) else (if (log_p) 0 else 1)
    dt1 <- if (lower_tail) (if (log_p) 0 else 1) else (if (log_p) -Inf else 0)

    # As base R, `q` within 1e-7 below a whole number counts as that number,
    # but tested on the distance to it: in f32 adding 1e-7 to `q` would round
    # up to the next whole number from much further away
    q_reg <- stop_gradient(nv_ifelse(regular, q, 0))
    k <- nv_floor(q_reg)
    k <- nv_ifelse((k + 1) - q_reg <= 1e-7, k + 1, k)
    lambda_reg <- nv_ifelse(regular, lambda, 1)
    raw <- ppois_raw(k, stop_gradient(lambda_reg), lower_tail, log_p, is_f32)

    # The summation is not differentiable, so the exact derivative wrt `lambda`,
    # -/+ p(k) (over the probability on the log scale), is attached by adding
    # zero times it. It is zeroed where it is not finite, which would make the
    # zero NaN.
    grad <- nv_exp(if (log_p) raw$log_density - raw$value else raw$log_density)
    grad <- nv_ifelse(nv_is_finite(grad), if (lower_tail) -grad else grad, 0)
    value <- raw$value + (lambda_reg - stop_gradient(lambda_reg)) * grad

    res <- nv_ifelse(below, dt0, nv_ifelse(at_one, dt1, nv_ifelse(lambda == Inf, dt0, value)))
    nv_ifelse(valid, res, NaN)
  },
  static = c("lower_tail", "log_p")
)

#' @rdname nv_poisson
#' @export
nv_qpois <- jit(
  function(p, lambda = 1, lower_tail = TRUE, log_p = FALSE) {
    assert_flag(lower_tail)
    assert_flag(log_p)
    args <- promote_distribution_args(p = p, lambda = lambda)
    args <- nv_broadcast_scalars(p = args$p, lambda = args$lambda)
    op_dtype <- assert_rng_float_dtype(dtype(args$p), arg = "p")
    # A step function has a zero gradient, and the search below is a while
    # loop, which has none at all
    p <- stop_gradient(args$p)
    lambda <- stop_gradient(args$lambda)
    is_f32 <- op_dtype == "f32"
    eps <- if (is_f32) 2^-23 else 2^-52
    max_float <- if (is_f32) (2 - 2^-23) * 2^127 else .Machine$double.xmax

    # Resolved directly, in base R's order: NaN outside the admissible range,
    # 0 for `lambda = 0` and at the bottom of the probability scale, and Inf
    # at the top
    if (log_p) {
      in_range <- p <= 0
      p_zero <- p == (if (lower_tail) -Inf else 0)
      p_one <- p == (if (lower_tail) 0 else -Inf)
    } else {
      in_range <- (p >= 0) & (p <= 1)
      p_zero <- p == (if (lower_tail) 0 else 1)
      p_one <- p == (if (lower_tail) 1 else 0)
    }
    valid <- in_range & nv_is_finite(lambda) & (lambda >= 0)
    zero_result <- (lambda == 0) | p_zero
    regular <- valid & !zero_result & !p_one
    p <- nv_ifelse(regular, p, if (log_p) -1 else 0.5)
    lambda <- nv_ifelse(regular, lambda, 1)

    # Cornish-Fisher starting value, with the skewness correction. base R rounds
    # it, but it lands about a half above the quantile of a `p` at a jump, so
    # rounding often starts the search one too high; rounded down, the
    # quantile is nearly always the start or the next count, which the search
    # below settles in two evaluations. The start does not change the answer.
    sigma <- nv_sqrt(lambda)
    z <- nv_qnorm(p, lower_tail = lower_tail, log_p = log_p)
    y0 <- nv_pmax(nv_floor(lambda + sigma * (z + (z * z - 1) / (6 * sigma))), 0)
    y0 <- nv_ifelse(nv_is_finite(y0), y0, 0)

    # The quantile is the smallest whole y at which this holds. As base R, the
    # comparison is fuzzed by a few ulp, so that a `p` computed as the
    # distribution function at a whole number is not missed through rounding.
    p_fuzzed <- if (log_p) {
      if (lower_tail) p * (1 + 2 * eps) else p * (1 - 2 * eps)
    } else {
      if (lower_tail) p * (1 - 8 * eps) else nv_ifelse(1 - p > 32 * eps, p * (1 + 8 * eps), p)
    }
    # In f64 the error of the distribution function itself also needs allowing
    # for, so that a `p` from base R at a jump is not missed: relative to the
    # smaller of the two tails it grows like (1 + |log tail|) ulp, plus
    # |y - lambda| ulp where a series was summed (from bd0()'s x log(x / np)).
    # The factor 16 is an empirical allowance, not a proven bound: measured
    # against base R for lambda from 0.5 to 1e14, and against MPFR at the jumps
    # for lambda up to 1e8, the error stayed under 9 such ulp in the series and
    # under 4 in the asymptotic expansion. Any wider and a `p` just above a jump
    # could be read as on it, giving a quantile one too low. In f32, rounding
    # `p` to f32 displaces it further than this, and a wider fuzz only costs
    # agreement elsewhere.
    if (!is_f32) {
      log_upper <- if (log_p) log1mexp(p) else nv_log1p(-p)
      log_lower <- if (log_p) p else nv_log(p)
      log_small_tail <- nv_pmin(log_lower, log_upper)
      # How far `p` may move per unit of relative error in the smaller tail
      scale <- if (log_p) nv_pmin(-p, 1) else nv_pmin(p, 1 - p)
    }
    satisfied <- function(y) {
      raw <- ppois_raw(y, lambda, lower_tail, log_p, is_f32)
      prob <- raw$value
      threshold <- p_fuzzed
      if (!is_f32) {
        ulps <- 1 - log_small_tail + nv_ifelse(raw$use_series, nv_abs(y - lambda), 0)
        extra <- nv_pmin(16 * eps * ulps, 0.25) * scale
        threshold <- if (lower_tail) threshold - extra else threshold + extra
      }
      # The largest float is taken as satisfying (see `next_probe()`). Below
      # lambda = the largest float that changes nothing: the condition holds a
      # few sd above lambda, far less than an ulp there, so the largest float
      # already satisfies it wherever the search reaches it. At lambda = the
      # largest float there may be no finite float that satisfies it, and this
      # returns the largest float instead of Inf: a saturating convention, as
      # base R's.
      (if (lower_tail) prob >= threshold else prob < threshold) | (y == max_float)
    }

    # Bracket the quantile between `lo` (not satisfied; -1 if none is known)
    # and `hi` (satisfied; Inf if none is known), galloping away from y0 until
    # both are known, then bisecting. The gallop steps are s, s, 2 s, 4 s, ...
    # (`step` starts at s / 2 and doubles, and moves by its ceiling), so that a
    # quantile two away is reached without overshooting; s is at least an ulp
    # of y0, so the first step always moves. Every element pays for as many
    # evaluations as the slowest one needs. The probe at y0 is made before the
    # loop: inside it, as the loop's first trip, the program would hold one copy
    # of the distribution function rather than two and compile faster, but each
    # call would pay a trip's worth of loop overhead.
    sat0 <- satisfied(y0)
    lo <- nv_ifelse(sat0, -1, y0)
    hi <- nv_ifelse(sat0, y0, Inf)
    step <- 0.5 * nv_pmax(nv_floor(y0 * (4 * eps)), 1)
    next_probe <- function(lo, hi, step) {
      up <- hi == Inf
      down <- !up & (lo == -1)
      move <- nv_ceiling(step)
      # Galloping up is capped at the largest float, which a step of a few ulps
      # would otherwise pass straight to Inf near the top of the range
      probe <- nv_ifelse(
        up,
        nv_pmin(lo + move, max_float),
        nv_ifelse(down, nv_pmax(hi - move, 0), nv_floor(lo + 0.5 * (hi - lo)))
      )
      list(probe = probe, active = (probe > lo) & (probe < hi), gallop = up | down)
    }
    out <- nv_while(
      init = list(n = nv_scalar_like(p, 0), lo = lo, hi = hi, step = step),
      cond = function(n, lo, hi, step) {
        (n < (if (is_f32) 320 else 2200)) & nv_any(next_probe(lo, hi, step)$active)
      },
      body = function(n, lo, hi, step) {
        nxt <- next_probe(lo, hi, step)
        sat <- satisfied(nv_ifelse(nxt$active, nxt$probe, 0))
        list(
          n = n + 1,
          lo = nv_ifelse(nxt$active & !sat, nxt$probe, lo),
          hi = nv_ifelse(nxt$active & sat, nxt$probe, hi),
          step = nv_ifelse(nxt$active & nxt$gallop, 2 * step, step)
        )
      }
    )

    res <- nv_ifelse(zero_result, 0, nv_ifelse(p_one, Inf, out$hi))
    nv_ifelse(valid, res, NaN)
  },
  static = c("lower_tail", "log_p")
)
