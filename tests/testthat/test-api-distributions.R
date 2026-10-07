# Limit configurations exercised against base R by the nv_dunif/nv_punif/nv_qunif
# agreement tests: ordinary, degenerate, reversed, infinite and NaN.
uniform_limit_cases <- function() {
  list(
    c(0, 1),
    c(-1, 2),
    c(1, 1),
    c(2, 1),
    c(-Inf, Inf),
    c(0, Inf),
    c(-Inf, 0),
    c(Inf, Inf),
    c(-Inf, -Inf),
    c(NaN, 1),
    c(0, NaN)
  )
}
as_f64 <- function(v) nv_array(v, dtype = "f64")
as_f64_scalar <- function(v) nv_scalar(v, dtype = "f64")

describe("nv_dnorm", {
  it("matches base R dnorm() with default mean/sd", {
    x <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_dnorm(nv_array(x))),
      dnorm(x),
      tolerance = 1e-6
    )
  })

  it("matches base R dnorm() with custom mean/sd", {
    x <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_dnorm(nv_array(x), mean = 1, sd = 2)),
      dnorm(x, mean = 1, sd = 2),
      tolerance = 1e-6
    )
  })

  it("log = TRUE matches base R dnorm(..., log = TRUE)", {
    x <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_dnorm(nv_array(x), log = TRUE)),
      dnorm(x, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("log = TRUE stays finite where the plain density underflows to 0", {
    x <- nv_array(40)
    expect_equal(as.vector(nv_dnorm(x)), 0)
    expect_equal(
      as.vector(nv_dnorm(x, log = TRUE)),
      dnorm(40, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("non-scalar mean/sd works", {
    x <- c(0, 0, 0)
    mean <- c(-1, 0, 1)
    sd <- c(1, 2, 3)
    expect_equal(
      as.vector(nv_dnorm(
        nv_array(x),
        mean = nv_array(mean),
        sd = nv_array(sd)
      )),
      dnorm(x, mean = mean, sd = sd),
      tolerance = 1e-6
    )
  })

  it("converts mean/sd to the dtype of x", {
    out <- nv_dnorm(nv_array(c(0, 1), dtype = "f32"), mean = 0L, sd = 1L)
    expect_dtype(out, "f32")
  })

  it("works under jit with log as a static argument", {
    x <- c(-1, 0, 1)
    expect_equal(
      as.vector(nv_dnorm(nv_array(x))),
      dnorm(x),
      tolerance = 1e-6
    )
    expect_equal(
      as.vector(nv_dnorm(nv_array(x), log = TRUE)),
      dnorm(x, log = TRUE),
      tolerance = 1e-6
    )
  })
})

describe("nv_pnorm", {
  it("matches base R pnorm() with default mean/sd", {
    q <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q))),
      pnorm(q),
      tolerance = 1e-6
    )
  })

  it("matches base R pnorm() with custom mean/sd", {
    q <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q), mean = 1, sd = 2)),
      pnorm(q, mean = 1, sd = 2),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R pnorm(..., lower.tail = FALSE)", {
    q <- c(-2, -1, 0, 0.5, 1, 2)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q), lower_tail = FALSE)),
      pnorm(q, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R pnorm(..., log.p = TRUE) around the direct/asymptotic threshold", {
    q <- c(-2, -1, 0, 0.5, 1, 2, -15, -20, -25)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q, dtype = "f64"), log_p = TRUE)),
      pnorm(q, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly, including via the asymptotic branch", {
    q <- c(-2, -1, 0, 0.5, 1, 2, 25)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q, dtype = "f64"), lower_tail = FALSE, log_p = TRUE)),
      pnorm(q, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE stays finite deep in the tail where erfc() underflows to 0", {
    q <- nv_array(-40, dtype = "f64")
    expect_equal(as.vector(nv_pnorm(q)), 0)
    expect_equal(
      as.vector(nv_pnorm(q, log_p = TRUE)),
      pnorm(-40, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("gradient stays finite deep in the tail (asymptotic branch doesn't poison it via nv_ifelse)", {
    f <- function(q) nv_pnorm(q, log_p = TRUE)
    g <- as.vector(jit(gradient(f, wrt = "q"))(nv_scalar(-40, dtype = "f64"))[[1L]])
    expect_true(is.finite(g))
  })

  it("log_p = TRUE has a dtype-aware lower threshold, closing the f32 gap between where erfc() underflows and a fixed f64 threshold would kick in", {
    q <- c(-11, -13, -15, -17, -19)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q, dtype = "f32"), log_p = TRUE)),
      pnorm(q, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE stays accurate far in the upper tail (probability close to 1), in both f32 and f64", {
    q32 <- c(6, 10, 12.9)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q32, dtype = "f32"), log_p = TRUE)),
      pnorm(q32, log.p = TRUE),
      tolerance = 1e-5
    )
    q64 <- c(9, 20, 40, 100)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q64, dtype = "f64"), log_p = TRUE)),
      pnorm(q64, log.p = TRUE),
      tolerance = 1e-5
    )
  })

  it("log_p = TRUE upper tail composes correctly with lower_tail = FALSE (mirrors the lower tail)", {
    q <- c(-9, -20, -40)
    expect_equal(
      as.vector(nv_pnorm(nv_array(q, dtype = "f64"), lower_tail = FALSE, log_p = TRUE)),
      pnorm(q, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-5
    )
  })

  it("gradient stays finite far in the upper tail (upper branch doesn't poison it via nv_ifelse)", {
    f <- function(q) nv_pnorm(q, log_p = TRUE)
    g <- as.vector(jit(gradient(f, wrt = "q"))(nv_scalar(40, dtype = "f64"))[[1L]])
    expect_true(is.finite(g))
  })

  it("non-scalar mean/sd works", {
    q <- c(0, 0, 0)
    mean <- c(-1, 0, 1)
    sd <- c(1, 2, 3)
    expect_equal(
      as.vector(nv_pnorm(
        nv_array(q),
        mean = nv_array(mean),
        sd = nv_array(sd)
      )),
      pnorm(q, mean = mean, sd = sd),
      tolerance = 1e-6
    )
  })

  it("converts mean/sd to the dtype of q", {
    out <- nv_pnorm(nv_array(c(0, 1), dtype = "f32"), mean = 0L, sd = 1L)
    expect_dtype(out, "f32")
  })
})

describe("nv_qnorm", {
  it("matches base R qnorm() with default mean/sd", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p))),
      qnorm(p),
      tolerance = 1e-6
    )
    # At `f64` the answer is accurate to `f64`, not to whatever the default
    # float is. The coefficients are plain R numbers with nothing typed to
    # yield to, so they used to materialize at the default -- and `qnorm(0.975)`
    # came back with an error of 1e-8, `f32` accuracy in an `f64` computation.
    expect_equal(
      as.vector(nv_qnorm(nv_array(p, dtype = "f64"))),
      qnorm(p),
      tolerance = 1e-13
    )
  })

  it("names the operand when it is not a float", {
    # Reported as a failure to bring `mean` to the operand's data type before.
    expect_error(nv_qnorm(nv_array(1L)), "`p` must be a float data type")
    expect_error(nv_pnorm(nv_array(1L)), "`q` must be a float data type")
    expect_error(nv_dnorm(nv_array(1L)), "`x` must be a float data type")
  })

  it("matches base R qnorm() with custom mean/sd", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p), mean = 1, sd = 2)),
      qnorm(p, mean = 1, sd = 2),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R qnorm(..., lower.tail = FALSE)", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p), lower_tail = FALSE)),
      qnorm(p, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("matches base R across both rational regimes and their crossover", {
    # exp(-2) is the central <-> tail threshold and exp(-32) the near <-> far
    # tail one; the near-1 values exercise the upper reflection.
    p <- c(1e-300, exp(-32), 1e-5, exp(-2), 0.3, 1 - exp(-2), 1 - 1e-10)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p, dtype = "f64"))),
      qnorm(p),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R qnorm(..., log.p = TRUE)", {
    lp <- c(-729, -100, -32, -25, -2, -0.7, -0.1, -1e-10)
    expect_equal(
      as.vector(nv_qnorm(nv_array(lp, dtype = "f64"), log_p = TRUE)),
      qnorm(lp, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE reaches quantiles a bare probability cannot express", {
    expect_equal(as.vector(nv_qnorm(nv_array(0, dtype = "f64"))), -Inf)
    expect_equal(
      as.vector(nv_qnorm(nv_array(-1000, dtype = "f64"), log_p = TRUE)),
      qnorm(-1000, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly", {
    lp <- c(-100, -2, -0.7, -0.1)
    expect_equal(
      as.vector(nv_qnorm(
        nv_array(lp, dtype = "f64"),
        lower_tail = FALSE,
        log_p = TRUE
      )),
      qnorm(lp, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("returns the infinite boundaries and NaN outside [0, 1]", {
    p <- c(0, 1, -0.25, 1.25, NaN)
    expect_equal(
      as.vector(nv_qnorm(nv_array(p, dtype = "f64"))),
      c(-Inf, Inf, NaN, NaN, NaN)
    )
    lp <- c(-Inf, 0, 0.5, NaN)
    expect_equal(
      as.vector(nv_qnorm(nv_array(lp, dtype = "f64"), log_p = TRUE)),
      c(-Inf, Inf, NaN, NaN)
    )
  })

  it("gradient matches 1 / dnorm(qnorm(p))", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9)
    f <- function(p) nv_sum(nv_qnorm(p))
    g <- as.vector(jit(gradient(f, wrt = "p"))(nv_array(p, dtype = "f64"))[[1L]])
    expect_equal(g, 1 / dnorm(qnorm(p)), tolerance = 1e-6)
  })

  it("gradient is not halved at the central/tail threshold", {
    f <- function(p) nv_sum(nv_qnorm(p))
    g <- as.vector(jit(gradient(f, wrt = "p"))(
      nv_array(exp(-2), dtype = "f64")
    )[[1L]])
    expect_equal(g, 1 / dnorm(qnorm(exp(-2))), tolerance = 1e-6)

    flog <- function(p) nv_sum(nv_qnorm(p, log_p = TRUE))
    glog <- as.vector(jit(gradient(flog, wrt = "p"))(
      nv_array(-2, dtype = "f64")
    )[[1L]])
    expect_equal(
      glog,
      exp(-2) / dnorm(qnorm(-2, log.p = TRUE)),
      tolerance = 1e-6
    )
  })

  it("gradient stays finite deep in the log tail", {
    f <- function(p) nv_sum(nv_qnorm(p, log_p = TRUE))
    g <- as.vector(jit(gradient(f, wrt = "p"))(
      nv_array(c(-1e4, -1e5), dtype = "f64")
    )[[1L]])
    expect_true(all(is.finite(g)))
  })

  it("gradients wrt mean/sd are exact", {
    p <- c(0.025, 0.9)
    f <- function(p, mean, sd) nv_sum(nv_qnorm(p, mean, sd))
    g <- jit(gradient(f, wrt = c("mean", "sd")))(
      nv_array(p, dtype = "f64"),
      nv_array(c(1, 1), dtype = "f64"),
      nv_array(c(2, 2), dtype = "f64")
    )
    expect_equal(as.vector(g[[1L]]), c(1, 1))
    expect_equal(as.vector(g[[2L]]), qnorm(p), tolerance = 1e-6)
  })

  it("inverts nv_pnorm", {
    x <- c(-4, -1, 0, 1, 4)
    expect_equal(
      as.vector(nv_qnorm(nv_pnorm(nv_array(x, dtype = "f64")))),
      x,
      tolerance = 1e-6
    )
  })

  it("non-scalar mean/sd works", {
    p <- c(0.1, 0.5, 0.9)
    mean <- c(-1, 0, 1)
    sd <- c(1, 2, 3)
    expect_equal(
      as.vector(nv_qnorm(
        nv_array(p),
        mean = nv_array(mean),
        sd = nv_array(sd)
      )),
      qnorm(p, mean = mean, sd = sd),
      tolerance = 1e-6
    )
  })

  it("converts mean/sd to the dtype of p", {
    # `p`'s data type, not the default float. The two coincide under the
    # standard defaults, which is what let the result follow the default
    # unnoticed while the coefficients materialized there.
    out <- nv_qnorm(nv_array(c(0.25, 0.75), dtype = "f32"), mean = 0L, sd = 1L)
    expect_dtype(out, "f32")
    out64 <- nv_qnorm(nv_array(c(0.25, 0.75), dtype = "f64"), mean = 0L, sd = 1L)
    expect_dtype(out64, "f64")
  })
})

describe("nv_dunif", {
  it("matches base R dunif() with default min/max", {
    x <- c(-0.5, 0, 0.25, 0.75, 1, 1.5)
    expect_equal(
      as.vector(nv_dunif(nv_array(x))),
      dunif(x),
      tolerance = 1e-6
    )
  })

  it("matches base R dunif() with custom min/max", {
    x <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_dunif(nv_array(x), min = -1, max = 2)),
      dunif(x, min = -1, max = 2),
      tolerance = 1e-6
    )
  })

  it("log = TRUE matches base R dunif(..., log = TRUE)", {
    x <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_dunif(nv_array(x), min = -1, max = 2, log = TRUE)),
      dunif(x, min = -1, max = 2, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("is constant on a support that includes both endpoints and zero outside it", {
    x <- nv_array(c(-1e-6, 0, 0.5, 1, 1 + 1e-6))
    expect_equal(as.vector(nv_dunif(x)), c(0, 1, 1, 1, 0))
    expect_equal(as.vector(nv_dunif(x, log = TRUE)), c(-Inf, 0, 0, 0, -Inf))
  })

  it("propagates NaN rather than reading it as outside the support", {
    # Every comparison against NaN is FALSE, so without explicit handling NaN
    # would silently become a density of zero
    x <- nv_array(c(NaN, 0.5))
    expect_equal(as.vector(nv_dunif(x)), c(NaN, 1))
    expect_equal(as.vector(nv_dunif(x, log = TRUE)), c(NaN, 0))
  })

  it("matches base R dunif() for degenerate, reversed, infinite and NaN limits", {
    # base R's dunif() is NaN whenever `max <= min` (so a degenerate interval is
    # NaN, unlike punif()/qunif()) but has no finiteness test, so an unbounded
    # interval has density zero rather than NaN
    x <- c(-Inf, -1, 0, 0.5, 1, 2, Inf, NaN)
    got <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(c(FALSE, TRUE), function(lg) {
        as.vector(nv_dunif(as_f64(x), min = as_f64_scalar(l[[1L]]), max = as_f64_scalar(l[[2L]]), log = lg))
      })
    }))
    want <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(c(FALSE, TRUE), function(lg) {
        suppressWarnings(dunif(x, l[[1L]], l[[2L]], log = lg))
      })
    }))
    expect_equal(got, want)
  })

  it("non-scalar min/max works", {
    x <- c(0.5, 0.5, 0.5)
    min <- c(0, -1, 0.6)
    max <- c(1, 3, 2)
    expect_equal(
      as.vector(nv_dunif(
        nv_array(x),
        min = nv_array(min),
        max = nv_array(max)
      )),
      dunif(x, min = min, max = max),
      tolerance = 1e-6
    )
  })

  it("converts min/max to the dtype of x", {
    out <- nv_dunif(nv_array(c(0, 1), dtype = "f32"), min = 0L, max = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_punif", {
  it("matches base R punif() with default min/max", {
    q <- c(-0.5, 0, 0.25, 0.75, 1, 1.5)
    expect_equal(
      as.vector(nv_punif(nv_array(q))),
      punif(q),
      tolerance = 1e-6
    )
  })

  it("matches base R punif() with custom min/max", {
    q <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_punif(nv_array(q), min = -1, max = 2)),
      punif(q, min = -1, max = 2),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R punif(..., lower.tail = FALSE)", {
    q <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_punif(nv_array(q), min = -1, max = 2, lower_tail = FALSE)),
      punif(q, min = -1, max = 2, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R punif(..., log.p = TRUE)", {
    q <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_punif(nv_array(q), min = -1, max = 2, log_p = TRUE)),
      punif(q, min = -1, max = 2, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly", {
    q <- c(-2, -1, 0, 1, 2, 3)
    expect_equal(
      as.vector(nv_punif(
        nv_array(q),
        min = -1,
        max = 2,
        lower_tail = FALSE,
        log_p = TRUE
      )),
      punif(q, min = -1, max = 2, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("saturates at 0 and 1 outside the support in both tails", {
    q <- nv_array(c(-10, 10))
    expect_equal(as.vector(nv_punif(q)), c(0, 1))
    expect_equal(as.vector(nv_punif(q, lower_tail = FALSE)), c(1, 0))
    expect_equal(as.vector(nv_punif(q, log_p = TRUE)), c(-Inf, 0))
    expect_equal(
      as.vector(nv_punif(q, lower_tail = FALSE, log_p = TRUE)),
      c(0, -Inf)
    )
  })

  it("log_p = TRUE keeps full relative accuracy where the probability is close to one", {
    # (q - min) / (max - min) rounds to within 1 ulp of 1 here, so log() of it
    # retains only ~5 digits; log1p() of the small opposite tail retains all of
    # them. base R's punif() gives -9.9997787828e-13 for the same input.
    q <- nv_array(1e12 - 1, dtype = "f64")
    min <- nv_array(0, dtype = "f64")
    max <- nv_array(1e12, dtype = "f64")
    expect_equal(
      as.vector(nv_punif(q, min = min, max = max, log_p = TRUE)),
      log1p(-1 / 1e12),
      tolerance = 1e-12
    )
    # the upper tail mirrors it
    expect_equal(
      as.vector(nv_punif(
        nv_array(1, dtype = "f64"),
        min = min,
        max = max,
        lower_tail = FALSE,
        log_p = TRUE
      )),
      log1p(-1 / 1e12),
      tolerance = 1e-12
    )
  })

  it("propagates NaN through the clamp to the support", {
    q <- nv_array(c(NaN, 0.25))
    expect_equal(as.vector(nv_punif(q)), c(NaN, 0.25))
    expect_equal(as.vector(nv_punif(q, log_p = TRUE)), c(NaN, log(0.25)))
  })

  it("gradients stay finite at an infinite q (endpoint branch doesn't poison them via nv_ifelse)", {
    # nv_ifelse() differentiates through both branches, so the untaken interior
    # (q - min) / width is evaluated even where an endpoint is selected. At an
    # infinite `q` that branch is infinite, and the reverse pass combines it as
    # 0 * Inf = NaN unless the interior is fed a clamped stand-in.
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    f <- function(q, min, max, lower_tail = TRUE, log_p = FALSE) {
      nv_sum(nv_punif(q, min, max, lower_tail = lower_tail, log_p = log_p))
    }
    grad <- jit(gradient(f, wrt = c("q", "min", "max")), static = c("lower_tail", "log_p"))
    for (k in seq_len(nrow(flags))) {
      g <- grad(
        nv_array(c(-Inf, Inf), dtype = "f64"),
        nv_array(c(-1, -1), dtype = "f64"),
        nv_array(c(2, 2), dtype = "f64"),
        lower_tail = flags$lower_tail[k],
        log_p = flags$log_p[k]
      )
      expect_equal(as.vector(g$q), c(0, 0))
      expect_equal(as.vector(g$min), c(0, 0))
      expect_equal(as.vector(g$max), c(0, 0))
    }
  })

  it("gradient is the density, either side of the log/log1p branch", {
    f <- function(q) nv_sum(nv_punif(q, log_p = TRUE))
    q <- c(0.4, 0.6, 1 - 1e-7)
    g <- as.vector(jit(gradient(f, wrt = "q"))(nv_array(q, dtype = "f64"))[[1L]])
    expect_equal(g, 1 / q, tolerance = 1e-9)

    fp <- function(q) nv_sum(nv_punif(q, min = -1, max = 2))
    gp <- as.vector(jit(gradient(fp, wrt = "q"))(
      nv_array(c(-2, 0.5, 3), dtype = "f64")
    )[[1L]])
    expect_equal(gp, c(0, 1 / 3, 0), tolerance = 1e-9)
  })

  it("matches base R punif() for degenerate, reversed, infinite and NaN limits", {
    # unlike dunif(), base R's punif() admits a degenerate interval but rejects
    # any non-finite limit
    q <- c(-Inf, -1, 0, 0.5, 1, 2, Inf, NaN)
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    got <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(k) {
        as.vector(nv_punif(
          as_f64(q),
          as_f64_scalar(l[[1L]]),
          as_f64_scalar(l[[2L]]),
          lower_tail = flags$lower_tail[k],
          log_p = flags$log_p[k]
        ))
      })
    }))
    want <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(k) {
        suppressWarnings(punif(q, l[[1L]], l[[2L]], lower.tail = flags$lower_tail[k], log.p = flags$log_p[k]))
      })
    }))
    expect_equal(got, want)
  })

  it("non-scalar min/max works", {
    q <- c(0.5, 0.5, 0.5)
    min <- c(0, -1, 0.6)
    max <- c(1, 3, 2)
    expect_equal(
      as.vector(nv_punif(
        nv_array(q),
        min = nv_array(min),
        max = nv_array(max)
      )),
      punif(q, min = min, max = max),
      tolerance = 1e-6
    )
  })

  it("converts min/max to the dtype of q", {
    out <- nv_punif(nv_array(c(0, 1), dtype = "f32"), min = 0L, max = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_qunif", {
  it("matches base R qunif() with default min/max", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(
      as.vector(nv_qunif(nv_array(p))),
      qunif(p),
      tolerance = 1e-6
    )
  })

  it("matches base R qunif() with custom min/max", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qunif(nv_array(p), min = -1, max = 2)),
      qunif(p, min = -1, max = 2),
      tolerance = 1e-6
    )
  })

  it("lower_tail = FALSE matches base R qunif(..., lower.tail = FALSE)", {
    p <- c(0.001, 0.025, 0.5, 0.975)
    expect_equal(
      as.vector(nv_qunif(nv_array(p), min = -1, max = 2, lower_tail = FALSE)),
      qunif(p, min = -1, max = 2, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R qunif(..., log.p = TRUE)", {
    lp <- c(-700, -10, -2, -0.7, -0.1, 0)
    expect_equal(
      as.vector(nv_qunif(nv_array(lp, dtype = "f64"), min = -1, max = 2, log_p = TRUE)),
      qunif(lp, min = -1, max = 2, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE and lower_tail = FALSE compose correctly", {
    lp <- c(-700, -10, -2, -0.7, -0.1, 0)
    expect_equal(
      as.vector(nv_qunif(
        nv_array(lp, dtype = "f64"),
        min = -1,
        max = 2,
        lower_tail = FALSE,
        log_p = TRUE
      )),
      qunif(lp, min = -1, max = 2, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("returns the endpoints of the interval at p = 0 and p = 1", {
    p <- nv_array(c(0, 1), dtype = "f64")
    expect_equal(as.vector(nv_qunif(p, min = -1, max = 2)), c(-1, 2))
    expect_equal(
      as.vector(nv_qunif(p, min = -1, max = 2, lower_tail = FALSE)),
      c(2, -1)
    )
    expect_equal(
      as.vector(nv_qunif(nv_array(c(-Inf, 0), dtype = "f64"), min = -1, max = 2, log_p = TRUE)),
      c(-1, 2)
    )
  })

  it("returns NaN outside [0, 1], and for positive log probabilities", {
    p <- c(-0.25, 1.25, NaN)
    expect_equal(
      as.vector(nv_qunif(nv_array(p, dtype = "f64"))),
      c(NaN, NaN, NaN)
    )
    lp <- c(0.5, Inf, NaN)
    expect_equal(
      as.vector(nv_qunif(nv_array(lp, dtype = "f64"), log_p = TRUE)),
      c(NaN, NaN, NaN)
    )
  })

  it("log_p = TRUE and lower_tail = FALSE keep full accuracy for probabilities close to one", {
    # 1 - exp(p) would cancel away all but ~7 digits of the complement here
    lp <- nv_array(-1e-10, dtype = "f64")
    expect_equal(
      as.vector(nv_qunif(lp, lower_tail = FALSE, log_p = TRUE)),
      -expm1(-1e-10),
      tolerance = 1e-12
    )
  })

  it("inverts nv_punif", {
    q <- c(-1, -0.5, 0.5, 1.5, 2)
    expect_equal(
      as.vector(nv_qunif(
        nv_punif(nv_array(q, dtype = "f64"), min = -1, max = 2),
        min = -1,
        max = 2
      )),
      q,
      tolerance = 1e-9
    )
  })

  it("gradient wrt p is the width of the interval", {
    f <- function(p) nv_sum(nv_qunif(p, min = -1, max = 2))
    g <- as.vector(jit(gradient(f, wrt = "p"))(
      nv_array(c(0.1, 0.5, 0.9), dtype = "f64")
    )[[1L]])
    expect_equal(g, c(3, 3, 3), tolerance = 1e-9)
  })

  it("gradients wrt min/max are exact", {
    p <- c(0.25, 0.75)
    f <- function(p, min, max) nv_sum(nv_qunif(p, min, max))
    g <- jit(gradient(f, wrt = c("min", "max")))(
      nv_array(p, dtype = "f64"),
      nv_array(c(0, 0), dtype = "f64"),
      nv_array(c(1, 1), dtype = "f64")
    )
    expect_equal(as.vector(g[[1L]]), 1 - p)
    expect_equal(as.vector(g[[2L]]), p)
  })

  it("gradients stay finite outside the admissible range (invalid branch doesn't poison them via nv_ifelse)", {
    # nv_ifelse() differentiates through both branches, so `min + u * (max - min)`
    # is evaluated even where the NaN branch is selected. At p = +Inf -- or at a
    # positive log probability, where exp(p) overflows -- that branch is
    # infinite, and the reverse pass combines it as 0 * Inf = NaN unless `p` is
    # replaced by an in-range stand-in first.
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    f <- function(p, min, max, lower_tail = TRUE, log_p = FALSE) {
      nv_sum(nv_qunif(p, min, max, lower_tail = lower_tail, log_p = log_p))
    }
    grad <- jit(gradient(f, wrt = c("p", "min", "max")), static = c("lower_tail", "log_p"))
    for (k in seq_len(nrow(flags))) {
      # -Inf is a legitimate log probability, so the out-of-range side differs
      p <- if (flags$log_p[k]) c(0.5, Inf) else c(-Inf, Inf)
      g <- grad(
        nv_array(p, dtype = "f64"),
        nv_array(c(-1, -1), dtype = "f64"),
        nv_array(c(2, 2), dtype = "f64"),
        lower_tail = flags$lower_tail[k],
        log_p = flags$log_p[k]
      )
      expect_equal(as.vector(g$p), c(0, 0))
      expect_equal(as.vector(g$min), c(0, 0))
      expect_equal(as.vector(g$max), c(0, 0))
    }
  })

  it("matches base R qunif() for degenerate, reversed, infinite and NaN limits", {
    # a degenerate interval collapses to `min`; a non-finite limit is NaN
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    grid <- function(log_p) if (log_p) c(-Inf, -2, -0.7, -1e-10, 0, 0.5, NaN) else c(-0.5, 0, 0.25, 1, 1.5, NaN)
    got <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(k) {
        as.vector(nv_qunif(
          as_f64(grid(flags$log_p[k])),
          as_f64_scalar(l[[1L]]),
          as_f64_scalar(l[[2L]]),
          lower_tail = flags$lower_tail[k],
          log_p = flags$log_p[k]
        ))
      })
    }))
    want <- unlist(lapply(uniform_limit_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(k) {
        suppressWarnings(qunif(
          grid(flags$log_p[k]),
          l[[1L]],
          l[[2L]],
          lower.tail = flags$lower_tail[k],
          log.p = flags$log_p[k]
        ))
      })
    }))
    expect_equal(got, want)
  })

  it("non-scalar min/max works", {
    p <- c(0.1, 0.5, 0.9)
    min <- c(0, -1, 0.6)
    max <- c(1, 3, 2)
    expect_equal(
      as.vector(nv_qunif(
        nv_array(p),
        min = nv_array(min),
        max = nv_array(max)
      )),
      qunif(p, min = min, max = max),
      tolerance = 1e-6
    )
  })

  it("converts min/max to the dtype of p", {
    out <- nv_qunif(nv_array(c(0.25, 0.75), dtype = "f32"), min = 0L, max = 1L)
    expect_equal(dtype(out), as_dtype("f32"))
  })

  it("names the operand when it is not a float", {
    # Reported as a failure to bring `min` to the operand's data type before.
    expect_error(nv_qunif(nv_array(1L)), "`p` must be a float data type")
    expect_error(nv_punif(nv_array(1L)), "`q` must be a float data type")
    expect_error(nv_dunif(nv_array(1L)), "`x` must be a float data type")
  })
})

# Size/prob configurations exercised against base R by the nv_dbinom/nv_pbinom/
# nv_qbinom agreement tests: ordinary, degenerate, invalid, infinite and NaN.
binomial_parameter_cases <- function() {
  list(
    c(10, 0.3),
    c(0, 0.3),
    c(10, 0),
    c(10, 1),
    c(10.5, 0.3),
    c(-1, 0.3),
    c(10, -0.1),
    c(10, 1.1),
    c(Inf, 0.3),
    c(Inf, 0),
    c(NaN, 0.3),
    c(10, NaN)
  )
}

describe("nv_dbinom", {
  it("matches base R dbinom()", {
    x <- c(0, 1, 2, 3, 5, 9, 10)
    expect_equal(
      as.vector(nv_dbinom(nv_array(x), size = 10, prob = 0.3)),
      dbinom(x, size = 10, prob = 0.3),
      tolerance = 1e-6
    )
  })

  it("log = TRUE matches base R dbinom(..., log = TRUE)", {
    x <- c(0, 1, 2, 3, 5, 9, 10)
    expect_equal(
      as.vector(nv_dbinom(nv_array(x), size = 10, prob = 0.3, log = TRUE)),
      dbinom(x, size = 10, prob = 0.3, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("keeps f64 accuracy across the support for small and large size", {
    for (args in list(c(20, 0.3), c(1000, 0.001), c(1e5, 0.5), c(1e5, 0.999))) {
      size <- args[[1L]]
      prob <- args[[2L]]
      x <- unique(round(c(0, 1, size * prob + c(-5, 0, 5) * sqrt(size * prob * (1 - prob)), size - 1, size)))
      x <- x[x >= 0 & x <= size]
      for (lg in c(FALSE, TRUE)) {
        expect_equal(
          as.vector(nv_dbinom(as_f64(x), size = size, prob = prob, log = lg)),
          dbinom(x, size = size, prob = prob, log = lg),
          tolerance = 1e-13
        )
      }
    }
  })

  it("stays accurate next to x = size, where log1p(-x / size) would cancel", {
    # base R's dbinom() itself is only good to ~2e-12 here; the exact value
    # comes from the closed form n p^(n - 1) (1 - p).
    x <- as_f64(99999)
    expect_equal(
      as.vector(nv_dbinom(x, size = 1e5, prob = 0.999, log = TRUE)),
      log(1e5) + 99999 * log(0.999) + log1p(-0.999),
      tolerance = 1e-14
    )
  })

  it("stays accurate for very large size, where size * prob must not be rounded", {
    # Reference values from Rmpfr at 400 bits. base R's dbinom() is only good to
    # ~1e-9 here, as it rounds size * prob before taking x - size * prob.
    expect_equal(
      as.vector(nv_dbinom(
        as_f64(c(7699989354000, 2999995653000)),
        size = 1e13,
        prob = as_f64(c(0.77, 0.3)),
        log = TRUE
      )),
      c(-47.018301202312358, -19.604563591068437),
      tolerance = 1e-14
    )
  })

  it("keeps f32 accuracy for large size", {
    # Against base R on the same f32 inputs, so only the f32 computation differs
    prob <- as.vector(nv_array(0.3, dtype = "f32"))
    x <- 3e6 + c(-4000, -1000, 0, 1000, 4000)
    expect_equal(
      as.vector(nv_dbinom(nv_array(x, dtype = "f32"), size = 1e7, prob = prob)),
      dbinom(x, size = 1e7, prob = prob),
      tolerance = 1e-5
    )
  })

  it("log = TRUE stays finite where the plain density underflows to 0", {
    x <- as_f64(0)
    expect_equal(as.vector(nv_dbinom(x, size = 1e4, prob = 0.5)), 0)
    expect_equal(
      as.vector(nv_dbinom(x, size = 1e4, prob = 0.5, log = TRUE)),
      dbinom(0, size = 1e4, prob = 0.5, log = TRUE),
      tolerance = 1e-14
    )
  })

  it("matches base R dbinom() off the support and for degenerate, invalid, infinite and NaN parameters", {
    x <- c(-Inf, -1, 0, 2.5, 3, 10, 11, Inf, NaN)
    got <- unlist(lapply(binomial_parameter_cases(), function(a) {
      lapply(c(FALSE, TRUE), function(lg) {
        as.vector(nv_dbinom(as_f64(x), size = as_f64_scalar(a[[1L]]), prob = as_f64_scalar(a[[2L]]), log = lg))
      })
    }))
    want <- unlist(lapply(binomial_parameter_cases(), function(a) {
      lapply(c(FALSE, TRUE), function(lg) {
        suppressWarnings(dbinom(x, a[[1L]], a[[2L]], log = lg))
      })
    }))
    expect_equal(got, want)
  })

  it("non-scalar size/prob works", {
    x <- c(0, 3, 7)
    size <- c(5, 10, 20)
    prob <- c(0.1, 0.5, 0.9)
    expect_equal(
      as.vector(nv_dbinom(
        nv_array(x),
        size = nv_array(size),
        prob = nv_array(prob)
      )),
      dbinom(x, size = size, prob = prob),
      tolerance = 1e-6
    )
  })

  it("gradient with respect to prob matches the score, including at the ends of the support", {
    # d/dp log p(x) = x / p - (n - x) / (1 - p)
    x <- c(0, 1, 3, 9, 10)
    f <- function(prob) nv_sum(nv_dbinom(as_f64(x), size = 10, prob = prob, log = TRUE))
    g <- as.vector(jit(gradient(f, wrt = "prob"))(as_f64_scalar(0.3))[[1L]])
    expect_equal(g, sum(x / 0.3 - (10 - x) / 0.7), tolerance = 1e-12)
  })

  it("gradient is right at prob of 0 and 1, where the density is a polynomial with a zero", {
    # d/dp (1 - p)^10 = -10 at p = 0 and d/dp p^10 = 10 at p = 1. Next to them,
    # d/dp 10 p (1 - p)^9 = 10 at p = 0 and d/dp 10 (1 - p) p^9 = -10 at p = 1,
    # although the density itself is zero there. Further in it is flat at zero.
    x <- c(0, 10, 1, 9, 2, 8, 5, 5)
    prob <- c(0, 1, 0, 1, 0, 1, 0, 1)
    f <- function(prob) nv_sum(nv_dbinom(as_f64(x), size = 10, prob = prob))
    g <- as.vector(jit(gradient(f, wrt = "prob"))(as_f64(prob))[[1L]])
    expect_equal(g, c(-10, 10, 10, -10, 0, 0, 0, 0))
  })

  it("gradient is right for size of 0 and 1, and unpoisoned by invalid elements", {
    # size = 0: the density is 1 at x = 0 whatever prob is. size = 1: it is
    # 1 - p at x = 0 and p at x = 1. Elements with an invalid size or prob, or
    # outside the support, have a NaN or zero density and a zero gradient, and
    # must leave the others' finite.
    x <- c(0, 0, 1, 0, 1, 5, 5, 5, 11, 3)
    size <- c(0, 1, 1, 1, 1, -1, 10.5, 10, 10, 10)
    prob <- c(0.3, 0.3, 0.3, 1, 0, 0.3, 0.3, 1.5, 0.3, 0.3)
    f <- function(prob) nv_sum(nv_dbinom(as_f64(x), size = as_f64(size), prob = prob))
    g <- as.vector(jit(gradient(f, wrt = "prob"))(as_f64(prob))[[1L]])
    expect_equal(g, c(0, -1, 1, -1, 1, 0, 0, 0, 0, dbinom(3, 10, 0.3) * (3 / 0.3 - 7 / 0.7)))
  })

  it("converts size/prob to the dtype of x", {
    out <- nv_dbinom(nv_array(c(0, 1), dtype = "f32"), size = 2L, prob = 0.5)
    expect_dtype(out, "f32")
  })
})

describe("nv_pbinom", {
  it("matches base R pbinom()", {
    q <- c(-1, 0, 1, 2, 3, 5, 9, 10, 11)
    expect_equal(
      as.vector(nv_pbinom(nv_array(q), size = 10, prob = 0.3)),
      pbinom(q, size = 10, prob = 0.3),
      tolerance = 1e-6
    )
  })

  it("lower_tail and log_p match base R in every combination", {
    q <- c(-1, 0, 1, 2, 3, 5, 9, 10, 11)
    for (lt in c(TRUE, FALSE)) {
      for (lp in c(FALSE, TRUE)) {
        expect_equal(
          as.vector(nv_pbinom(as_f64(q), size = 10, prob = 0.3, lower_tail = lt, log_p = lp)),
          pbinom(q, size = 10, prob = 0.3, lower.tail = lt, log.p = lp),
          tolerance = 1e-14
        )
      }
    }
  })

  it("keeps f64 accuracy in both tails and across both continued fraction branches", {
    # The branch switches at the mean, where the directly computed tail changes
    # sides; 8 sd out the other tail comes as a complement.
    for (args in list(c(20, 0.3), c(1000, 0.001), c(1e4, 0.5), c(1e5, 0.77))) {
      size <- args[[1L]]
      prob <- args[[2L]]
      q <- unique(floor(size * prob + c(-8, -3, -1, 0, 1, 3, 8) * sqrt(size * prob * (1 - prob))))
      q <- q[q >= 0 & q < size]
      for (lt in c(TRUE, FALSE)) {
        for (lp in c(FALSE, TRUE)) {
          expect_equal(
            as.vector(nv_pbinom(as_f64(q), size = size, prob = prob, lower_tail = lt, log_p = lp)),
            pbinom(q, size = size, prob = prob, lower.tail = lt, log.p = lp),
            tolerance = 1e-13
          )
        }
      }
    }
  })

  it("stays accurate for very large size, near the mean and in the tails", {
    # Near the mean the continued fraction would need too many iterations here;
    # the asymptotic expansion takes over
    for (size in c(1e12, 1e15)) {
      for (prob in c(0.001, 0.5)) {
        q <- floor(size * prob + c(-5, -1, 0, 1, 5) * sqrt(size * prob * (1 - prob)))
        for (lp in c(FALSE, TRUE)) {
          expect_equal(
            as.vector(nv_pbinom(as_f64(q), size = size, prob = prob, log_p = lp)),
            pbinom(q, size = size, prob = prob, log.p = lp),
            tolerance = 1e-12
          )
        }
      }
    }
  })

  it("stays accurate where size is huge but size * prob is about 1", {
    # The continued fraction's terms then span many orders of magnitude, which
    # its scaling must keep representable. Against base R on the same rounded
    # inputs, in both tails and on both scales.
    for (dt in c("f32", "f64")) {
      for (size in c(1e10, 1e20)) {
        prob <- as.vector(nv_array(1 / size, dtype = dt))
        q <- c(0, 1, 2, 5)
        for (lt in c(TRUE, FALSE)) {
          for (lp in c(FALSE, TRUE)) {
            expect_equal(
              as.vector(nv_pbinom(nv_array(q, dtype = dt), size = size, prob = prob, lower_tail = lt, log_p = lp)),
              pbinom(q, size = size, prob = prob, lower.tail = lt, log.p = lp),
              tolerance = if (dt == "f32") 1e-5 else 1e-12,
              info = paste(dt, size, lt, lp)
            )
          }
        }
      }
    }
    # P(X <= 0) = (1 - prob)^size exactly
    expect_equal(
      as.vector(nv_pbinom(as_f64(0), size = 1e50, prob = 1e-50, log_p = TRUE)),
      1e50 * log1p(-1e-50),
      tolerance = 1e-14
    )
  })

  it("stays accurate deep in the lower tail for huge size, where base R fails", {
    # Reference values from Rmpfr at 400 bits, summing the exact Binomial
    # probabilities. base R gives -Inf or positive log probabilities here.
    q <- 0:5
    expect_equal(
      as.vector(nv_pbinom(as_f64(q), size = 1e20, prob = 1e3 / 1e20, log_p = TRUE)),
      c(
        -1000.0000000000001,
        -993.09124522068487,
        -986.87563662392711,
        -981.06549213528399,
        -975.54302871710217,
        -970.24370784624102
      ),
      tolerance = 1e-14
    )
    expect_equal(
      as.vector(nv_pbinom(as_f64(q), size = 1e50, prob = 1e3 / 1e50, log_p = TRUE)),
      c(-1000, -993.09124522068487, -986.87563662392711, -981.06549213528388, -975.54302871710206, -970.24370784624102),
      tolerance = 1e-14
    )
  })

  it("agrees with base R across a batch mixing the asymptotic expansion and the continued fraction", {
    q <- c(0, 10, 299000, 299800, 300000, 300200, 301000, 999999)
    expect_equal(
      as.vector(nv_pbinom(as_f64(q), size = 1e6, prob = 0.3)),
      pbinom(q, size = 1e6, prob = 0.3),
      tolerance = 1e-12
    )
  })

  it("keeps f32 accuracy for large size", {
    # Against base R on the same f32 inputs, so only the f32 computation differs
    prob <- as.vector(nv_array(0.99, dtype = "f32"))
    q <- 9.9e6 + c(-1000, -300, 0, 300, 1000)
    for (lt in c(TRUE, FALSE)) {
      expect_equal(
        as.vector(nv_pbinom(nv_array(q, dtype = "f32"), size = 1e7, prob = prob, lower_tail = lt)),
        pbinom(q, size = 1e7, prob = prob, lower.tail = lt),
        tolerance = 1e-5
      )
    }
  })

  it("log_p = TRUE stays finite where the probability underflows", {
    q <- as_f64(c(10, 9990))
    expect_equal(
      as.vector(nv_pbinom(q, size = 1e4, prob = 0.5, log_p = TRUE))[[1L]],
      pbinom(10, size = 1e4, prob = 0.5, log.p = TRUE),
      tolerance = 1e-14
    )
    expect_equal(
      as.vector(nv_pbinom(q, size = 1e4, prob = 0.5, lower_tail = FALSE, log_p = TRUE))[[2L]],
      pbinom(9990, size = 1e4, prob = 0.5, lower.tail = FALSE, log.p = TRUE),
      tolerance = 1e-14
    )
  })

  it("rounds q down, with base R's fuzz just below a whole number", {
    q <- as_f64(c(2.5, 3 - 1e-9, 3 - 1e-6))
    expect_equal(
      as.vector(nv_pbinom(q, size = 10, prob = 0.3)),
      pbinom(c(2.5, 3 - 1e-9, 3 - 1e-6), size = 10, prob = 0.3),
      tolerance = 1e-14
    )
  })

  it("matches base R pbinom() off the support and for degenerate, invalid, infinite and NaN parameters", {
    q <- c(-Inf, -1, 0, 2.5, 3, 10, 11, Inf, NaN)
    combos <- expand.grid(lt = c(TRUE, FALSE), lp = c(FALSE, TRUE))
    got <- unlist(lapply(binomial_parameter_cases(), function(a) {
      Map(
        function(lt, lp) {
          as.vector(nv_pbinom(
            as_f64(q),
            size = as_f64_scalar(a[[1L]]),
            prob = as_f64_scalar(a[[2L]]),
            lower_tail = lt,
            log_p = lp
          ))
        },
        combos$lt,
        combos$lp
      )
    }))
    want <- unlist(lapply(binomial_parameter_cases(), function(a) {
      Map(
        function(lt, lp) {
          suppressWarnings(pbinom(q, a[[1L]], a[[2L]], lower.tail = lt, log.p = lp))
        },
        combos$lt,
        combos$lp
      )
    }))
    expect_equal(got, want, tolerance = 1e-14)
  })

  it("non-scalar size/prob works", {
    q <- c(0, 3, 7)
    size <- c(5, 10, 20)
    prob <- c(0.1, 0.5, 0.9)
    expect_equal(
      as.vector(nv_pbinom(
        nv_array(q),
        size = nv_array(size),
        prob = nv_array(prob)
      )),
      pbinom(q, size = size, prob = prob),
      tolerance = 1e-6
    )
  })

  it("converts size/prob to the dtype of q", {
    out <- nv_pbinom(nv_array(c(0, 1), dtype = "f32"), size = 2L, prob = 0.5)
    expect_dtype(out, "f32")
  })
})

describe("nv_qbinom", {
  it("matches base R qbinom()", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(
      as.vector(nv_qbinom(nv_array(p), size = 10, prob = 0.3)),
      qbinom(p, size = 10, prob = 0.3)
    )
  })

  it("lower_tail and log_p match base R in every combination, for small and large size", {
    u <- c(1e-12, 0.001, 0.01, 0.2, 0.5, 0.8, 0.99, 0.999, 1 - 1e-12)
    for (args in list(c(20, 0.3), c(1000, 0.001), c(1e5, 0.5))) {
      for (lt in c(TRUE, FALSE)) {
        for (lp in c(FALSE, TRUE)) {
          p <- if (lp) log(u) else u
          expect_equal(
            as.vector(nv_qbinom(as_f64(p), size = args[[1L]], prob = args[[2L]], lower_tail = lt, log_p = lp)),
            qbinom(p, size = args[[1L]], prob = args[[2L]], lower.tail = lt, log.p = lp)
          )
        }
      }
    }
  })

  it("matches base R qbinom() for very large size", {
    u <- c(1e-10, 0.001, 0.3, 0.5, 0.7, 0.999)
    for (size in c(1e6, 1e12)) {
      expect_equal(
        as.vector(nv_qbinom(as_f64(u), size = size, prob = 0.3)),
        qbinom(u, size = size, prob = 0.3)
      )
    }
  })

  it("is NaN where an evaluation of the distribution function fails, and unaffected elsewhere", {
    # No input is known to make the distribution function fail, so one is made
    # to: for size = 20 only. The search must not read the NaN as an answer.
    # The fresh jit() traces nv_qbinom() anew, so the substitute is seen, and
    # nothing compiled with it is cached by nv_qbinom() itself.
    real_cdf <- binom_cdf
    local_mocked_bindings(binom_cdf = function(k, n, p, lower_tail, log_p, op_dtype) {
      nv_ifelse(n == 20, NaN, real_cdf(k, n, p, lower_tail, log_p, op_dtype))
    })
    p <- c(0.1, 0.5, 0.9)
    size <- c(10, 20, 30)
    out <- jit(function(p, size) nv_qbinom(p, size, 0.3))(as_f64(p), as_f64(size))
    expect_equal(as.vector(out), c(qbinom(0.1, 10, 0.3), NaN, qbinom(0.9, 30, 0.3)))
  })

  it("matches base R qbinom() where size is huge but size * prob is about 1", {
    for (dt in c("f32", "f64")) {
      for (size in c(1e10, 1e20)) {
        prob <- as.vector(nv_array(1 / size, dtype = dt))
        p <- as.vector(nv_array(c(0.1, 0.5, 0.73, 0.95), dtype = dt))
        expect_equal(
          as.vector(nv_qbinom(nv_array(p, dtype = dt), size = size, prob = prob)),
          qbinom(p, size = size, prob = prob),
          info = paste(dt, size)
        )
      }
    }
  })

  it("matches base R qbinom() at f32 for large size", {
    # On the same f32 inputs. Away from p that round onto a value of the
    # distribution function, where the coarser f32 tolerance decides.
    p <- as.vector(nv_array(c(0.001, 0.2, 0.5, 0.8), dtype = "f32"))
    prob <- as.vector(nv_array(0.3, dtype = "f32"))
    expect_equal(
      as.vector(nv_qbinom(nv_array(p, dtype = "f32"), size = 1e7, prob = prob)),
      qbinom(p, size = 1e7, prob = prob)
    )
  })

  it("maps the distribution function at each support point back to it", {
    # The search is continuous from the left, so a probability computed at a
    # support point is mapped back to that point despite its rounding.
    q <- 0:20
    for (lt in c(TRUE, FALSE)) {
      for (lp in c(FALSE, TRUE)) {
        p <- nv_pbinom(as_f64(q), size = 20, prob = 0.3, lower_tail = lt, log_p = lp)
        got <- as.vector(nv_qbinom(p, size = 20, prob = 0.3, lower_tail = lt, log_p = lp))
        # The upper tail reaches 0 at q = 20, which maps to the start of the
        # support it is attained on
        expect_equal(got, if (lt) q else c(q[-21L], 20))
      }
    }
  })

  it("matches base R qbinom() at the ends of p and for degenerate, invalid, infinite and NaN parameters", {
    p <- c(-0.1, 0, 0.3, 1, 1.1, NaN)
    lp <- c(-Inf, -1, 0, 0.1, NaN)
    combos <- expand.grid(lt = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    got <- unlist(lapply(binomial_parameter_cases(), function(a) {
      Map(
        function(lt, log_p) {
          as.vector(nv_qbinom(
            as_f64(if (log_p) lp else p),
            size = as_f64_scalar(a[[1L]]),
            prob = as_f64_scalar(a[[2L]]),
            lower_tail = lt,
            log_p = log_p
          ))
        },
        combos$lt,
        combos$log_p
      )
    }))
    want <- unlist(lapply(binomial_parameter_cases(), function(a) {
      Map(
        function(lt, log_p) {
          suppressWarnings(qbinom(if (log_p) lp else p, a[[1L]], a[[2L]], lower.tail = lt, log.p = log_p))
        },
        combos$lt,
        combos$log_p
      )
    }))
    expect_equal(got, want)
  })

  it("non-scalar size/prob works", {
    p <- c(0.1, 0.5, 0.9)
    size <- c(5, 10, 20)
    prob <- c(0.1, 0.5, 0.9)
    expect_equal(
      as.vector(nv_qbinom(
        nv_array(p),
        size = nv_array(size),
        prob = nv_array(prob)
      )),
      qbinom(p, size = size, prob = prob)
    )
  })

  it("converts size/prob to the dtype of p", {
    out <- nv_qbinom(nv_array(c(0.1, 0.9), dtype = "f32"), size = 2L, prob = 0.5)
    expect_dtype(out, "f32")
  })

  it("names the operand when it is not a float", {
    expect_error(nv_qbinom(nv_array(1L), 2, 0.5), "`p` must be a float data type")
    expect_error(nv_pbinom(nv_array(1L), 2, 0.5), "`q` must be a float data type")
    expect_error(nv_dbinom(nv_array(1L), 2, 0.5), "`x` must be a float data type")
  })
})

describe("eager/jit equivalence", {
  it("agrees for nv_dnorm(), nv_pnorm() and nv_qnorm()", {
    f64 <- function() nv_array(c(0.25, 0.75), dtype = "f64")
    expect_eager_jit_equal_grid(list(
      dnorm = function(x, v) nv_dnorm(f64(), mean = v),
      pnorm = function(x, v) nv_pnorm(f64(), sd = v),
      qnorm = function(x, v) nv_qnorm(f64(), mean = v)
    ))
  })

  it("agrees for nv_dbinom(), nv_pbinom() and nv_qbinom()", {
    f64 <- function() nv_array(c(0.25, 0.75), dtype = "f64")
    expect_eager_jit_equal_grid(list(
      dbinom = function(x, v) nv_dbinom(f64(), size = 3, prob = v),
      pbinom = function(x, v) nv_pbinom(f64(), size = v, prob = 0.5),
      qbinom = function(x, v) nv_qbinom(f64(), size = 3, prob = v)
    ))
  })
})

describe("the float category", {
  it("nv_pnorm() and nv_qnorm() still need a 32- or 64-bit float", {
    # `f16` / `bf16` are float data types, so the general float check accepts
    # them -- but these two carry one coefficient set per width, and a narrower
    # float would silently take the `f64` set.
    #
    # The `jit()` is needed: eagerly, `nv_convert()` runs at once and has to
    # materialise a `bf16` buffer, which no backend does, so the call dies with
    # "Unsupported type: bf16" before it reaches the check. Under tracing the
    # array stays abstract and the check runs.
    expect_error(
      jit(function(x) nv_pnorm(nv_convert(x, "bf16")))(nv_array(c(0.5, 0.5))),
      "must be a 32- or 64-bit float data type"
    )
    expect_error(
      jit(function(x) nv_qnorm(nv_convert(x, "bf16")))(nv_array(c(0.5, 0.5))),
      "must be a 32- or 64-bit float data type"
    )
  })

  it("nv_dbinom(), nv_pbinom() and nv_qbinom() still need a 32- or 64-bit float", {
    # As above: they carry one series length or tolerance per width.
    for (f in list(nv_dbinom, nv_pbinom, nv_qbinom)) {
      expect_error(
        jit(function(x) f(nv_convert(x, "bf16"), 2, 0.5))(nv_array(c(0.5, 0.5))),
        "must be a 32- or 64-bit float data type"
      )
    }
  })
})
