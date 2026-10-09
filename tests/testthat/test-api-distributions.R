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

# Means exercised against base R by the nv_dpois/nv_ppois/nv_qpois agreement
# tests: ordinary, zero (a point mass at zero), infinite, negative and NaN.
pois_lambda_cases <- function() {
  c(0.5, 4, 0, Inf, -1, NaN)
}

# A grid of (k, lambda) pairs spanning the support of each lambda out to far in
# both tails, which reaches both series and, for lambda >= 200, the asymptotic
# expansion of nv_ppois().
pois_grid <- function() {
  lambda <- c(0.001, 0.3, 1, 2.5, 7, 20, 99.5, 210, 1000, 1e4, 3e5)
  do.call(
    rbind,
    lapply(lambda, function(l) {
      k <- floor(c(0:12, l + sqrt(l) * seq(-12, 12, by = 0.5), l * c(0.5, 0.79, 0.81, 1.2, 1.26, 2, 5)))
      data.frame(k = unique(k[k >= 0]), lambda = l)
    })
  )
}

# Element-by-element relative comparison, with NaN and infinities matched
# exactly. `expect_equal()` measures the mean relative difference over the whole
# vector, which tiny elements barely move, and compares absolutely when the
# expected values are smaller than the tolerance.
expect_equal_elementwise <- function(object, expected, tolerance = 1e-13, info = NULL) {
  close <- (abs(object - expected) <= tolerance * abs(expected)) %in% TRUE
  ok <- ifelse(is.nan(expected), is.nan(object), !is.nan(object) & (object == expected | close))
  bad <- which(!ok)
  expect(
    length(bad) == 0L,
    sprintf(
      "Elements differ at %s: got %s, expected %s.",
      paste(bad, collapse = ", "),
      paste(format(object[bad], digits = 17), collapse = ", "),
      paste(format(expected[bad], digits = 17), collapse = ", ")
    ),
    info = info
  )
  invisible(object)
}

describe("nv_dpois", {
  it("matches base R dpois() with default lambda", {
    x <- c(0, 1, 2, 3, 4, 5, 6)
    expect_equal(as.vector(nv_dpois(nv_array(x))), dpois(x, 1), tolerance = 1e-6)
  })

  it("matches base R dpois() with custom lambda", {
    x <- c(0, 1, 2, 5, 10, 20)
    expect_equal_elementwise(as.vector(nv_dpois(nv_array(x), lambda = 4.5)), dpois(x, 4.5), tolerance = 1e-6)
  })

  it("log = TRUE matches base R dpois(..., log = TRUE)", {
    x <- c(0, 1, 2, 5, 10, 20)
    expect_equal(
      as.vector(nv_dpois(nv_array(x), lambda = 4.5, log = TRUE)),
      dpois(x, 4.5, log = TRUE),
      tolerance = 1e-6
    )
  })

  it("keeps full relative accuracy far into both tails, in both f32 and f64", {
    g <- pois_grid()
    for (lg in c(FALSE, TRUE)) {
      got <- as.vector(nv_dpois(as_f64(g$k), as_f64(g$lambda), log = lg))
      expect_equal_elementwise(got, dpois(g$k, g$lambda, log = lg), tolerance = 1e-12)
    }
    # compared at the f32 lambda; away from the far tails, where the error of
    # an f32 log-probability of magnitude ~|k - lambda| is amplified by exp()
    near <- abs(g$k - g$lambda) <= 6 * sqrt(g$lambda) + 6
    k <- g$k[near]
    lambda <- as.vector(nv_array(g$lambda[near], dtype = "f32"))
    for (lg in c(FALSE, TRUE)) {
      got <- as.vector(nv_dpois(nv_array(k, dtype = "f32"), nv_array(lambda, dtype = "f32"), log = lg))
      expect_equal_elementwise(got, dpois(k, lambda, log = lg), tolerance = if (lg) 1e-5 else 2e-5)
    }
  })

  it("log = TRUE stays finite where the probability underflows to 0", {
    x <- nv_array(1000, dtype = "f64")
    expect_equal(as.vector(nv_dpois(x)), 0)
    expect_equal(as.vector(nv_dpois(x, log = TRUE)), dpois(1000, 1, log = TRUE), tolerance = 1e-14)
  })

  it("log = TRUE stays accurate where x / lambda over- or underflows", {
    x <- c(1e10, 1, 5)
    lambda <- c(1e-300, 1e308, 1e-30)
    got <- as.vector(nv_dpois(as_f64(x), as_f64(lambda), log = TRUE))
    expect_equal_elementwise(got, dpois(x, lambda, log = TRUE), tolerance = 1e-14)
  })

  it("log = TRUE stays finite at means near the top of the range", {
    for (dt in c("f32", "f64")) {
      lambda <- if (dt == "f32") c(2e38, 3e38) else c(1e308, 1.7e308)
      got <- as.vector(nv_dpois(nv_array(lambda, dtype = dt), nv_array(lambda, dtype = dt), log = TRUE))
      expect_equal(got, -0.5 * (log(2 * pi) + log(lambda)), tolerance = 1e-6)
      got <- as.vector(nv_dpois(nv_array(0.9 * lambda, dtype = dt), nv_array(lambda, dtype = dt), log = TRUE))
      expect_equal(got, -0.9 * lambda * log(0.9) - 0.1 * lambda, tolerance = 1e-4)
    }
  })

  it("is zero at negative, infinite and non-whole x, but rounds an x within a relative 1e-9 of a whole number", {
    # base R warns about a non-whole x as well. A tiny negative x rounds to
    # zero, but is tested for its sign first and so is outside the support
    x <- c(-Inf, -1, -1e-12, 0.5, 2 + 1e-6, 2 + 1e-12, 1e6 + 1e-4, Inf)
    for (lg in c(FALSE, TRUE)) {
      expect_equal(
        as.vector(nv_dpois(as_f64(x), lambda = 2, log = lg)),
        suppressWarnings(dpois(x, 2, log = lg))
      )
    }
  })

  it("propagates NaN rather than reading it as outside the support", {
    x <- nv_array(c(NaN, 0))
    expect_equal(as.vector(nv_dpois(x)), c(NaN, exp(-1)), tolerance = 1e-6)
    expect_equal(as.vector(nv_dpois(x, log = TRUE)), c(NaN, -1), tolerance = 1e-6)
  })

  it("matches base R dpois() for zero, infinite, negative and NaN lambda", {
    # lambda = 0 is the point mass at zero, and lambda = Inf has probability
    # zero everywhere
    x <- c(-Inf, -1, 0, 0.5, 1, 3, Inf, NaN)
    got <- unlist(lapply(pois_lambda_cases(), function(l) {
      lapply(c(FALSE, TRUE), function(lg) as.vector(nv_dpois(as_f64(x), as_f64_scalar(l), log = lg)))
    }))
    want <- unlist(lapply(pois_lambda_cases(), function(l) {
      lapply(c(FALSE, TRUE), function(lg) suppressWarnings(dpois(x, l, log = lg)))
    }))
    expect_equal_elementwise(got, want)
  })

  it("gradient wrt lambda matches the analytic derivative, and is zero wrt x", {
    x <- c(0, 1, 3, 10, 25, 250, 1100)
    lambda <- c(0.5, 2, 3, 7.5, 40, 240, 1000)
    f <- function(x, lambda, log = FALSE) nv_sum(nv_dpois(x, lambda, log = log))
    grad <- jit(gradient(f, wrt = c("x", "lambda")), static = "log")

    g <- grad(as_f64(x), as_f64(lambda))
    expect_equal(as.vector(g$lambda), dpois(x, lambda) * (x / lambda - 1), tolerance = 1e-12)
    expect_equal(as.vector(g$x), rep(0, length(x)))

    g <- grad(as_f64(x), as_f64(lambda), log = TRUE)
    expect_equal(as.vector(g$lambda), x / lambda - 1, tolerance = 1e-12)
    expect_equal(as.vector(g$x), rep(0, length(x)))
  })

  it("gradient wrt lambda keeps its x / lambda term at tiny means", {
    # There the density (the gradient arriving at the log in bd0()) is tiny and
    # x / lambda huge; dividing one by the other before multiplying by
    # 1 / lambda would underflow. Counts whose density is a normal float only:
    # a subnormal one is flushed to zero, gradient and all.
    f <- function(x, lambda) nv_sum(nv_dpois(x, lambda))
    grad <- jit(gradient(f, wrt = "lambda"))
    cases <- list(
      list(dt = "f64", lambda = c(1e-300, 1e-200, 1e-150, 1e-100), x = c(1, 1, 2, 3)),
      list(dt = "f32", lambda = c(1e-30, 1e-20, 1e-12), x = c(1, 1, 3))
    )
    for (cs in cases) {
      lambda <- as.vector(nv_array(cs$lambda, dtype = cs$dt))
      for (i in seq_along(lambda)) {
        got <- as.vector(grad(nv_array(cs$x[i], dtype = cs$dt), nv_array(lambda[i], dtype = cs$dt))$lambda)
        want <- dpois(cs$x[i], lambda[i]) * (cs$x[i] / lambda[i] - 1)
        expect_equal_elementwise(
          got,
          want,
          tolerance = if (cs$dt == "f64") 1e-12 else 1e-5,
          info = paste(cs$dt, lambda[i])
        )
      }
    }
  })

  it("gradients stay finite where the probability is resolved directly (untaken branches don't poison them via nv_ifelse)", {
    x <- c(-Inf, -1, 0.5, 0, 2, Inf, 3)
    lambda <- c(2, 2, 2, 0, 0, 2, Inf)
    f <- function(x, lambda, log = FALSE) nv_sum(nv_dpois(x, lambda, log = log))
    grad <- jit(gradient(f, wrt = c("x", "lambda")), static = "log")
    for (lg in c(FALSE, TRUE)) {
      g <- grad(as_f64(x), as_f64(lambda), log = lg)
      expect_equal(as.vector(g$x), rep(0, length(x)))
      # d/dlambda exp(-lambda) is -1 at x = lambda = 0, on both scales
      expect_equal(as.vector(g$lambda), c(0, 0, 0, -1, 0, 0, 0))
    }
  })

  it("non-scalar lambda works", {
    x <- c(2, 2, 2)
    lambda <- c(0.5, 2, 6)
    expect_equal(
      as.vector(nv_dpois(nv_array(x), lambda = nv_array(lambda))),
      dpois(x, lambda),
      tolerance = 1e-6
    )
  })

  it("works on arrays with more than one axis", {
    x <- matrix(c(0, 3, 5, 12, 40, 7), nrow = 2)
    out <- as_array(nv_dpois(nv_array(x, dtype = "f64"), 6))
    expect_equal(dim(out), dim(x))
    expect_equal_elementwise(as.vector(out), as.vector(dpois(x, 6)), tolerance = 1e-14)
  })

  it("converts lambda to the dtype of x", {
    out <- nv_dpois(nv_array(c(0, 1), dtype = "f32"), lambda = 2L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_ppois", {
  it("matches base R ppois() with default lambda", {
    q <- c(-1, 0, 0.5, 1, 2, 5)
    expect_equal(as.vector(nv_ppois(nv_array(q))), ppois(q, 1), tolerance = 1e-6)
  })

  it("matches base R ppois() with custom lambda", {
    q <- c(0, 1, 2, 5, 10, 20)
    expect_equal(as.vector(nv_ppois(nv_array(q), lambda = 4.5)), ppois(q, 4.5), tolerance = 1e-6)
  })

  it("lower_tail = FALSE matches base R ppois(..., lower.tail = FALSE)", {
    q <- c(0, 1, 2, 5, 10, 20)
    expect_equal_elementwise(
      as.vector(nv_ppois(nv_array(q), lambda = 4.5, lower_tail = FALSE)),
      ppois(q, 4.5, lower.tail = FALSE),
      tolerance = 1e-6
    )
  })

  it("log_p = TRUE matches base R ppois(..., log.p = TRUE)", {
    q <- c(0, 1, 2, 5, 10, 20)
    expect_equal_elementwise(
      as.vector(nv_ppois(nv_array(q), lambda = 4.5, log_p = TRUE)),
      ppois(q, 4.5, log.p = TRUE),
      tolerance = 1e-6
    )
  })

  it("matches base R across both series and the asymptotic expansion, in all four tail/scale combinations and both f32 and f64", {
    g <- pois_grid()
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    for (i in seq_len(nrow(flags))) {
      lt <- flags$lower_tail[i]
      lp <- flags$log_p[i]
      got <- as.vector(nv_ppois(as_f64(g$k), as_f64(g$lambda), lower_tail = lt, log_p = lp))
      expect_equal_elementwise(got, ppois(g$k, g$lambda, lower.tail = lt, log.p = lp), tolerance = 1e-12)

      # compared at the f32 lambda, away from the far tails (see nv_dpois)
      near <- abs(g$k - g$lambda) <= 6 * sqrt(g$lambda) + 6
      k <- g$k[near]
      lambda <- as.vector(nv_array(g$lambda[near], dtype = "f32"))
      got <- as.vector(nv_ppois(
        nv_array(k, dtype = "f32"),
        nv_array(lambda, dtype = "f32"),
        lower_tail = lt,
        log_p = lp
      ))
      want <- ppois(k, lambda, lower.tail = lt, log.p = lp)
      # leave out what f32 holds only as a subnormal, which XLA flushes to zero
      keep <- abs(want) >= 2^-126
      expect_equal_elementwise(got[keep], want[keep], tolerance = 2e-5)
    }
  })

  it("log_p = TRUE keeps full relative accuracy for probabilities close to one", {
    q <- c(30, 2000)
    lambda <- c(5, 1000)
    got <- as.vector(nv_ppois(as_f64(q), as_f64(lambda), log_p = TRUE))
    expect_equal_elementwise(got, ppois(q, lambda, log.p = TRUE), tolerance = 1e-12)
    expect_true(all(got < 0))
  })

  it("log_p = TRUE keeps full accuracy for probabilities near 0.6, where XLA's f64 log1p() does not", {
    # log1p(x) on XLA's CPU backend is up to ~120 ulp out for x in about
    # [-0.414, -0.36] in plugins built before openxla/xla#46765; ppois(7, 7)
    # and pnorm() in the asymptotic expansion at ppois(1007, 1000) both land
    # there. base R is within 2 ulp of both.
    q <- c(7, 1007, 992)
    lambda <- c(7, 1000, 1000)
    lt <- c(TRUE, TRUE, FALSE)
    got <- vapply(
      1:3,
      function(i) {
        as.vector(nv_ppois(as_f64(rep(q[i], 4)), as_f64(rep(lambda[i], 4)), lower_tail = lt[i], log_p = TRUE))[1L]
      },
      numeric(1)
    )
    want <- vapply(1:3, function(i) ppois(q[i], lambda[i], lower.tail = lt[i], log.p = TRUE), numeric(1))
    expect_equal_elementwise(got, want, tolerance = 1e-15)
  })

  it("rounds q down, but counts a q within 1e-7 below a whole number as it", {
    q <- c(2.5, 3 - 1e-8, 3 - 1e-6)
    expect_equal(as.vector(nv_ppois(as_f64(q), 2)), ppois(q, 2), tolerance = 1e-14)
    # In f32, adding the 1e-7 to q would round up to 1 from 1 - 2^-23
    q <- c(1 - 2^-23, 1 - 2^-24)
    expect_equal(as.vector(nv_ppois(nv_array(q, dtype = "f32"), 2)), ppois(c(0, 1), 2), tolerance = 1e-6)
  })

  it("matches base R ppois() for zero, infinite, negative and NaN lambda", {
    q <- c(-Inf, -1, 0, 0.5, 1, 3, Inf, NaN)
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    got <- unlist(lapply(pois_lambda_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(i) {
        as.vector(nv_ppois(as_f64(q), as_f64_scalar(l), lower_tail = flags$lower_tail[i], log_p = flags$log_p[i]))
      })
    }))
    want <- unlist(lapply(pois_lambda_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(i) {
        suppressWarnings(ppois(q, l, lower.tail = flags$lower_tail[i], log.p = flags$log_p[i]))
      })
    }))
    expect_equal_elementwise(got, want)
  })

  it("does not depend on the other elements of the array", {
    # The series run in one loop for the whole array, until every element has
    # converged, and an element that converged early must not keep accumulating;
    # and the series and the asymptotic expansion are each skipped when no
    # element needs them, which must not change the elements that do. At
    # lambda = 300 the counts take both. Each element is compared with a batch
    # of copies of itself of the same length, which needs only its own branch:
    # XLA can compile arrays of different lengths to code that rounds
    # differently (by an ulp in f32), which is not what is tested here.
    lambda <- 300
    k <- unique(floor(seq(0, 600, length.out = 41)))
    for (dt in c("f32", "f64")) {
      together <- as.vector(nv_ppois(nv_array(k, dtype = dt), lambda))
      alone <- vapply(
        k,
        function(i) {
          as.vector(nv_ppois(nv_array(rep(i, length(k)), dtype = dt), lambda))[1L]
        },
        numeric(1)
      )
      expect_identical(together, alone)
    }
  })

  it("gradient wrt lambda is the analytic derivative, and is zero wrt q", {
    q <- c(0, 1, 3, 10, 25, 250, 1100)
    lambda <- c(0.5, 2, 3, 7.5, 40, 240, 1000)
    f <- function(q, lambda, lower_tail = TRUE, log_p = FALSE) {
      nv_sum(nv_ppois(q, lambda, lower_tail = lower_tail, log_p = log_p))
    }
    grad <- jit(gradient(f, wrt = c("q", "lambda")), static = c("lower_tail", "log_p"))
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    for (i in seq_len(nrow(flags))) {
      lt <- flags$lower_tail[i]
      lp <- flags$log_p[i]
      g <- grad(as_f64(q), as_f64(lambda), lower_tail = lt, log_p = lp)
      # d/dlambda P(X <= q) = -dpois(q), over the probability on the log scale
      want <- (if (lt) -1 else 1) * dpois(q, lambda) / (if (lp) ppois(q, lambda, lower.tail = lt) else 1)
      expect_equal(as.vector(g$lambda), want, tolerance = 1e-12)
      expect_equal(as.vector(g$q), rep(0, length(q)))
    }
  })

  it("gradients stay finite where the probability is resolved directly", {
    q <- c(-Inf, -1, 0.5, 0, Inf, 3, 3)
    lambda <- c(2, 2, 2, 0, 2, Inf, 1e300)
    f <- function(q, lambda, log_p = FALSE) nv_sum(nv_ppois(q, lambda, log_p = log_p))
    grad <- jit(gradient(f, wrt = c("q", "lambda")), static = "log_p")
    for (lp in c(FALSE, TRUE)) {
      g <- grad(as_f64(q), as_f64(lambda), log_p = lp)
      expect_equal(as.vector(g$q), rep(0, length(q)))
      expect_true(all(is.finite(as.vector(g$lambda))))
    }
  })

  it("stays finite at means near the top of the range, in both f32 and f64", {
    # XLA's reciprocal of so large a k is subnormal and flushes to zero, and
    # 2 * k overflows; base R itself fails at the f64 analogues
    for (dt in c("f32", "f64")) {
      lambda <- if (dt == "f32") c(2e38, 3e38, 1e30) else c(1e308, 1.7e308, 1e300)
      at <- function(k) as.vector(nv_ppois(nv_array(k, dtype = dt), nv_array(lambda, dtype = dt)))
      expect_equal(at(lambda), rep(0.5, 3), tolerance = 1e-4)
      expect_equal(at(0.9 * lambda), rep(0, 3))
      expect_equal(at(1.1 * pmin(lambda, if (dt == "f32") 3e38 else 1.6e308)), rep(1, 3))
    }
  })

  it("non-scalar lambda works", {
    q <- c(2, 2, 2)
    lambda <- c(0.5, 2, 6)
    expect_equal(
      as.vector(nv_ppois(nv_array(q), lambda = nv_array(lambda))),
      ppois(q, lambda),
      tolerance = 1e-6
    )
  })

  it("works on arrays with more than one axis", {
    q <- matrix(c(0, 3, 5, 12, 40, 7), nrow = 2)
    expect_equal(as_array(nv_ppois(nv_array(q, dtype = "f64"), 6)), ppois(q, 6), tolerance = 1e-14)
  })

  it("converts lambda to the dtype of q", {
    out <- nv_ppois(nv_array(c(0, 1), dtype = "f32"), lambda = 2L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_qpois", {
  it("matches base R qpois() with default lambda", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(as.vector(nv_qpois(nv_array(p))), qpois(p, 1), tolerance = 0)
  })

  it("matches base R qpois() with custom lambda", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(as.vector(nv_qpois(nv_array(p), lambda = 4.5)), qpois(p, 4.5), tolerance = 0)
  })

  it("lower_tail = FALSE matches base R qpois(..., lower.tail = FALSE)", {
    p <- c(0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999)
    expect_equal(
      as.vector(nv_qpois(nv_array(p), lambda = 4.5, lower_tail = FALSE)),
      qpois(p, 4.5, lower.tail = FALSE),
      tolerance = 0
    )
  })

  it("log_p = TRUE matches base R qpois(..., log.p = TRUE)", {
    p <- c(-700, -20, -2, -0.5, -0.01)
    expect_equal(
      as.vector(nv_qpois(as_f64(p), lambda = 4.5, log_p = TRUE)),
      qpois(p, 4.5, log.p = TRUE),
      tolerance = 0
    )
  })

  it("matches base R exactly over a wide range of lambda and p, in all four tail/scale combinations and both f32 and f64", {
    withr::local_seed(1)
    lambda <- rep(c(0.01, 0.5, 3, 17.3, 150, 210, 1000, 4096, 5e4, 1e6), each = 40)
    p <- c(runif(length(lambda) / 2), 10^-runif(length(lambda) / 2, 0, 15))
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    for (dt in c("f64", "f32")) {
      # compared at the probabilities and means as held in `dt`
      lambda_dt <- as.vector(nv_array(lambda, dtype = dt))
      for (i in seq_len(nrow(flags))) {
        lt <- flags$lower_tail[i]
        lp <- flags$log_p[i]
        p_dt <- as.vector(nv_array(if (lp) log(p) else p, dtype = dt))
        got <- as.vector(nv_qpois(
          nv_array(p_dt, dtype = dt),
          nv_array(lambda_dt, dtype = dt),
          lower_tail = lt,
          log_p = lp
        ))
        expect_equal(got, qpois(p_dt, lambda_dt, lower.tail = lt, log.p = lp), tolerance = 0)
      }
    }
  })

  it("matches base R exactly at the jumps of base R's distribution function, in all four tail/scale combinations", {
    # A p computed by base R at a whole number k must give back k. nv_ppois()
    # rounds differently from base R, by more than base R's own allowance,
    # so the comparison allows for its error too.
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    for (lambda in c(3, 25.4, 100, 1000, 1234.56789)) {
      k <- unique(floor(seq(0, lambda + 12 * sqrt(lambda) + 20, length.out = 200)))
      for (i in seq_len(nrow(flags))) {
        lt <- flags$lower_tail[i]
        lp <- flags$log_p[i]
        p <- ppois(k, lambda, lower.tail = lt, log.p = lp)
        # leave out the ends and what f64 holds only as a subnormal, which XLA
        # flushes to zero
        keep <- if (lp) p < 0 & p > -Inf & abs(p) >= 2^-1022 else p > 2^-1022 & p < 1
        got <- as.vector(nv_qpois(as_f64(p[keep]), lambda, lower_tail = lt, log_p = lp))
        expect_equal(got, qpois(p[keep], lambda, lower.tail = lt, log.p = lp), tolerance = 0)
      }
    }
  })

  it("resolves probabilities just either side of a jump, in all four tail/scale combinations", {
    # The tolerance that lets a `p` at a jump give back its own count could
    # instead swallow a `p` just above the jump and give a quantile one too
    # low. Probes a relative 1e-10 of the smaller tail either side of each jump
    # must resolve: that is far outside both base R's error in locating the
    # jump (~1e-14, checked against MPFR) and the allowance nv_qpois() makes
    # (under ~1e-11 at these means).
    delta <- 1e-10
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    for (lambda in c(3, 25.4, 100, 1234.56789, 1e4, 1e8, 1e12)) {
      k <- unique(floor(c(1:10, lambda + sqrt(lambda) * seq(-8, 8, length.out = 33))))
      k <- k[k >= 1]
      lower <- ppois(k, lambda)
      upper <- ppois(k, lambda, lower.tail = FALSE)
      small <- pmin(lower, upper)
      for (i in seq_len(nrow(flags))) {
        lt <- flags$lower_tail[i]
        lp <- flags$log_p[i]
        tail_is_small <- if (lt) lower <= upper else upper < lower
        at_jump <- if (lt) lower else upper
        for (above in c(FALSE, TRUE)) {
          # the requested tail grows above the jump for the lower tail, and
          # below it for the upper tail
          tail_up <- if (lt) above else !above
          moved <- small * (1 + ifelse(tail_is_small == tail_up, delta, -delta))
          p <- ifelse(tail_is_small, moved, 1 - moved)
          if (lp) {
            p <- ifelse(tail_is_small, log(moved), log1p(-moved))
          }
          # on the plain scale near one a move of the smaller tail can be below
          # an ulp, or within base R's fuzz of 8 ulp
          keep <- delta * small > 64 * .Machine$double.eps * at_jump
          got <- as.vector(nv_qpois(as_f64(p[keep]), lambda, lower_tail = lt, log_p = lp))
          expect_equal(got, (if (above) k + 1 else k)[keep], tolerance = 0)
        }
      }
    }
  })

  it("matches MPFR-verified quantiles that earlier versions of the comparison got wrong", {
    # Each answer y was verified at 320-bit precision against Poisson tails
    # summed in MPFR (anvl-notes/poisson/quantile): the quantile condition holds
    # at y and fails at y - 1, by margins far above the precision used. The
    # literals are hex so they are the very doubles that were checked. `was` is
    # the earlier answer, and `cause` the comparison that gave it:
    # - "fuzz": base R's fixed fuzz alone, which nv_ppois()'s own rounding
    #   overcomes at a jump, giving a quantile one too high
    # - "lambda": an allowance growing with lambda, one too low at large means
    # - "distance": an allowance growing with |y - lambda| also in the
    #   asymptotic expansion, one too low at very large means
    cases <- data.frame(
      matrix(
        ncol = 7,
        byrow = TRUE,
        list(
          100,
          FALSE,
          FALSE,
          0x1.2b1e5611c6d49p-9,
          129,
          130,
          "fuzz",
          100,
          FALSE,
          TRUE,
          -0x1.475a3e3abcfb3p-14,
          64,
          65,
          "fuzz",
          100,
          TRUE,
          FALSE,
          0x1.47f1267cd788dp-7,
          77,
          78,
          "fuzz",
          100,
          TRUE,
          TRUE,
          -0x1.2b75d820b0f07p-9,
          129,
          130,
          "fuzz",
          100,
          TRUE,
          TRUE,
          -0x1.d41bef6b1a85dp-30,
          164,
          165,
          "fuzz",
          0x1.34a4584f4c6e7p+10,
          FALSE,
          FALSE,
          0x1.234cfdec0cbd6p-40,
          1489,
          1490,
          "fuzz",
          0x1.34a4584f4c6e7p+10,
          FALSE,
          FALSE,
          0x1.0c4ae432e94efp-52,
          1530,
          1531,
          "fuzz",
          0x1.34a4584f4c6e7p+10,
          FALSE,
          TRUE,
          -0x1.cd704403c5466p-58,
          946,
          947,
          "fuzz",
          0x1.34a4584f4c6e7p+10,
          TRUE,
          FALSE,
          0x1.b65e4bce1d5a9p-883,
          236,
          237,
          "fuzz",
          0x1.34a4584f4c6e7p+10,
          TRUE,
          TRUE,
          -0x1.0c57544085651p-15,
          1377,
          1378,
          "fuzz",
          1e9,
          TRUE,
          FALSE,
          0x1.f8bea04ep-1,
          1000069335,
          1000069334,
          "lambda",
          1e9,
          TRUE,
          FALSE,
          0x1.6ac40cfp-1,
          1000017364,
          1000017363,
          "lambda",
          1e9,
          TRUE,
          FALSE,
          0x1.7344ecaep-1,
          1000018916,
          1000018915,
          "lambda",
          1e13,
          TRUE,
          FALSE,
          0x1.f723413ep-1,
          10000006681246,
          10000006681245,
          "distance",
          1e13,
          TRUE,
          FALSE,
          0x1.85be73eap-1,
          10000002245939,
          10000002245938,
          "distance",
          1e14,
          TRUE,
          FALSE,
          0x1.8929be36p-1,
          100000007319374,
          100000007319373,
          "distance",
          1e14,
          TRUE,
          FALSE,
          0x1.8d1a2a4cp-1,
          100000007573846,
          100000007573845,
          "distance"
        )
      )
    )
    names(cases) <- c("lambda", "lower_tail", "log_p", "p", "quantile", "was", "cause")
    cases[] <- lapply(cases, unlist)
    for (i in seq_len(nrow(cases))) {
      r <- cases[i, ]
      got <- as.vector(nv_qpois(as_f64(rep(r$p, 4)), r$lambda, lower_tail = r$lower_tail, log_p = r$log_p))[1L]
      expect_identical(got, r$quantile, info = sprintf("case %d (%s, was %.0f)", i, r$cause, r$was))
    }
  })

  it("matches base R for random probabilities at very large means", {
    # the allowance for nv_ppois()'s error must not grow faster than the
    # steps of the distribution function shrink
    withr::local_seed(2)
    for (lambda in c(1e10, 1e12, 1e14)) {
      p <- runif(200)
      expect_equal(as.vector(nv_qpois(as_f64(p), lambda)), qpois(p, lambda), tolerance = 0)
    }
  })

  it("stays accurate at means near the top of the range, in both f32 and f64", {
    for (dt in c("f32", "f64")) {
      lambda <- if (dt == "f32") c(2e38, 1e30) else c(1e308, 1e300)
      lambda <- as.vector(nv_array(lambda, dtype = dt))
      got <- as.vector(nv_qpois(nv_array(c(0.5, 0.5), dtype = dt), nv_array(lambda, dtype = dt)))
      # the representable count below lambda is so far below the median that
      # its distribution function is ~0, so the quantile is lambda itself
      expect_equal(got, lambda, tolerance = 0)
    }
  })

  it("finds quantiles far from the Cornish-Fisher starting value", {
    p <- c(1e-300, 1e-10, 0.5, 1 - 1e-10)
    lambda <- c(0.01, 1e12, 1e12, 1e12)
    expect_equal(as.vector(nv_qpois(as_f64(p), as_f64(lambda))), qpois(p, lambda), tolerance = 0)
  })

  it("returns 0 and Inf at the ends of [0, 1], and NaN outside", {
    p <- c(-0.1, 0, 1, 1.1, NaN)
    expect_equal(as.vector(nv_qpois(as_f64(p), 3)), c(NaN, 0, Inf, NaN, NaN), tolerance = 0)
    expect_equal(as.vector(nv_qpois(as_f64(p), 3, lower_tail = FALSE)), c(NaN, Inf, 0, NaN, NaN), tolerance = 0)
    lp <- c(-Inf, 0, 0.1)
    expect_equal(as.vector(nv_qpois(as_f64(lp), 3, log_p = TRUE)), c(0, Inf, NaN), tolerance = 0)
  })

  it("matches base R qpois() for zero, infinite, negative and NaN lambda", {
    flags <- expand.grid(lower_tail = c(TRUE, FALSE), log_p = c(FALSE, TRUE))
    p_cases <- list(c(-0.1, 0, 0.3, 0.5, 1, 1.1, NaN), c(-Inf, -1, -0.1, 0, 0.5, NaN))
    got <- unlist(lapply(pois_lambda_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(i) {
        p <- p_cases[[flags$log_p[i] + 1L]]
        as.vector(nv_qpois(as_f64(p), as_f64_scalar(l), lower_tail = flags$lower_tail[i], log_p = flags$log_p[i]))
      })
    }))
    want <- unlist(lapply(pois_lambda_cases(), function(l) {
      lapply(seq_len(nrow(flags)), function(i) {
        p <- p_cases[[flags$log_p[i] + 1L]]
        suppressWarnings(qpois(p, l, lower.tail = flags$lower_tail[i], log.p = flags$log_p[i]))
      })
    }))
    expect_identical(got, want)
  })

  it("inverts nv_ppois", {
    # in f32 only while the steps of the distribution function are well above
    # base R's fuzz of 8 ulps on p, as base R itself needs in f64
    k <- 0:40
    for (dt in c("f64", "f32")) {
      p <- nv_ppois(nv_array(k, dtype = dt), 20)
      expect_equal(as.vector(nv_qpois(p, 20)), k, tolerance = 0)
      p <- nv_ppois(nv_array(k, dtype = dt), 20, lower_tail = FALSE, log_p = TRUE)
      expect_equal(as.vector(nv_qpois(p, 20, lower_tail = FALSE, log_p = TRUE)), k, tolerance = 0)
    }
  })

  it("has a zero gradient", {
    f <- function(p, lambda) nv_sum(nv_qpois(p, lambda))
    g <- jit(gradient(f, wrt = c("p", "lambda")))(as_f64(c(0.1, 0.5, 0.9)), as_f64(c(1, 5, 50)))
    expect_equal(as.vector(g$p), c(0, 0, 0))
    expect_equal(as.vector(g$lambda), c(0, 0, 0))
  })

  it("non-scalar lambda works", {
    p <- c(0.5, 0.5, 0.5)
    lambda <- c(0.5, 2, 60)
    expect_equal(as.vector(nv_qpois(nv_array(p), lambda = nv_array(lambda))), qpois(p, lambda), tolerance = 0)
  })

  it("works on arrays with more than one axis", {
    p <- matrix(c(0.01, 0.2, 0.4, 0.6, 0.8, 0.99), nrow = 2)
    expect_equal(as_array(nv_qpois(nv_array(p, dtype = "f64"), 6)), qpois(p, 6), tolerance = 0)
  })

  it("converts lambda to the dtype of p", {
    out <- nv_qpois(nv_array(c(0.25, 0.75), dtype = "f32"), lambda = 2L)
    expect_equal(dtype(out), as_dtype("f32"))
  })
})

describe("nv_dpois(), nv_ppois() and nv_qpois()", {
  it("broadcast a scalar operand against a non-scalar lambda", {
    lambda <- c(1, 2, 3)
    expect_equal(as.vector(nv_dpois(as_f64_scalar(2), as_f64(lambda))), dpois(2, lambda), tolerance = 1e-14)
    expect_equal(as.vector(nv_ppois(as_f64_scalar(2), as_f64(lambda))), ppois(2, lambda), tolerance = 1e-14)
    expect_equal(as.vector(nv_qpois(as_f64_scalar(0.5), as_f64(lambda))), qpois(0.5, lambda), tolerance = 0)
  })

  it("give a scalar lambda the sum of the per-element derivatives, masked elements included", {
    # A scalar lambda is kept scalar rather than broadcast, so its gradient is
    # the reduction of every element's derivative, including those resolved
    # directly (outside the support, or infinite), which must contribute zero
    x <- c(-Inf, -1, 0.5, 0, 1, 3, 10, 990, 1000, 1010, Inf)
    k <- ifelse(is.finite(x) & x >= 0, floor(x), NA)
    in_support <- !is.na(k) & x == k
    fd <- function(x, lambda, log = FALSE) nv_sum(nv_dpois(x, lambda, log = log))
    gd <- jit(gradient(fd, wrt = "lambda"), static = "log")
    fp <- function(q, lambda, lower_tail = TRUE, log_p = FALSE) {
      nv_sum(nv_ppois(q, lambda, lower_tail = lower_tail, log_p = log_p))
    }
    gp <- jit(gradient(fp, wrt = "lambda"), static = c("lower_tail", "log_p"))
    # The error of a sum is relative to the magnitudes summed, not to the sum,
    # which can cancel (to ~2e-16 for the density at the tiny mean), where
    # expect_equal() would fall back to an absolute comparison
    expect_sum <- function(got, terms, tol, info) {
      err <- abs(got - sum(terms)) / sum(abs(terms))
      expect(
        err <= tol,
        sprintf("got %.17g, expected %.17g (error %.3g of the magnitudes summed)", got, sum(terms), err),
        info = info
      )
    }
    # small and large counts take each series, those near 1000 the asymptotic
    # expansion, and a tiny mean puts most densities near the bottom of the range
    for (lambda in c(3, 1000, 1e-300)) {
      xs <- x[in_support]
      for (lg in c(FALSE, TRUE)) {
        terms <- if (lg) xs / lambda - 1 else dpois(xs, lambda) * (xs / lambda - 1)
        got <- as.vector(gd(as_f64(x), as_f64_scalar(lambda), log = lg)$lambda)
        expect_sum(got, terms, 1e-12, paste("dpois", lambda, lg))
      }
      ks <- k[!is.na(k)]
      for (lt in c(TRUE, FALSE)) {
        for (lp in c(FALSE, TRUE)) {
          # d/dlambda P(X <= k) = -dpois(k), over the probability on the log
          # scale. That ratio is taken as a difference of logs, which neither
          # underflows, but which leaves the reference itself about
          # eps |log dpois| out (~1e-10 at the tiny mean)
          log_density <- dpois(ks, lambda, log = TRUE)
          each <- if (lp) exp(log_density - ppois(ks, lambda, lower.tail = lt, log.p = TRUE)) else dpois(ks, lambda)
          tol <- if (lp) max(1e-12, 4 * .Machine$double.eps * max(abs(log_density))) else 1e-12
          got <- as.vector(gp(as_f64(x), as_f64_scalar(lambda), lower_tail = lt, log_p = lp)$lambda)
          expect_sum(got, (if (lt) -1 else 1) * each, tol, paste("ppois", lambda, lt, lp))
        }
      }
    }
    # At a tiny mean d/dlambda log P(X > k) is (k + 1) / lambda to within a
    # relative lambda, a reference with no error of its own. nv_ppois() forms it
    # as exp(log p(k) - log P(X > k)), so it is itself good to about
    # eps |log p(k)| (measured: within that for each count here)
    got <- as.vector(gp(as_f64(x), as_f64_scalar(1e-300), lower_tail = FALSE, log_p = TRUE)$lambda)
    tol <- .Machine$double.eps * max(abs(dpois(k, 1e-300, log = TRUE)), na.rm = TRUE)
    expect_sum(got, (k[!is.na(k)] + 1) / 1e-300, tol, "exact, tiny mean")
    # At lambda = 0 only nv_dpois(0, lambda) has a (right) derivative; the
    # other probabilities resolved there directly contribute zero by convention
    for (lg in c(FALSE, TRUE)) {
      expect_identical(as.vector(gd(as_f64(x), as_f64_scalar(0), log = lg)$lambda), -1)
    }
    expect_identical(as.vector(gp(as_f64(x), as_f64_scalar(0))$lambda), 0)
  })

  it("name the operand when it is not a float", {
    expect_error(nv_dpois(nv_array(1L)), "`x` must be a float data type")
    expect_error(nv_ppois(nv_array(1L)), "`q` must be a float data type")
    expect_error(nv_qpois(nv_array(1L)), "`p` must be a float data type")
  })

  it("need a 32- or 64-bit float", {
    # See "the float category" below for why `jit()` is needed
    for (fn in list(nv_dpois, nv_ppois, nv_qpois)) {
      expect_error(
        jit(function(x) fn(nv_convert(x, "bf16")))(nv_array(c(0.5, 0.5))),
        "must be a 32- or 64-bit float data type"
      )
    }
  })
})

describe("nv_dpois(), nv_ppois() and nv_qpois() at the ends of the range", {
  # Found by a sweep over the smallest and largest means and counts each dtype
  # holds; the regions the accuracy grids above sample never reach them.
  at <- function(x, dt) nv_array(rep(x, 4), dtype = dt)
  first <- function(a) as.vector(a)[1L]

  it("the zero count at a tiny mean, in all four tail/scale combinations", {
    # The finite lower series has no terms at k = 0, but its steps once still
    # computed n / lambda, which overflows here, and made the zero term NaN
    for (dt in c("f64", "f32")) {
      lambda <- if (dt == "f64") 1e-307 else 2^-126
      for (lt in c(TRUE, FALSE)) {
        for (lp in c(FALSE, TRUE)) {
          got <- first(nv_ppois(at(0, dt), at(lambda, dt), lower_tail = lt, log_p = lp))
          want <- ppois(0, lambda, lower.tail = lt, log.p = lp)
          expect_equal_elementwise(got, want, tolerance = 1e-6, info = paste(dt, lt, lp))
        }
      }
      expect_identical(first(nv_qpois(at(0.5, dt), at(lambda, dt))), 0)
    }
  })

  it("the upper tail on the log scale at a tiny mean", {
    # The upper series starts at lambda / (k + 1), which underflows here; its
    # log, and with it log P(X > k), was -Inf
    cases <- list(
      list(dt = "f64", lambda = 2^-1022, k = c(1, 5, 1e10)),
      list(dt = "f64", lambda = 1e-300, k = c(1e10, 1e30)),
      list(dt = "f32", lambda = 2^-126, k = c(1, 5, 1e10)),
      list(dt = "f32", lambda = 1e-30, k = c(1e10, 1e30))
    )
    for (cs in cases) {
      lambda <- as.vector(nv_array(cs$lambda, dtype = cs$dt))
      got <- as.vector(nv_ppois(
        nv_array(cs$k, dtype = cs$dt),
        nv_array(rep(lambda, length(cs$k)), dtype = cs$dt),
        lower_tail = FALSE,
        log_p = TRUE
      ))
      want <- ppois(as.vector(nv_array(cs$k, dtype = cs$dt)), lambda, lower.tail = FALSE, log.p = TRUE)
      expect_equal_elementwise(got, want, tolerance = if (cs$dt == "f64") 1e-12 else 1e-6, info = cs$dt)
    }
    # which the quantile search relies on: every probe used to satisfy it
    expect_identical(
      first(nv_qpois(at(-1000, "f32"), at(2^-126, "f32"), lower_tail = FALSE, log_p = TRUE)),
      qpois(-1000, 2^-126, lower.tail = FALSE, log.p = TRUE)
    )
  })

  it("quantiles at the largest means stay finite", {
    # The upward gallop's first step is a few ulps, which passed straight from
    # just below the largest float to Inf. At lambda = the float below the
    # largest, the largest float is the smallest that satisfies the condition
    # (it holds a few sd above lambda, far less than an ulp there); at lambda =
    # the largest float, none may, and the largest float is returned by a
    # saturating convention, as base R does.
    for (dt in c("f64", "f32")) {
      top <- if (dt == "f32") (2 - 2^-23) * 2^127 else .Machine$double.xmax
      ulp_below <- if (dt == "f32") top * (1 - 2^-24) else top * (1 - 2^-53)
      for (lambda in c(top, ulp_below)) {
        lambda <- as.vector(nv_array(lambda, dtype = dt))
        for (lt in c(TRUE, FALSE)) {
          for (lp in c(FALSE, TRUE)) {
            # a probability whose quantile lies above lambda
            p <- if (lt) 0.9 else 0.1
            if (lp) {
              p <- log(p)
            }
            got <- first(nv_qpois(at(p, dt), at(lambda, dt), lower_tail = lt, log_p = lp))
            expect_identical(got, top, info = paste(dt, lambda == top, lt, lp))
          }
        }
      }
    }
  })

  it("the log density near the largest count, on both sides of the mean", {
    # Above the mean, x log(x / lambda) + lambda overflowed (it is now summed as
    # x log(x / lambda) + (lambda - x)); below it, neither grouping can. base R
    # itself overflows at the f64 cases; the reference is the saddle point form,
    # whose stirlerr and log(2 pi x) / 2 are negligible here. The tail beyond x
    # is then within rounding of the density.
    for (dt in c("f32", "f64")) {
      top <- if (dt == "f32") (2 - 2^-23) * 2^127 else .Machine$double.xmax
      for (above in c(TRUE, FALSE)) {
        x <- as.vector(nv_array(if (above) top else top / 2, dtype = dt))
        lambda <- as.vector(nv_array(if (above) top / 2 else top, dtype = dt))
        want <- -(x * log(x / lambda) + (lambda - x))
        info <- paste(dt, if (above) "above" else "below")
        expect_equal(first(nv_dpois(at(x, dt), at(lambda, dt), log = TRUE)), want, tolerance = 1e-6, info = info)
        expect_equal(
          first(nv_ppois(at(x, dt), at(lambda, dt), lower_tail = !above, log_p = TRUE)),
          want,
          tolerance = 1e-6,
          info = info
        )
      }
    }
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
})
