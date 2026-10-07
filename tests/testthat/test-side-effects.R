# A second side effect beside `rng`: an `i32` counter, read from and written
# back to `slot$count`, and `count()`, a primitive that advances it by one and
# returns it. Registered for the calling scope only. A compiled program keeps the
# slot it was compiled with, so each registration comes with its own primitive.
local_count_side_effect <- function(envir = parent.frame()) {
  slot <- new.env()
  register_side_effect(
    "count",
    dtype = "i32",
    shape = integer(),
    slot = function() {
      list(env = slot, name = "count", init = function() nv_scalar(0L, dtype = "i32"), dtype = "i32", shape = integer())
    },
    label = "The test counter",
    use = "counted"
  )
  withr::defer(unregister_count(), envir = envir)
  count <- new_primitive(
    "count",
    function() {
      desc <- current_descriptor()
      out <- graph_desc_add(self, list(n = side_effect_get(desc, "count")), infer_fn = function(n) list(n), desc = desc)
      side_effect_set(desc, "count", out[[1L]])
      out[[1L]]
    },
    register = FALSE
  )
  count[["stablehlo"]] <- function(n) {
    list(hlo_add(n, hlo_scalar(1L, dtype = "i32", func = n$func)))
  }
  list(slot = slot, count = count)
}

unregister_count <- function() {
  globals$side_effects$count <- NULL
}

describe("side effects", {
  it("are read from and written back to their slot by every call", {
    fx <- local_count_side_effect()
    slot <- fx$slot
    prim_count <- fx$count
    expect_equal(as.integer(as_array(prim_count())), 1L)
    expect_equal(as.integer(as_array(prim_count())), 2L)
    expect_equal(as.integer(as_array(slot$count)), 2L)
  })

  it("are the last inputs and outputs of a traced graph, in registry order", {
    prim_count <- local_count_side_effect()$count
    graph <- trace_fn(function(x) x + nv_runif(1L) + nv_convert(prim_count(), "f32"), list(nv_array(1)))
    expect_length(graph$inputs, 3L)
    expect_length(graph$outputs, 3L)
    expect_identical(as.character(dtype(graph$inputs[[2L]]$aval)), "ui64")
    expect_identical(as.character(dtype(graph$inputs[[3L]]$aval)), "i32")
    expect_identical(as.character(dtype(graph$outputs[[2L]]$aval)), "ui64")
    expect_identical(as.character(dtype(graph$outputs[[3L]]$aval)), "i32")
  })

  it("leave out a side effect a graph does not use", {
    prim_count <- local_count_side_effect()$count
    graph <- trace_fn(function() prim_count(), list())
    expect_length(graph$inputs, 1L)
    expect_identical(as.character(dtype(graph$inputs[[1L]]$aval)), "i32")
  })

  it("are threaded through loops and branches alongside one another", {
    fx <- local_count_side_effect()
    slot <- fx$slot
    prim_count <- fx$count
    local_nv_seed(NULL)
    f <- jit(function(pred) {
      out <- nv_while(
        list(i = 0L, x = nv_scalar(0)),
        function(i, x) i < 3L,
        function(i, x) {
          x <- x + nv_runif(integer())
          prim_count()
          list(i = i + 1L, x = x)
        }
      )
      counted <- nv_if(pred, function() prim_count(), function() nv_scalar(-1L, dtype = "i32"))
      scanned <- nv_scan(
        init = list(nv_scalar(0L, dtype = "i32")),
        body = function(carry, x) list(carry = carry, out = prim_count()),
        steps = 2L
      )$out
      list(x = out$x, counted = counted, scanned = scanned, after = nv_runif(integer()))
    })

    expected <- with_nv_seed(1L, lapply(1:4, function(i) as_array(nv_runif(integer()))))
    out <- with_nv_seed(1L, f(nv_scalar(TRUE)))
    expect_equal(as_array(out$x), expected[[1L]] + expected[[2L]] + expected[[3L]], tolerance = 1e-6)
    expect_identical(as_array(out$after), expected[[4L]])
    # three loop iterations, the true branch, two scan steps
    expect_equal(as.integer(as_array(out$counted)), 4L)
    expect_equal(as.integer(as.vector(as_array(out$scanned))), 5:6)
    expect_equal(as.integer(as_array(slot$count)), 6L)

    # the false branch leaves the counter alone
    out <- f(nv_scalar(FALSE))
    expect_equal(as.integer(as_array(out$counted)), -1L)
    expect_equal(as.integer(as_array(slot$count)), 11L)
  })

  it("are threaded through a function differentiated by gradient()", {
    fx <- local_count_side_effect()
    slot <- fx$slot
    prim_count <- fx$count
    f <- jit(function(x) {
      g <- gradient(function(x) nv_sum(x * nv_convert(prim_count(), "f32")))(x)
      list(grad = g$x, after = prim_count())
    })
    out <- f(nv_array(c(1, 1)))
    expect_equal(as.vector(as_array(out$grad)), c(1, 1))
    expect_equal(as.integer(as_array(out$after)), 2L)
  })

  it("thread two side effects that the branches of nv_if use one each", {
    fx <- local_count_side_effect()
    prim_count <- fx$count
    local_nv_seed(NULL)
    f <- jit(function(pred) {
      nv_if(
        pred,
        function() nv_convert(prim_count(), "f32"),
        function() nv_runif(integer())
      )
    })
    with_nv_seed(1L, {
      f(nv_scalar(TRUE))
      expect_equal(as.integer(as_array(fx$slot$count)), 1L)
      # the true branch drew nothing, so the next draw is the seed's first
      expect_identical(as_array(f(nv_scalar(FALSE))), with_nv_seed(1L, as_array(nv_runif(integer()))))
      expect_equal(as.integer(as_array(fx$slot$count)), 1L)
    })
  })

  it("thread two side effects through a reversed nv_scan", {
    fx <- local_count_side_effect()
    prim_count <- fx$count
    local_nv_seed(NULL)
    f <- jit(function() {
      nv_scan(
        init = list(nv_scalar(0L, dtype = "i32")),
        body = function(carry, x) {
          list(carry = list(carry[[1L]] + 1L), out = list(n = prim_count(), u = nv_runif(integer())))
        },
        steps = 3L,
        reverse = TRUE
      )
    })
    expected <- with_nv_seed(1L, vapply(1:3, function(i) as.numeric(as_array(nv_runif(integer()))), numeric(1L)))
    out <- with_nv_seed(1L, f())
    expect_equal(as.integer(as_array(out$carry[[1L]])), 3L)
    # a reversed scan stacks the first step's outputs last
    expect_equal(as.integer(as.vector(as_array(out$out$n))), 3:1)
    expect_equal(as.numeric(as.vector(as_array(out$out$u))), rev(expected), tolerance = 1e-6)
    expect_equal(as.integer(as_array(fx$slot$count)), 3L)
  })

  it("are refused by the quickr backend", {
    skip_if_no_quickr()
    local_backend("quickr")
    f <- jit(function() nv_runif(2L))
    expect_error(f(), "The global RNG state is not supported")
  })

  it("cannot be used where a graph cannot hand the value back", {
    prim_count <- local_count_side_effect()$count
    f <- jit(function() nv_while(list(i = 0L), function(i) prim_count() < 3L, function(i) list(i = i + 1L)))
    expect_error(f(), "The test counter cannot be counted here")
  })
})
