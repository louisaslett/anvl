#' @include aaa.R
NULL

#' @title Set the Global RNG Seed
#' @description
#' Seeds the global RNG state, analogous to base R's [set.seed()]. Calling it
#' is optional: until the global state is first drawn from, it is not set, and
#' the first draw derives it from base R's RNG, so a [set.seed()] before that
#' makes the draws reproducible too.
#'
#' All random draws in anvl -- the samplers [nv_runif()], [nv_rnorm()],
#' [nv_rbinom()], [nv_sample_int()] and [nv_sample()], and the primitive
#' [prim_random_bits()] they build on -- draw from one global RNG state and
#' advance it. A [jit()]-compiled function that draws from it takes the global
#' state as a hidden input and hands the advanced state back as a hidden
#' output, so every draw inside one call advances the same state, and
#' consecutive calls continue where the previous one stopped. How the draws are
#' split into compiled functions does not matter: drawing in one jitted
#' function gives the same values as drawing one after the other outside of
#' [jit()].
#'
#' The global state is also threaded through the branches of [nv_if()], the
#' bodies of [nv_while()] and [nv_scan()], and functions differentiated by
#' [gradient()], so a draw in a loop body gives a new sample on every
#' iteration. It cannot be drawn from in the condition of [nv_while()], nor in
#' the functions of a reduction or a scatter, as these
#' cannot hand an advanced state back.
#'
#' @section Seeding from base R:
#' When the global state is not set, the next draw derives it from base R's
#' RNG, consuming two of its uniform draws. This happens only once: after that,
#' the global state is advanced by anvl's draws alone, and **calling
#' [set.seed()] again does not reseed it** -- call `nv_set_seed()` for that.
#' `nv_set_seed(NULL)` unsets the global state, so that the next draw derives
#' it from base R's RNG again.
#'
#' `local_nv_seed()` and `with_nv_seed()` seed the global state for the calling
#' scope or for `code` only, like [withr::local_seed()] and
#' [withr::with_seed()] do for base R's RNG: the global state from before is
#' restored afterwards.
#'
#' The global state is seeded when R runs the call, so these functions cannot be
#' called in a function traced by [jit()], where they would seed it once, while
#' tracing, rather than on every call.
#'
#' @section Devices:
#' The global state lives on the device of the call that last drew from it,
#' and a call that runs on another device -- however that device is decided --
#' copies it there. A seed therefore gives the same sequence whatever devices
#' the draws run on.
#' @param seed (`integer(1)` | `NULL`)\cr
#'   The seed, or `NULL` to derive the global state from base R's RNG.
#' @param code (any)\cr
#'   An expression to evaluate with the seeded global state.
#' @param envir (`environment`)\cr
#'   The scope whose exit restores the global state.
#' @return `nv_set_seed()` and `local_nv_seed()` return `NULL`, invisibly;
#'   `with_nv_seed()` returns the result of evaluating `code`.
#' @family rng
#' @examplesIf pjrt::plugins_downloaded()
#' nv_set_seed(42L)
#' nv_runif(3L)
#' nv_runif(3L)
#'
#' # setting the seed again repeats the sequence
#' nv_set_seed(42L)
#' nv_runif(3L)
#'
#' # an unset state is derived from base R's RNG on the next draw
#' nv_set_seed(NULL)
#' set.seed(1)
#' nv_runif(3L)
#'
#' # draws inside a jitted function share the global state
#' f <- jit(function() nv_rnorm(2L) + nv_runif(2L))
#' f()
#'
#' # seed for one expression only
#' with_nv_seed(1L, nv_runif(2L))
#' with_nv_seed(1L, nv_runif(2L))
#' @export
nv_set_seed <- function(seed) {
  if (currently_tracing()) {
    cli_abort(c(
      "The global RNG state cannot be seeded in a function traced by {.fn jit}.",
      i = "Seed it before calling the function."
    ))
  }
  globals$rng_state <- if (!is.null(seed)) rng_state_from_seed(assert_int(seed, coerce = TRUE))
  invisible(NULL)
}

#' @rdname nv_set_seed
#' @export
local_nv_seed <- function(seed, envir = parent.frame()) {
  state <- globals$rng_state
  nv_set_seed(seed)
  withr::defer(assign("rng_state", state, envir = globals), envir = envir)
  invisible(NULL)
}

#' @rdname nv_set_seed
#' @export
with_nv_seed <- function(seed, code) {
  local_nv_seed(seed)
  code
}

# The global RNG state of the trace `desc`, a GraphBox, registered on first use
# as `desc$rng_mode` says:
#   "input":   a fresh input of the graph, `desc$rng_input`, which is not among
#              `desc$inputs`: whoever traced `desc` adds it where it belongs and
#              returns the state the trace leaves behind (see `rng_finish()`).
#              The root of a jit call and the bodies of `prim_while()` and
#              `prim_scan()` take the state this way.
#   "capture": the state of the enclosing trace `desc$rng_parent`, which the
#              graph closes over. The branches of `prim_if()` and the function
#              `gradient()` differentiates take the state this way.
# A trace with neither cannot pass a state on, so drawing there is an error.
rng_state_get <- function(desc) {
  if (!is.null(desc$rng_state)) {
    return(desc$rng_state)
  }
  mode <- desc$rng_mode
  if (identical(mode, "input")) {
    gval <- GraphValue(AbstractArray(dtype = "ui64", shape = 2L))
    desc$rng_input <- gval
    desc$rng_state <- register_gval(desc, gval)
  } else if (identical(mode, "capture")) {
    desc$rng_state <- rng_state_get(desc$rng_parent)
  } else if (!length(globals[["DESCRIPTOR_STASH"]])) {
    cli_abort("The global RNG state can only be drawn from in a function called through {.fn jit}.")
  } else {
    cli_abort(c(
      "The global RNG state cannot be drawn from here.",
      i = "It cannot be drawn from in the condition of {.fn nv_while}, nor in the function of a reduction or a scatter."
    ))
  }
  desc$rng_state
}

# Lets the trace `desc` take the global RNG state in `mode` (see
# `rng_state_get()`), from the enclosing trace `parent` for `"capture"`.
rng_enable <- function(desc, mode, parent = NULL) {
  desc$rng_mode <- mode
  desc$rng_parent <- parent
  invisible(desc)
}

# For a trace `desc` that took the global RNG state as an input: that input and
# the node of the state it leaves behind, as `list(input, output)`. `NULL` when
# the trace did not draw from it.
rng_finish <- function(desc) {
  if (is.null(desc$rng_input)) {
    return(NULL)
  }
  list(input = desc$rng_input, output = desc$rng_state$gnode)
}

# `graph` taking the global RNG state as its last input and returning the state
# it leaves behind as its last output, for `rng` as `rng_finish()` returns it;
# `graph` itself when that is `NULL`.
rng_add_io <- function(graph, rng) {
  if (!is.null(rng)) {
    graph$inputs <- c(graph$inputs, list(rng$input))
    graph$outputs <- c(graph$outputs, list(rng$output))
  }
  graph
}

# The global RNG state slot of a program compiled from a trace that drew from
# it (see `dispatcher()`'s `state`): pjrt's engine reads `globals$rng_state`
# before every run -- deriving it from base R's RNG when it is not set --
# copying it to the program's device, and writes the advanced state back after
# it.
global_rng_slot <- function() {
  list(env = globals, name = "rng_state", init = rng_state_from_r, dtype = "ui64", shape = 2L)
}

# A fresh RNG state from two uniform draws of base R's RNG: a key of 53 bits,
# 32 from the first draw and 21 from the second, and the counter `0`. A
# uniform draw of R's default generator has 32 random bits.
rng_state_from_r <- function() {
  u <- stats::runif(2L)
  key <- floor(u[[1L]] * 2^32) + floor(u[[2L]] * 2^21) * 2^32
  nv_array(c(key, 0), dtype = "ui64")
}

# The RNG state of the integer `seed`, a `ui64[2]` array on `device`: the key,
# `seed` read as an unsigned 32-bit integer, and the counter `0`.
rng_state_from_seed <- function(seed, device = NULL) {
  nv_array(c(seed %% 2^32, 0), dtype = "ui64", device = device)
}

# Makes the branches of a `prim_if()` call -- their traces `descs` and graphs
# `graphs` -- return the global RNG state after their output when one of them
# drew from it: the state it leaves behind, or the enclosing trace `desc`'s
# unchanged, which the branch then closes over. Returns whether they do.
rng_if_outputs <- function(desc, descs, graphs) {
  drew <- vapply(descs, function(d) !is.null(d$rng_state), logical(1L))
  if (!any(drew)) {
    return(FALSE)
  }
  unchanged <- desc$rng_state$gnode
  for (i in seq_along(graphs)) {
    graph <- graphs[[i]]
    out <- if (drew[[i]]) descs[[i]]$rng_state$gnode else unchanged
    if (!drew[[i]]) {
      graph$constants <- c(graph$constants, list(unchanged))
    }
    graph$outputs <- c(graph$outputs, list(out))
  }
  TRUE
}
