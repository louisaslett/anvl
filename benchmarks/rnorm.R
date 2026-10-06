devtools::load_all("~/r-xla/anvl")
library(pjrt)

f <- function() {
  nv_rnorm(dtype = "f32", shape = c(2, 3))
}


# the global RNG state is an input and an output of the program
trace_f <- function() {
  desc <- rng_enable(local_descriptor(), "input")
  graph <- trace_fn(f, list(), desc = desc)
  rng <- rng_finish(desc)
  graph$inputs <- c(graph$inputs, list(rng$input))
  graph$outputs <- c(graph$outputs, list(rng$output))
  graph
}
graph <- trace_f()

hlo_g <- function() {
  stablehlo(graph)[[1L]]
}

hlo <- hlo_g()

cmp_g <- function() {
  pjrt::pjrt_compile(pjrt::pjrt_program(stablehlo::repr(hlo)))
}

bench::mark(
  hlo_g(),
  trace_f(),
  cmp_g(),
  check = FALSE,
  memory = FALSE
)
