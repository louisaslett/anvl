#' @include aaa.R
NULL

# Side effects: values a program needs but the user never passes, such as the
# global RNG state. Every graph being traced holds a current value per side
# effect; a primitive that uses a side effect reads that value with
# `side_effect_get()` and makes the value it leaves behind the current one with
# `side_effect_set()`. Sub-graphs thread the values they use -- loop bodies as
# extra carries, branches as extra outputs, the function `gradient()`
# differentiates by closing over them -- and a compiled program takes them as
# hidden inputs after the user's and returns them as hidden outputs after the
# user's, which pjrt reads from and writes back to a state slot per side effect.
# A graph that uses no side effect is traced and compiled as if side effects did
# not exist.
#
# The side effects are registered in `globals$side_effects`, whose order is the
# order of the hidden inputs, outputs and carries.

# Registers the side effect `name`: its value is an array of data type `dtype`
# and shape `shape`; `slot()` returns the pjrt state slot a compiled program
# reads it from and writes it back to (see `dispatcher()`'s `state`); a program
# calls `slot()` when it is compiled and keeps that slot, so a side effect is
# registered once, when the package is loaded. `label` names the value and `use`
# what a primitive does with it, for error messages ("The global RNG state",
# "drawn from").
register_side_effect <- function(name, dtype, shape, slot, label, use) {
  globals$side_effects[[name]] <- list(
    name = name,
    dtype = dtype,
    shape = shape,
    slot = slot,
    label = label,
    use = use
  )
  invisible(NULL)
}

side_effect_def <- function(name) {
  globals$side_effects[[name]] %||% cli_abort("Internal error: unknown side effect {.val {name}}.")
}

# Lets the trace `desc` use side effects in `mode`:
#   "input":   a side effect's value is a fresh input of the graph, not among
#              `desc$inputs`: whoever traced `desc` adds it where it belongs
#              and returns the value the trace leaves behind (see
#              `side_effects_finish()`). The root of a jit call and of
#              `trace_fn()`, and the bodies of `prim_while()` and
#              `prim_scan()`, take side effects this way.
#   "capture": a side effect's value is that of the enclosing trace `parent`,
#              which the graph closes over. The branches of `prim_if()` and
#              the function `gradient()` differentiates take side effects this
#              way.
# A trace with neither cannot hand a value on, so using a side effect there is
# an error.
side_effects_enable <- function(desc, mode, parent = NULL) {
  desc$side_effect_mode <- mode
  desc$side_effect_parent <- parent
  invisible(desc)
}

# The current value of the side effect `name` in the trace `desc`, a GraphBox,
# registered on first use as `desc$side_effect_mode` says.
side_effect_get <- function(desc, name) {
  entry <- desc$side_effects[[name]]
  if (!is.null(entry)) {
    return(entry$value)
  }
  def <- side_effect_def(name)
  mode <- desc$side_effect_mode
  if (identical(mode, "input")) {
    gval <- GraphValue(AbstractArray(dtype = def$dtype, shape = def$shape))
    entry <- list(value = register_gval(desc, gval), input = gval)
  } else if (identical(mode, "capture")) {
    entry <- list(value = side_effect_get(desc$side_effect_parent, name))
  } else if (!length(globals[["DESCRIPTOR_STASH"]])) {
    cli_abort("{def$label} can only be {def$use} in a function called through {.fn jit}.")
  } else {
    cli_abort(c(
      "{def$label} cannot be {def$use} here.",
      i = "It cannot be {def$use} in the condition of {.fn nv_while}, nor in the function of a reduction or a scatter."
    ))
  }
  desc$side_effects[[name]] <- entry
  entry$value
}

# Makes `value`, a GraphBox, the current value of the side effect `name` in
# `desc`, which must have read it with `side_effect_get()` first.
side_effect_set <- function(desc, name, value) {
  entry <- desc$side_effects[[name]]
  if (is.null(entry)) {
    cli_abort("Internal error: side effect {.val {name}} set before it was read.")
  }
  entry$value <- value
  desc$side_effects[[name]] <- entry
  invisible(desc)
}

# The names of the side effects the trace `desc` used, in registry order.
side_effects_used <- function(desc) {
  intersect(names(globals$side_effects), names(desc$side_effects))
}

# The current values of the side effects the trace `desc` used, as nodes named
# by side effect.
side_effects_values <- function(desc) {
  used <- side_effects_used(desc)
  setNames(lapply(used, function(name) desc$side_effects[[name]]$value$gnode), used)
}

# For a trace `desc` that took side effects as inputs: per side effect it used,
# in registry order, its `name`, its `input` and the node of the value it leaves
# behind (`output`).
side_effects_finish <- function(desc) {
  lapply(side_effects_used(desc), function(name) {
    entry <- desc$side_effects[[name]]
    list(name = name, input = entry$input, output = entry$value$gnode)
  })
}

# The names of the side effects `fx` (as `side_effects_finish()` returns them).
side_effects_names <- function(fx) {
  vapply(fx, `[[`, character(1L), "name")
}

# `graph` taking the side effect values `fx` (as `side_effects_finish()` returns
# them) as its last inputs and returning the values it leaves behind as its last
# outputs.
side_effects_add_io <- function(graph, fx) {
  graph$inputs <- c(graph$inputs, lapply(fx, `[[`, "input"))
  graph$outputs <- c(graph$outputs, lapply(fx, `[[`, "output"))
  graph
}

# The pjrt state slots of a program that takes the side effect values `fx` as
# its last inputs and outputs, or `NULL` for none.
side_effects_slots <- function(fx) {
  if (!length(fx)) {
    return(NULL)
  }
  lapply(fx, function(f) side_effect_def(f$name)$slot())
}

# Makes the side effect values `values` -- GraphBoxes, named by side effect --
# the current ones of `desc`.
side_effects_set_all <- function(desc, values) {
  for (name in names(values)) {
    side_effect_set(desc, name, values[[name]])
  }
  invisible(desc)
}

# Makes the branches of a `prim_if()` call -- their traces `descs` and graphs
# `graphs` -- return the value of each side effect either of them used after
# their outputs: the value the branch leaves behind, or the enclosing trace
# `desc`'s unchanged, which the branch then closes over. Returns the names of
# these side effects, in registry order.
side_effects_if_outputs <- function(desc, descs, graphs) {
  used <- intersect(names(globals$side_effects), unlist(lapply(descs, side_effects_used)))
  for (name in used) {
    unchanged <- side_effect_get(desc, name)$gnode
    for (i in seq_along(graphs)) {
      graph <- graphs[[i]]
      entry <- descs[[i]]$side_effects[[name]]
      if (is.null(entry)) {
        graph$constants <- c(graph$constants, list(unchanged))
        out <- unchanged
      } else {
        out <- entry$value$gnode
      }
      graph$outputs <- c(graph$outputs, list(out))
    }
  }
  used
}

# The side effect values among the outputs `out` of a call whose last
# `length(names)` outputs are the values of the side effects `names`: these
# become the current values of `desc`, and the other outputs are returned.
side_effects_take_outputs <- function(desc, names, out, at = length(out) - length(names)) {
  if (!length(names)) {
    return(out)
  }
  idx <- at + seq_along(names)
  side_effects_set_all(desc, setNames(out[idx], names))
  out[-idx]
}
