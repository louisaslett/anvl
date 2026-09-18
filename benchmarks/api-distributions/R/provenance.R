## ---------------------------------------------------------------------------
## The environment fingerprint recorded with every run.
##
## A sweep result is meaningless without it. Half of this repository's existing
## .txt artifacts had to be discarded not because they were wrong but because
## nothing recorded which version of the stack produced them, and the stack had
## since changed underneath. Everything here is cheap to collect at write time
## and impossible to reconstruct afterwards.
##
## Git SHAs are read straight out of .git rather than by running `git`. That is
## deliberate twice over: it invokes no git operation, and it still works on a
## compute node where git is not installed or the checkout is a bare copy.
## ---------------------------------------------------------------------------

## Resolve HEAD by reading .git, with a graceful NA when anything is missing.
git_head <- function(repo) {
  gd <- file.path(repo, ".git")
  if (!file.exists(gd)) {
    return(NA_character_)
  }
  ## a worktree or submodule has .git as a file pointing elsewhere
  if (!dir.exists(gd)) {
    l <- tryCatch(readLines(gd, warn = FALSE), error = function(e) character(0))
    p <- sub("^gitdir: ", "", grep("^gitdir: ", l, value = TRUE)[1L])
    if (is.na(p) || !nzchar(p)) {
      return(NA_character_)
    }
    gd <- if (startsWith(p, "/")) p else file.path(repo, p)
  }
  head <- tryCatch(readLines(file.path(gd, "HEAD"), warn = FALSE)[1L], error = function(e) NA_character_)
  if (is.na(head)) {
    return(NA_character_)
  }
  if (!startsWith(head, "ref: ")) {
    return(substr(head, 1L, 40L))
  } # detached
  ref <- sub("^ref: ", "", head)
  loose <- file.path(gd, ref)
  if (file.exists(loose)) {
    return(substr(readLines(loose, warn = FALSE)[1L], 1L, 40L))
  }
  packed <- file.path(gd, "packed-refs")
  if (file.exists(packed)) {
    m <- grep(paste0(" ", ref, "$"), readLines(packed, warn = FALSE), value = TRUE)
    if (length(m)) {
      return(substr(m[1L], 1L, 40L))
    }
  }
  NA_character_
}

git_branch <- function(repo) {
  gd <- file.path(repo, ".git")
  if (!dir.exists(gd)) {
    return(NA_character_)
  }
  head <- tryCatch(readLines(file.path(gd, "HEAD"), warn = FALSE)[1L], error = function(e) NA_character_)
  if (is.na(head) || !startsWith(head, "ref: ")) {
    return(NA_character_)
  }
  sub("^ref: refs/heads/", "", head)
}

## The sibling checkouts, three levels up from here.
ECOSYSTEM <- c("anvl", "stablehlo", "pjrt", "tengen", "xlamisc")

cpu_model <- function() {
  out <- switch(
    Sys.info()[["sysname"]],
    Darwin = tryCatch(
      system2("sysctl", c("-n", "machdep.cpu.brand_string"), stdout = TRUE, stderr = FALSE),
      error = function(e) NA_character_
    ),
    Linux = tryCatch(
      sub("^model name\\s*:\\s*", "", grep("^model name", readLines("/proc/cpuinfo", warn = FALSE), value = TRUE)[1L]),
      error = function(e) NA_character_
    ),
    NA_character_
  )
  if (!length(out) || is.na(out[1L])) NA_character_ else trimws(out[1L])
}

## A short label that identifies "the same machine and backend" across runs.
## This is what a cross-platform comparison groups by, so it must stay stable
## for one machine and differ between machines that could disagree numerically.
platform_key <- function(pv) {
  sprintf("%s-%s-%s", tolower(pv$os), pv$arch, pv$device)
}

collect_provenance <- function(depth, device = NULL) {
  si <- Sys.info()
  root <- normalizePath(file.path(here(), "..", "..", ".."), mustWork = FALSE)

  pkg <- function(p) tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)
  sha <- setNames(
    vapply(ECOSYSTEM, function(p) git_head(file.path(root, p)), ""),
    paste0(ECOSYSTEM, "_sha")
  )
  ver <- setNames(vapply(ECOSYSTEM, pkg, ""), paste0(ECOSYSTEM, "_version"))

  ## anvl's configurable defaults change what a bare literal commits to, so a
  ## result is only comparable to another taken at the same settings.
  dd <- tryCatch(
    {
      d <- anvl::default_dtypes()
      c(default_float = as.character(d$float), default_int = as.character(d$int))
    },
    error = function(e) c(default_float = NA_character_, default_int = NA_character_)
  )

  if (is.null(device)) {
    device <- Sys.getenv("NV_SWEEP_DEVICE", "cpu")
  }

  pv <- list(
    run_id = sprintf(
      "%s-%s-%04X",
      format(Sys.time(), "%Y%m%dT%H%M%S"),
      gsub("[^A-Za-z0-9]", "", substr(si[["nodename"]], 1L, 12L)),
      sample.int(65535L, 1L)
    ),
    started_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    host = unname(si[["nodename"]]),
    os = unname(si[["sysname"]]),
    os_version = unname(si[["release"]]),
    arch = unname(si[["machine"]]),
    cpu = cpu_model(),
    n_cores = unname(parallel::detectCores()),
    device = device,
    r_version = paste0(R.version$major, ".", R.version$minor),
    depth = depth,
    branch = git_branch(file.path(root, "anvl")),
    sweep_seed = SWEEP_SEED
  )
  pv <- c(pv, as.list(sha), as.list(ver), as.list(dd))
  pv$platform_key <- platform_key(pv)
  pv
}

provenance_row <- function(pv) {
  as.data.frame(lapply(pv, function(v) if (is.null(v)) NA else v))
}
