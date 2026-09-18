## ---------------------------------------------------------------------------
## Bit-level helpers shared by the sweep engine.
##
## Everything here is about moving between an IEEE-754 value and the integer
## bit pattern that encodes it. The pattern is the ground truth: it survives
## transport exactly, it is what a regression test pins, and it is what makes
## "is this the same failing input as last month" a string comparison rather
## than a floating-point one.
## ---------------------------------------------------------------------------

LITTLE_ENDIAN <- .Platform$endian == "little"

## ---- f32 -------------------------------------------------------------------

## 32-bit pattern (as a signed R integer) -> the float32 it encodes, widened to
## a double.
f32_from_bits <- function(i) {
  readBin(writeBin(i, raw(), size = 4L), "double", size = 4L, n = length(i))
}

## double -> nearest float32, widened back to a double (round-to-nearest-even).
as_f32 <- function(x) {
  readBin(writeBin(x, raw(), size = 4L), "double", size = 4L, n = length(x))
}

## ---- f64 -------------------------------------------------------------------

## (high word, low word) -> the float64 those 64 bits encode. Both words are
## signed R integers holding raw patterns.
f64_from_words <- function(hi, lo) {
  readBin(
    writeBin(as.vector(if (LITTLE_ENDIAN) rbind(lo, hi) else rbind(hi, lo)), raw(), size = 4L),
    "double",
    size = 8L,
    n = length(hi)
  )
}

## n uniform 32-bit words, as R integers holding the raw bit patterns.
## as.integer(-2^31) overflows to NA_integer_, whose own bit pattern is
## 0x80000000 -- exactly the word we wanted -- so the coercion is correct.
rand_word32 <- function(n) {
  suppressWarnings(as.integer(sample.int(2^32, n, replace = TRUE) - (2^31 + 1)))
}

## ---- pattern formatting ----------------------------------------------------

## Signed R integer -> the unsigned value its bits encode, held as a double.
##
## NA_integer_ is not missingness here. R represents NA_integer_ as the bit
## pattern 0x80000000, so readBin() hands back NA for precisely the word whose
## unsigned value is 2^31 -- and without this branch every -0, and every f64
## whose high or low word happens to be 0x80000000, formatted as "0x00NA00NA".
u32 <- function(i) ifelse(is.na(i), 2^31, ifelse(i < 0, i + 2^32, i))

hex32 <- function(p) sprintf("%04X%04X", p %/% 65536, p %% 65536)

## Hex bit patterns, as strings. Strings are deliberate: nanoparquet widens
## 64-bit integers to doubles on the way back out, which would silently round
## any pattern above 2^53; a string round-trips exactly. Verified, not assumed.
bits_of <- function(x, dtype) {
  if (!length(x)) {
    return(character(0))
  }
  if (dtype == "f32") {
    w <- readBin(writeBin(as_f32(x), raw(), size = 4L), "integer", size = 4L, n = length(x))
    sprintf("0x%s", hex32(u32(w)))
  } else {
    w <- matrix(
      readBin(writeBin(x, raw(), size = 8L), "integer", size = 4L, n = 2 * length(x)),
      nrow = 2
    )
    ## sprintf rather than paste0: keeps a length-0 input at length 0
    sprintf("0x%s%s", hex32(u32(w[2, ])), hex32(u32(w[1, ])))
  }
}

## ---- ulp spacing -----------------------------------------------------------

## The distance to the next representable neighbour at |x|, which is the unit
## the ulp error metric is measured in. Subnormals share the spacing of the
## smallest normal binade, hence the floor on the exponent.
##
## At an exact power of two the spacing below x is half the spacing above it;
## this returns the spacing above. That one-binade asymmetry is immaterial for
## a diagnostic that is read on a log scale, and taking the larger of the two
## keeps the metric from overstating an error.
ulp_size <- function(x, dtype) {
  p <- if (dtype == "f32") 23L else 52L
  emin <- if (dtype == "f32") -126L else -1022L
  e <- pmax(floor(log2(abs(x))), emin)
  e[!is.finite(x) | x == 0] <- emin
  2^(e - p)
}

## ---- formatting ------------------------------------------------------------

## Compact, readable numbers: no 17-digit doubles and no e+00 on small integers.
fmt_num <- function(x) {
  if (length(x) != 1L) {
    return(vapply(x, fmt_num, ""))
  }
  ## is.na() is TRUE for NaN, so NaN must be tested first or it prints as NA --
  ## a distinction that matters a great deal in these results.
  if (is.nan(x)) {
    return("NaN")
  }
  if (is.na(x)) {
    return("NA")
  }
  if (is.infinite(x)) {
    return(if (x > 0) "Inf" else "-Inf")
  }
  if (x == 0) {
    return("0")
  }
  if (abs(x) >= 1e-3 && abs(x) < 1e5) format(signif(x, 4), trim = TRUE) else sprintf("%.2e", x)
}
