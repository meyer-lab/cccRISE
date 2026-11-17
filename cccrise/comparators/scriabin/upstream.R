# Upstream Scriabin (BlishLab/scriabin) at bce695b, loaded from a clone of the
# repository rather than installed: the R files holding the functions these
# scripts call (GenerateCCIM, InteractionPrograms, LoadLR), sourced with the
# minimal text patches below. README.md has the setup. The clone's path is
# SCRIABIN_SOURCE (default rcheck/upstream/scriabin); its commit is checked.
#
# Patches (each must match the pinned files exactly as often as stated):
# - scriabin::mapvalues -> plyr::mapvalues: upstream's copy of plyr's
#   function, identical for the atomic vectors it is given here.
# - scriabin::LoadLR -> LoadLR, sourced from the clone's R/utils.R, its
#   OmniPath resource read from the clone's inst/lr_resources.rds (utils.R
#   reads it in three places, LoadLR's and LoadRawLR's). Together
#   with the mapvalues swap, nothing needs the scriabin package installed
#   (which would pull in CelliD, nichenetr and others never called).
# - InteractionPrograms: clamp the averaged similarity to [0, 1] (0.5 + 0.5 *
#   cor can land one ulp above 1 and stop WGCNA's range check), and one block
#   of all columns in pickSoftThreshold.fromSimilarity (WGCNA drops a
#   one-column last block to a vector). Neither changes a run that does not
#   otherwise error.
# - GetAssayData(slot = ), defunct since SeuratObject 5, is passed on as
#   layer = (a wrapper, not a patch).
SCRIABIN_COMMIT <- "bce695b"
SCRIABIN_SOURCE <- Sys.getenv("SCRIABIN_SOURCE", "rcheck/upstream/scriabin")
local({
  commit <- system2("git", c("-C", SCRIABIN_SOURCE, "rev-parse", "--short=7", "HEAD"), stdout = TRUE)
  if (!identical(commit, SCRIABIN_COMMIT))
    stop("Scriabin source ", SCRIABIN_SOURCE, " is at ", commit, ", not ", SCRIABIN_COMMIT)
})
`%notin%` <- Negate(`%in%`)

# Source upstream R/<file> after replacing each `from` with `to`; `n`: how
# many times `from` must occur
source_upstream <- function(file, patches = list(), envir = globalenv()) {
  text <- paste(readLines(file.path(SCRIABIN_SOURCE, "R", file)), collapse = "\n")
  for (p in patches) {
    found <- gregexpr(p$from, text, fixed = TRUE)[[1]]
    if (sum(found > 0) != p$n) stop("upstream patch: expected ", p$n, " of: ", p$from)
    text <- gsub(p$from, p$to, text, fixed = TRUE)
  }
  eval(parse(text = text, keep.source = FALSE), envir = envir)
}
patch <- function(from, to, n = 1) list(from = from, to = to, n = n)

source_upstream("utils.R", list(patch(
  'system.file(package = "scriabin", "lr_resources.rds")',
  deparse(file.path(normalizePath(SCRIABIN_SOURCE), "inst", "lr_resources.rds")), 3)))
source_upstream("interaction_graph.R", list(
  patch("scriabin::LoadLR(", "LoadLR(", 3),
  patch("scriabin::mapvalues(", "plyr::mapvalues(", 2)))
source_upstream("interaction_programs.R", list(
  patch("scriabin::LoadLR(", "LoadLR("),
  patch("scriabin::mapvalues(", "plyr::mapvalues("),
  patch("  if(!is.null(threads)) {",
        "  m_cor[which(m_cor > 1)] <- 1; m_cor[which(m_cor < 0)] <- 0\n  if(!is.null(threads)) {"),
  patch("pickSoftThreshold.fromSimilarity(m_cor, RsquaredCut = r2_cutoff)",
        "pickSoftThreshold.fromSimilarity(m_cor, RsquaredCut = r2_cutoff, blockSize = ncol(m_cor))")))
rm(patch)

GetAssayData <- function(object, assay = NULL, slot = NULL, layer = NULL, ...)
  SeuratObject::GetAssayData(object, assay = assay, layer = if (is.null(layer)) slot else layer, ...)
