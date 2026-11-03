# Source upstream Scriabin's interaction_programs.R with text patches, each of
# which must match the pinned file (bce695b) exactly once, so a patch cannot
# silently miss or drift.
#
# Always applied: the minimal fix for upstream defect 2
# (CONCORDANCE_PLAN.md §1.4). The averaged similarity 0.5 + 0.5 * cor can land
# one rounding step outside [0, 1] (1 + 2.2e-16), and WGCNA's range check then
# stops InteractionPrograms ("some entries are not between -1 and 1").
# Entries are clamped to [0, 1] right before the network steps, so only runs
# that would otherwise error change, by at most one ulp. The port does the
# same (`scriabin.averaged_similarity`).
#
# Also applied: the fix for a WGCNA bug met on IBD (defect 6).
# pickSoftThreshold.fromSimilarity works in column blocks of 1,000; with
# exactly 1,001 LR pairs (or any n = 1 mod 1000) the last block is one
# column, which drops to a vector ("non-numeric matrix extent"). One block of
# all columns gives bit-identical results otherwise (checked at n = 1000,
# 1002).
#
# Usage: source("rcheck/upstream_fixes.R"); source_upstream_ip()
#        (`extra`: further c(from, to) patches, e.g. rcheck/patched_ip.R)
UPSTREAM_IP <- "rcheck/upstream/scriabin/R/interaction_programs.R"

swap_once <- function(text, from, to) {
  hits <- gregexpr(from, text, fixed = TRUE)[[1]]
  if (sum(hits > 0) != 1) stop("upstream patch: expected one match of: ", from)
  sub(from, to, text, fixed = TRUE)
}

source_upstream_ip <- function(extra = list(), envir = globalenv()) {
  src <- paste(readLines(UPSTREAM_IP), collapse = "\n")
  src <- swap_once(src, "  if(!is.null(threads)) {",
                   "  m_cor[which(m_cor > 1)] <- 1; m_cor[which(m_cor < 0)] <- 0\n  if(!is.null(threads)) {")
  src <- swap_once(src, "pickSoftThreshold.fromSimilarity(m_cor, RsquaredCut = r2_cutoff)",
                   "pickSoftThreshold.fromSimilarity(m_cor, RsquaredCut = r2_cutoff, blockSize = ncol(m_cor))")
  for (p in extra) src <- swap_once(src, p[1], p[2])
  eval(parse(text = src, keep.source = FALSE), envir = envir)
  invisible(NULL)
}
