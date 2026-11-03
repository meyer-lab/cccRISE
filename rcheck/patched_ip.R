# Upstream InteractionPrograms (scriabin bce695b) with its two hard-coded
# module-discovery settings exposed, for the S7 parameter sweep's R reruns:
#   max.soft.power     soft powers above this fall back to 1 (upstream: 3;
#                      Inf = no cap)
#   tree.cut.quantile  cutHeight = quantile(tree heights, this) (upstream
#                      hard-codes 0.5 and ignores its own tree.cut.quantile
#                      argument, whose default is 0.4)
# The defaults (3, 0.5) reproduce upstream exactly. The function is patched
# from the sourced upstream text, not copied, and each patch must match
# exactly once, so it cannot drift from the pinned source.
#
# Usage: source("rcheck/patched_ip.R") after sourcing
# rcheck/upstream/scriabin/R/interaction_programs.R; it replaces
# InteractionPrograms in the calling environment.
patch_interaction_programs <- function(fun) {
  src <- paste(deparse(fun, width.cutoff = 500L, control = "useSource"), collapse = "\n")
  swap <- function(text, from, to) {
    hits <- gregexpr(from, text, fixed = TRUE)[[1]]
    if (sum(hits > 0) != 1) stop("patched_ip.R: expected one match of: ", from)
    sub(from, to, text, fixed = TRUE)
  }
  src <- swap(src, "softPower>3", "softPower>max.soft.power")
  src <- swap(src, "quantile(geneTree$height, 0.5)", "quantile(geneTree$height, tree.cut.quantile)")
  patched <- eval(parse(text = src), envir = environment(fun))
  f <- formals(patched)
  f$tree.cut.quantile <- 0.5
  formals(patched) <- c(f, alist(max.soft.power = 3))
  patched
}
InteractionPrograms <- patch_interaction_programs(InteractionPrograms)
