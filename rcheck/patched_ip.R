# Upstream InteractionPrograms (scriabin bce695b) with its two hard-coded
# module-discovery settings exposed, for the S7 parameter sweep's R reruns:
#   max.soft.power     soft powers above this fall back to 1 (upstream: 3;
#                      Inf = no cap); a fixed softPower bypasses it, as upstream
#   tree.cut.quantile  cutHeight = quantile(tree heights, this). Upstream
#                      hard-codes 0.5 and ignores its own tree.cut.quantile
#                      argument, whose default is 0.4; the default here is 0.5
# so the defaults reproduce upstream exactly. Built from the pinned file text
# by rcheck/upstream_fixes.R (each patch must match once; the defect-2 range
# fix is included).
#
# Usage: source("rcheck/patched_ip.R") replaces InteractionPrograms (and the
# rest of interaction_programs.R) in the global environment.
source("rcheck/upstream_fixes.R")
source_upstream_ip(list(
  c("softPower>3", "softPower>max.soft.power"),
  c("quantile(geneTree$height, 0.5)", "quantile(geneTree$height, tree.cut.quantile)"),
  c("tree.cut.quantile = 0.4,", "tree.cut.quantile = 0.5, max.soft.power = 3,")
))
