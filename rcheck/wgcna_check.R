# Run the WGCNA / dynamicTreeCut / hclust primitives Scriabin's
# InteractionPrograms uses, on similarity matrices written by Python, at every
# module-discovery setting in <dir>/settings.csv (the S7 knobs that act on a
# given similarity: r2_cutoff, max_soft_power (NA = no cap), soft_power
# (NA = pick), min_module_size, cut_height_quantile).
# Usage: Rscript rcheck/wgcna_check.R <dir>
#   reads sim_<tag>.csv and settings.csv; writes r_fit_<tag>.csv and, per
#   setting k, r_tom_<tag>_<k>.csv, r_merge_<tag>_<k>.csv, r_order_<tag>_<k>.csv,
#   r_mods_<tag>_<k>.csv
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({library(WGCNA); library(flashClust); library(dynamicTreeCut)})
dir <- commandArgs(TRUE)[1]
settings <- read.csv(file.path(dir, "settings.csv"))
wt <- function(x, f) write.table(x, file.path(dir, f), sep = ",", row.names = FALSE, col.names = FALSE)
for (f in list.files(dir, pattern = "^sim_.*\\.csv$")) {
  tag <- sub("^sim_(.*)\\.csv$", "\\1", f)
  m_cor <- as.matrix(read.csv(file.path(dir, f), header = FALSE))
  dimnames(m_cor) <- list(paste0("lr", seq_len(nrow(m_cor))), paste0("lr", seq_len(nrow(m_cor))))
  for (k in seq_len(nrow(settings))) {
    s <- settings[k, ]
    sp <- pickSoftThreshold.fromSimilarity(m_cor, RsquaredCut = s$r2_cutoff, verbose = 0)
    if (k == 1) write.csv(sp$fitIndices[, 1:3], file.path(dir, paste0("r_fit_", tag, ".csv")), row.names = FALSE)
    # as upstream: the cap / fallback applies only to a picked power
    if (!is.na(s$soft_power)) {
      softPower <- s$soft_power
    } else {
      softPower <- sp$powerEstimate
      if (is.na(softPower) || (!is.na(s$max_soft_power) && softPower > s$max_soft_power)) softPower <- 1
    }
    adj <- m_cor^softPower
    tom <- TOMsimilarity(adj, TOMType = "signed", verbose = 0)
    tree <- flashClust(as.dist(1 - tom), method = "complete")
    tree2 <- hclust(as.dist(1 - tom), method = "complete")
    stopifnot(identical(tree$merge, tree2$merge), identical(tree$order, tree2$order))
    id <- paste0(tag, "_", k)
    wt(tom, paste0("r_tom_", id, ".csv"))
    wt(cbind(tree$merge, tree$height), paste0("r_merge_", id, ".csv"))
    wt(tree$order, paste0("r_order_", id, ".csv"))
    mods <- cutreeDynamic(dendro = tree, method = "tree", minClusterSize = s$min_module_size,
                          cutHeight = quantile(tree$height, s$cut_height_quantile), verbose = 0)
    colors <- labels2colors(mods)
    con <- intramodularConnectivity(adj, colors)
    write.csv(data.frame(power = softPower, mods = mods, colors = colors, kTotal = con$kTotal, kWithin = con$kWithin),
              file.path(dir, paste0("r_mods_", id, ".csv")), row.names = FALSE)
  }
}
