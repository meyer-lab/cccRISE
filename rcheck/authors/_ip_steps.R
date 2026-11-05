# Upstream InteractionPrograms' module step and InteractionProgramSignificance's
# per-module test (rcheck/upstream/scriabin/R/interaction_programs.R at
# bce695b), copied line for line so one LR similarity matrix (`cor_mat`, which
# no soft power or R^2 threshold changes) can be cut at several soft powers
# without rebuilding its CCIM. panc8_paper_r2_sweep.R checks modules_at()
# against InteractionPrograms(softPower = p) on every subsample.
suppressPackageStartupMessages({ library(WGCNA); library(flashClust); library(dynamicTreeCut) })

# InteractionPrograms, from `adj <- m_cor^softPower` to `names(modules) = ...`
modules_at <- function(m_cor, softPower, min.size = 5) {
  adj <- m_cor^softPower
  tom <- TOMsimilarity(adj, TOMType = "signed", verbose = 0)
  colnames(tom) <- rownames(tom) <- rownames(m_cor)
  geneTree <- flashClust(as.dist(1 - tom), method = "complete")
  dynamicMods <- cutreeDynamic(dendro = geneTree, method = "tree", minClusterSize = min.size,
                               cutHeight = quantile(geneTree$height, 0.5), verbose = 0)
  dynamicColors <- labels2colors(dynamicMods)
  module_colors <- setdiff(unique(dynamicColors), "grey")
  modules <- lapply(module_colors, function(x) colnames(m_cor)[which(dynamicColors == x)])
  names(modules) <- module_colors
  modules
}

# InteractionProgramSignificance: the fraction of n.replicate random LR sets
# of the module's size that its within-module similarity fails to exceed
# (one-sided Mann-Whitney, p > 0.05); the module is non-significant in the
# sample when that fraction is above 0.05
module_failure <- function(m_cor, mod, n.replicate = 10000) {
  random_connectivity_test <- function(m_cor = m_cor, mod = mod) {
    vars <- sample(colnames(m_cor), length(mod))
    a <- as.vector(m_cor[rownames(m_cor) %in% mod, colnames(m_cor) %in% mod])
    b <- as.vector(m_cor[vars, vars])
    wilcox.test(a, b, alternative = "greater", exact = FALSE)$p.value
  }
  p <- replicate(n.replicate, random_connectivity_test(m_cor = m_cor, mod = mod))
  sum(p > 0.05) / n.replicate
}

# WGCNA's recommended soft power at each R^2 cut (NA: none reaches it), the
# rule InteractionPrograms applies before its fallback to 1 above power 3
power_estimates <- function(m_cor, cuts) {
  vapply(cuts, function(cut) {
    utils::capture.output(sp <- suppressWarnings(suppressMessages(pickSoftThreshold.fromSimilarity(
      m_cor, RsquaredCut = cut, blockSize = ncol(m_cor), verbose = 0))))
    as.numeric(sp$powerEstimate)
  }, numeric(1))
}

# atomic, flushed checkpoint write (the .rds is the job's done-file)
save_atomic <- function(x, file) {
  partial <- paste0(file, ".partial")
  saveRDS(x, partial)
  system2("sync", c("-d", partial))
  file.rename(partial, file)
}
