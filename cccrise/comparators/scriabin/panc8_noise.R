# Fig. 2d-e of the Scriabin paper on its Fluidigm C1 pancreas islets
# (figureR_scriabin panel A), one random 300-cell subsample per call (the
# legend: "12 different random subsamples of 300 cells"). From PR #275's
# rcheck/authors/panc8_paper_noise.R (same seeds, so the same draws); RNG of
# every step in panc8_steps.R. Panel B, on the same subsamples, is
# panc8_programs.R.
#
# - GT: the subsample's full-depth counts, SCTransform v1.
# - down1: a downsampling draw to inDrop coverage (5,828 / the C1 mean of
#   1,580,200 counts per cell, the paper's ~270-fold), raw (SCTransform) and
#   ALRA-denoised.
# - LISI (the paper's read-out): upstream GenerateCCIM of GT and of down1,
#   raw and ALRA; Seurat anchor integration (CCA, 30 dims) of each with the
#   GT CCIM; per-GT-pair LISI on the integrated data's 30 PCs.
#
# The figure's runs were made when this script also built draw 2 and ran
# InteractionPrograms on every arm (now panc8_programs.R). Neither touches
# panel A: each draw is seeded on its own beforehand, and nothing after the
# programs draws random numbers from R (panel A integrates all pairs).
#
# All 90,000 GT cell pairs of the subsample (the same pairs on both sides)
# enter integration, as the paper's Methods imply. PR #275 capped them at
# 10,000 (sampled), believing CCA on all pairs needed a dense 90,000 x 90,000
# matrix; Seurat 5 solves it matrix-free (RSpectra). The cap is not neutral:
# fewer pairs raise the fraction with LISI > 1 (panc8_lisi_checks.R).
#
#   Rscript cccrise/comparators/scriabin/panc8_noise.R <index> <out_dir>
source("cccrise/comparators/scriabin/panc8_steps.R")
args <- commandArgs(TRUE)
index <- as.integer(args[1]); out <- args[2]
file <- file.path(out, sprintf("sub%02d.rds", index))
if (file.exists(file)) quit(save = "no")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), sprintf("sub%02d", index), ..., "\n")

sub <- c1_subsample(index)
data <- arms(sub, index, draws = 1)
umis <- mean(colSums(GetAssayData(data$down1, assay = "RNA", layer = "counts")))
log("preprocessed; downsampled mean counts", round(umis))

gt <- ccim(data$gt, "GT")
pairs <- colnames(gt)
lisi_rows <- list(); lisi_values <- list()
for (arm in c("down1", "down1_alra")) {
  dn <- ccim(data[[arm]], arm)[, pairs]
  shared <- intersect(rownames(gt), rownames(dn))
  rho <- cor(as.vector(as.matrix(GetAssayData(gt, layer = "data")[shared, ])),
             as.vector(as.matrix(GetAssayData(dn, layer = "data")[shared, pairs])), method = "spearman")
  lisi <- integrate(gt, dn)
  lisi_values[[arm]] <- lisi
  lisi_rows[[arm]] <- data.frame(index = index, arm = arm, n_pairs = length(pairs),
                                 frac_lisi_gt_1 = mean(lisi > 1 + 1e-6), lisi_mean = mean(lisi),
                                 ccim_spearman = rho)
  log(arm, "LISI mean", round(mean(lisi), 3), "frac>1", round(mean(lisi > 1 + 1e-6), 3), "rho", round(rho, 3))
}
save_atomic(list(index = index, cells = sub$cells, prop = sub$prop, downsampled_mean_counts = umis,
                 lisi = do.call(rbind, lisi_rows), lisi_values = lisi_values), file)
log("done")
