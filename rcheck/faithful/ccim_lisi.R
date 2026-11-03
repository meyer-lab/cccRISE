# The paper's technical-noise analysis (Wilk et al. 2023, Fig. 2d-e and
# Methods "Analysis of the impact of sparsity"), on ALAD: per sample, the
# unweighted CCIM (upstream GenerateCCIM) of the full-depth data is the ground
# truth (GT); CCIMs of UMI-downsampled data, raw (SCT) and ALRA-denoised, are
# integrated with it by Seurat's anchor integration on 30 PCs, and LISI
# (immunogenomics/lisi) of every GT cell pair measures mixing: 1 = only GT
# pairs nearby, 2 = GT and downsampled pairs evenly mixed (recapitulated).
# Also reported: Spearman of the CCIM entries, GT vs downsampled. At most
# 10,000 cell pairs (the same random pairs on both sides) enter the
# integration: anchor integration of 20-40k "cells" took 20+ min per arm.
#
#   Rscript ccim_lisi.R <gt_dir> <out.csv> <sample> <down_dir> [<down_dir> ...]
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tibble); library(pbapply)
})
pboptions(type = "none")
options(future.globals.maxSize = Inf)
`%notin%` <- Negate(`%in%`)
source("rcheck/upstream/scriabin/R/interaction_graph.R")
# GenerateCCIM still calls GetAssayData(slot = ), defunct since SeuratObject 5;
# InteractionPrograms was updated to `layer` upstream, GenerateCCIM was not
GetAssayData <- function(object, assay = NULL, slot = NULL, layer = NULL, ...)
  SeuratObject::GetAssayData(object, assay = assay, layer = if (is.null(layer)) slot else layer, ...)
args <- commandArgs(TRUE)
gt_dir <- args[1]; out_csv <- args[2]; s <- args[3]; down_dirs <- args[-(1:3)]
meta <- readRDS(file.path(gt_dir, "meta.rds"))
cells <- rownames(meta)[meta$sample == s]
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

MAX_PAIRS <- 10000
ccim <- function(file, label) {
  m <- readRDS(file)[, cells]
  obj <- CreateSeuratObject(counts = m, assay = "X")
  obj <- SetAssayData(obj, layer = "data", new.data = m)
  cc <- suppressMessages(GenerateCCIM(obj, assay = "X", slot = "data"))
  cc <- DietSeurat(cc, assays = "CCIM")
  # Seurat < 5 (the paper's) set data = counts on creation; v5 leaves it empty
  cc <- SetAssayData(cc, layer = "data", new.data = GetAssayData(cc, layer = "counts"))
  cc$dataset <- label
  cc
}

integrate <- function(gt, down) {
  objs <- lapply(list(gt, down), function(o) {
    o <- RenameCells(o, add.cell.id = o$dataset[1])
    VariableFeatures(o) <- rownames(o)
    ScaleData(o, verbose = FALSE)
  })
  feats <- intersect(rownames(objs[[1]]), rownames(objs[[2]]))
  anchors <- FindIntegrationAnchors(objs, anchor.features = feats, dims = 1:30, verbose = FALSE)
  int <- IntegrateData(anchors, dims = 1:30, verbose = FALSE)
  int <- ScaleData(int, verbose = FALSE) |> RunPCA(npcs = 30, verbose = FALSE)
  emb <- Embeddings(int, "pca")
  lisi <- lisi::compute_lisi(emb, int@meta.data, "dataset")$dataset
  lisi[int$dataset == gt$dataset[1]]
}

gts <- list(sct = ccim(file.path(gt_dir, "sct_lr.rds"), "GT"),
            alra = ccim(file.path(gt_dir, "alra_lr.rds"), "GT"))
set.seed(1)
pairs <- colnames(gts$sct)
if (length(pairs) > MAX_PAIRS) pairs <- sort(sample(pairs, MAX_PAIRS))
gts <- lapply(gts, function(g) g[, pairs])
log(s, "GT CCIM", nrow(gts$sct), "LR x", ncol(gts$sct), "cell pairs used")
# (downsampled data, ground truth): the paper's raw and ALRA arms against the
# full-depth raw CCIM, and ALRA against full-depth ALRA
arms <- list(c("sct", "sct"), c("alra", "sct"), c("alra", "alra"))
rows <- list()
for (d in down_dirs) for (arm in arms) {
  gt <- gts[[arm[2]]]
  down <- ccim(file.path(d, paste0(arm[1], "_lr.rds")), paste(basename(d), arm[1]))[, pairs]
  shared <- intersect(rownames(gt), rownames(down))
  a <- GetAssayData(gt, layer = "data")[shared, ]
  b <- GetAssayData(down, layer = "data")[shared, colnames(gt)]
  rho <- cor(as.vector(as.matrix(a)), as.vector(as.matrix(b)), method = "spearman")
  # anchor integration can fail when too few anchors survive (very sparse data)
  lisi <- tryCatch(integrate(gt, down), error = function(e) NA_real_)
  rows[[length(rows) + 1]] <- data.frame(sample = s, depth = basename(d), data = arm[1],
    ground_truth = arm[2], n_pairs = ncol(gt), lisi_mean = mean(lisi),
    lisi_median = median(lisi), frac_lisi_gt_1.5 = mean(lisi > 1.5), ccim_spearman = rho)
  log(s, basename(d), arm[1], "vs GT", arm[2], "LISI", round(mean(lisi), 3), "rho", round(rho, 3))
  write.csv(do.call(rbind, rows), paste0(out_csv, ".partial"), row.names = FALSE)
}
file.rename(paste0(out_csv, ".partial"), out_csv)
