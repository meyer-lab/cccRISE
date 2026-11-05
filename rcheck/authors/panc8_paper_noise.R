# The Scriabin paper's technical-noise analysis (Wilk et al. 2024, Fig. 2d-e;
# Methods "Analysis of the impact of sparsity on single-cell-resolution CCC
# analysis") on its Fluidigm C1 pancreas islets (panc8), plus the read-out the
# paper does not make: whether the same interaction programs come back.
#
# One random subsample of 300 cells per call (the legend: "12 different
# random subsamples of 300 cells"), set.seed(index):
#
# - Ground truth (GT): the full-depth counts, SCTransform (v1, regressing %
#   mitochondrial and % ribosomal reads: Methods, "unless otherwise noted").
# - Downsampled: scuttle::downsampleMatrix to inDrop coverage, 5,828 / the C1
#   mean of 1,580,200 counts per cell (x0.0037, the paper's ~270-fold), twice
#   with independent draws; each "raw" (SCTransform) and "ALRA" (RunALRA on
#   it).
# - CCIM arm (the paper's): upstream GenerateCCIM (OmniPath) of GT and of
#   draw 1, raw and ALRA; Seurat anchor integration of each with the GT CCIM
#   on 30 PCs; LISI of each GT cell pair (1 = only GT pairs nearby, 2 = evenly
#   mixed). Read out as the paper's: the fraction of GT pairs with LISI > 1,
#   and mean LISI.
# - Program arm (added): upstream InteractionPrograms on GT, GT + ALRA, and
#   both draws, raw and ALRA. At 300 cells it uses the full CCIM (no
#   subsampling, no seed), so programs differ only through the data. Modules
#   at upstream's power (R^2 0.6, 1 above 3) and at the uncapped recommended
#   power.
#
# Not stated in the paper, so chosen here: at most 10,000 GT cell pairs (the
# same pairs on both sides) enter integration, as in rcheck/faithful/
# ccim_lisi.R on ALAD; Seurat's CCA on all 90,000 pairs of a 300-cell CCIM
# needs a dense 90,000 x 90,000 matrix. The paper's kNN- and cluster-
# aggregated arms are not run.
#
#   Rscript rcheck/authors/panc8_paper_noise.R <index> <out_dir>
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(qlcMatrix); library(pbapply); library(SeuratWrappers)
})
pboptions(type = "none")
options(future.globals.maxSize = Inf)
`%notin%` <- Negate(`%in%`)
source("rcheck/upstream_fixes.R"); source_upstream_ip()
source("rcheck/upstream/scriabin/R/interaction_graph.R")
source("rcheck/authors/_ip_steps.R")
# GenerateCCIM still calls GetAssayData(slot = ), defunct since SeuratObject 5
GetAssayData <- function(object, assay = NULL, slot = NULL, layer = NULL, ...)
  SeuratObject::GetAssayData(object, assay = assay, layer = if (is.null(layer)) slot else layer, ...)
args <- commandArgs(TRUE)
index <- as.integer(args[1]); out <- args[2]
file <- file.path(out, sprintf("sub%02d.rds", index))
if (file.exists(file)) quit(save = "no")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
N_CELLS <- 300; MAX_PAIRS <- 10000; INDROP_UMI <- 5828
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), sprintf("sub%02d", index), ..., "\n")

src <- "singlecell_cache/authors/panc8_paper"
counts_all <- readRDS(file.path(src, "c1_counts.rds")); meta <- readRDS(file.path(src, "c1_meta.rds"))
prop <- INDROP_UMI / mean(colSums(counts_all))
set.seed(index)
cells <- sort(sample(colnames(counts_all), N_CELLS))
counts <- counts_all[, cells]; rm(counts_all)

preprocess <- function(m, alra = FALSE) {
  obj <- CreateSeuratObject(counts = m, meta.data = meta[colnames(m), , drop = FALSE])
  pct <- function(pattern) 100 * colSums(m[grepl(pattern, rownames(m)), , drop = FALSE]) / colSums(m)
  obj$percent.mt <- pct("^MT-")
  obj$percent.ribo <- pct("^RP[SL]")
  regress <- Filter(function(v) var(obj[[v]][, 1]) > 0, c("percent.mt", "percent.ribo"))
  obj <- suppressWarnings(SCTransform(obj, vst.flavor = "v1", vars.to.regress = regress, verbose = FALSE))
  if (alra) {
    set.seed(index)  # ALRA's randomized SVD
    obj <- suppressMessages(RunALRA(obj))
  }
  obj
}
assay_of <- function(obj) if ("alra" %in% Assays(obj)) "alra" else "SCT"

data <- list(gt = preprocess(counts))
data$gt_alra <- preprocess(counts, alra = TRUE)
for (d in 1:2) {
  set.seed(1000 * index + d)
  down <- scuttle::downsampleMatrix(counts, prop = prop, bycol = TRUE)
  data[[sprintf("down%d", d)]] <- preprocess(down)
  data[[sprintf("down%d_alra", d)]] <- preprocess(down, alra = TRUE)
}
umis <- vapply(1:2, function(d) mean(colSums(GetAssayData(data[[sprintf("down%d", d)]], assay = "RNA", layer = "counts"))), 0)
log("preprocessed; downsampled mean counts", paste(round(umis), collapse = ", "))

# --- programs (added read-out) ---
programs <- lapply(data, function(obj) {
  ip <- suppressWarnings(suppressMessages(InteractionPrograms(
    obj, assay = assay_of(obj), slot = "data", cell_types = "celltype", return.mat = TRUE)))
  est <- power_estimates(ip$cor_mat, 0.6)
  used <- if (is.na(est) || est > 3) 1 else est
  list(n_lr = ncol(ip$cor_mat), recommended = est, used = used, modules = ip$modules,
       modules_uncapped = if (is.na(est) || est == used) ip$modules else modules_at(ip$cor_mat, est))
})
log("programs:", paste(names(programs), vapply(programs, function(p) length(p$modules), 0L), collapse = " "))

# --- CCIM integration and LISI (the paper's read-out) ---
ccim <- function(obj, label) {
  cc <- suppressMessages(GenerateCCIM(obj, assay = assay_of(obj), slot = "data"))
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
  lisi <- lisi::compute_lisi(Embeddings(int, "pca"), int@meta.data, "dataset")$dataset
  lisi[int$dataset == gt$dataset[1]]
}
gt <- ccim(data$gt, "GT")
set.seed(index)
pairs <- colnames(gt)
if (length(pairs) > MAX_PAIRS) pairs <- sort(sample(pairs, MAX_PAIRS))
gt <- gt[, pairs]
lisi_rows <- list()
for (arm in c("down1", "down1_alra")) {
  dn <- ccim(data[[arm]], arm)[, pairs]
  shared <- intersect(rownames(gt), rownames(dn))
  rho <- cor(as.vector(as.matrix(GetAssayData(gt, layer = "data")[shared, ])),
             as.vector(as.matrix(GetAssayData(dn, layer = "data")[shared, pairs])), method = "spearman")
  lisi <- tryCatch(integrate(gt, dn), error = function(e) { log(arm, "integration failed:", conditionMessage(e)); NA_real_ })
  lisi_rows[[arm]] <- data.frame(index = index, arm = arm, n_pairs = length(pairs),
                                 frac_lisi_gt_1 = mean(lisi > 1 + 1e-6), lisi_mean = mean(lisi),
                                 frac_lisi_gt_1.5 = mean(lisi > 1.5), ccim_spearman = rho)
  log(arm, "LISI mean", round(mean(lisi), 3), "frac>1", round(mean(lisi > 1 + 1e-6), 3), "rho", round(rho, 3))
}
save_atomic(list(index = index, cells = cells, prop = prop, downsampled_mean_counts = umis,
                 lisi = do.call(rbind, lisi_rows), programs = programs), file)
log("done")
