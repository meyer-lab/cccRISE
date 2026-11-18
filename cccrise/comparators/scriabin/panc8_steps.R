# Steps shared by panc8_noise.R and panc8_lisi_checks.R: the Scriabin paper's
# technical-noise analysis (Wilk et al. 2024, Nat Biotechnol 42:470, Fig. 2d-e;
# Methods "Analysis of the impact of sparsity on single-cell-resolution CCC
# analysis"), run with the authors' own package code: upstream Scriabin at
# bce695b (GenerateCCIM, InteractionPrograms; upstream.R), Seurat's anchor
# integration and the lisi package. README.md has the R setup.
#
# Random numbers. Every random draw in the analysis, and who seeds it:
#
#   Global RNG (seeded here, by the caller, with set.seed):
#     - which 300 C1 cells form a subsample      (base::sample)
#     - the read downsampling to inDrop depth    (scuttle::downsampleMatrix)
#     - ALRA's randomized SVD                    (rsvd, via RunALRA)
#     - which GT cell-cell pairs enter integration (base::sample; only
#       panc8_lisi_checks.R's capped settings: panel A uses all of them)
#   Hard-coded inside the authors' / Seurat's code (the caller cannot reach
#   them through set.seed; same draw every run unless the argument is changed):
#     - SCTransform's 2,000-gene step-1 subsample  set.seed(seed.use = 1448145)
#     - RunPCA's irlba start                       set.seed(seed.use = 42)
#     - RunCCA's SVD start (anchor finding): set.seed(seed.use = 42) for
#       irlba, but above 10^6 cross-product entries (>= 1,000 pairs a side,
#       every run here) Seurat 5 uses RSpectra::svds, whose start vector is
#       drawn in C++ by Spectra's SimpleRandom at seed 0 (R's RNG never used)
#     - Annoy's kNN index (anchor finding)         RcppAnnoy's fixed default seed
#     - InteractionPrograms' cell subsampling      set.seed(seed = 1111); only
#       used above iterate.threshold = 300 cells, so never at 300 cells
#   Deterministic: GenerateCCIM, ScaleData, IntegrateData's weights, the WGCNA
#   program steps, and lisi::compute_lisi (exact kd-tree kNN, RANN::nn2).
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(qlcMatrix); library(pbapply); library(SeuratWrappers)
  library(WGCNA); library(flashClust); library(dynamicTreeCut)
})
pboptions(type = "none")
options(future.globals.maxSize = Inf)
source("cccrise/comparators/scriabin/upstream.R")

N_CELLS <- 300
#: Fig. 2e's inDrop coverage: mean UMIs per cell of panc8's inDrop cells
INDROP_UMI <- 5828
SRC <- "singlecell_cache/authors/panc8_paper"

# The C1 subsample `index`: its 300 cells (global RNG: set.seed(index)), their
# full-depth counts, and the downsampling proportion to inDrop depth
c1_subsample <- function(index) {
  counts_all <- readRDS(file.path(SRC, "c1_counts.rds"))
  prop <- INDROP_UMI / mean(colSums(counts_all))
  set.seed(index)
  cells <- sort(sample(colnames(counts_all), N_CELLS))
  list(counts = counts_all[, cells], cells = cells, prop = prop,
       meta = readRDS(file.path(SRC, "c1_meta.rds"))[cells, , drop = FALSE])
}

# Downsampling draw d of subsample `index` (global RNG: set.seed(1000 * index + d))
downsample <- function(counts, prop, index, d) {
  set.seed(1000 * index + d)
  scuttle::downsampleMatrix(counts, prop = prop, bycol = TRUE)
}

# SCTransform v1 (the paper's sctransform 0.3.3 default flavor), regressing %
# mitochondrial and % ribosomal reads (Methods, "unless otherwise noted");
# `alra`: then RunALRA, whose randomized SVD is seeded set.seed(index).
# `sct_seed`: SCTransform's hard-coded seed.use (1448145 unless varied).
preprocess <- function(m, meta, index, alra = FALSE, sct_seed = 1448145) {
  obj <- CreateSeuratObject(counts = m, meta.data = meta[colnames(m), , drop = FALSE])
  pct <- function(pattern) 100 * colSums(m[grepl(pattern, rownames(m)), , drop = FALSE]) / colSums(m)
  obj$percent.mt <- pct("^MT-")
  obj$percent.ribo <- pct("^RP[SL]")
  regress <- Filter(function(v) var(obj[[v]][, 1]) > 0, c("percent.mt", "percent.ribo"))
  obj <- suppressWarnings(SCTransform(obj, vst.flavor = "v1", vars.to.regress = regress,
                                      seed.use = sct_seed, verbose = FALSE))
  if (alra) {
    set.seed(index)
    obj <- suppressMessages(RunALRA(obj))
  }
  obj
}
assay_of <- function(obj) if ("alra" %in% Assays(obj)) "alra" else "SCT"

# The subsample `sub` (c1_subsample) preprocessed at full depth (gt) and for
# each downsampling draw d in `draws`, raw (down<d>) and ALRA-denoised
# (down<d>_alra). Every draw is seeded on its own (downsample, preprocess),
# so draw 1 is the same whichever draws are built.
arms <- function(sub, index, draws) {
  data <- list(gt = preprocess(sub$counts, sub$meta, index))
  for (d in draws) {
    down <- downsample(sub$counts, sub$prop, index, d)
    data[[sprintf("down%d", d)]] <- preprocess(down, sub$meta, index)
    data[[sprintf("down%d_alra", d)]] <- preprocess(down, sub$meta, index, alra = TRUE)
  }
  data
}

# upstream GenerateCCIM, as a Seurat object labelled `label`
ccim <- function(obj, label) {
  cc <- suppressMessages(GenerateCCIM(obj, assay = assay_of(obj), slot = "data"))
  cc <- DietSeurat(cc, assays = "CCIM")
  # Seurat < 5 (the paper's) set data = counts on creation; v5 leaves it empty
  cc <- SetAssayData(cc, layer = "data", new.data = GetAssayData(cc, layer = "counts"))
  cc$dataset <- label
  cc
}

# The GT cell-cell pairs that enter integration, in draw order (global RNG:
# set.seed(index)); the first `cap` of them are sample(pairs, cap)
pair_order <- function(pairs, index) {
  set.seed(index)
  sample(pairs)
}

# Seurat anchor integration (CCA) of the GT and downsampled CCIMs on 30
# dimensions, then LISI of each GT pair (1: only GT pairs nearby; 2: evenly
# mixed) on the integrated data's first 30 PCs. `pca_seed`, `cca_seed`: the
# hard-coded RunPCA / RunCCA seeds (the latter inert on the RSpectra path).
integrate <- function(gt, down, pca_seed = 42, cca_seed = 42) {
  objs <- lapply(list(gt, down), function(o) {
    o <- RenameCells(o, add.cell.id = o$dataset[1])
    VariableFeatures(o) <- rownames(o)
    ScaleData(o, verbose = FALSE)
  })
  feats <- intersect(rownames(objs[[1]]), rownames(objs[[2]]))
  anchors <- with_cca_seed(cca_seed, FindIntegrationAnchors(
    objs, anchor.features = feats, reduction = "cca", dims = 1:30, verbose = FALSE))
  int <- IntegrateData(anchors, dims = 1:30, verbose = FALSE)
  int <- ScaleData(int, verbose = FALSE) |> RunPCA(npcs = 30, seed.use = pca_seed, verbose = FALSE)
  gt_lisi(int)
}

# The same two CCIMs merged without integration (control: how much of the
# mixing the integration step itself makes)
unintegrated <- function(gt, down, pca_seed = 42) {
  m <- merge(RenameCells(gt, add.cell.id = "GT"), RenameCells(down, add.cell.id = "DN"))
  m <- JoinLayers(m)
  VariableFeatures(m) <- intersect(rownames(gt), rownames(down))
  m <- ScaleData(m, verbose = FALSE) |> RunPCA(npcs = 30, seed.use = pca_seed, verbose = FALSE)
  gt_lisi(m)
}

gt_lisi <- function(obj) {
  lisi <- lisi::compute_lisi(Embeddings(obj, "pca"), obj@meta.data, "dataset")$dataset
  lisi[obj$dataset == "GT"]
}

# RunCCA hard-codes set.seed(42) (its seed.use default, which
# FindIntegrationAnchors does not pass on); `seed` replaces that default for
# the duration of `expr`
with_cca_seed <- function(seed, expr) {
  if (seed == 42) return(expr)
  original <- Seurat:::RunCCA.default
  patched <- original
  formals(patched)$seed.use <- seed
  utils::assignInNamespace("RunCCA.default", patched, "Seurat")
  on.exit(utils::assignInNamespace("RunCCA.default", original, "Seurat"))
  force(expr)
}

# atomic, flushed checkpoint write (the .rds is the job's done-file)
save_atomic <- function(x, file) {
  partial <- paste0(file, ".partial")
  saveRDS(x, partial)
  system2("sync", c("-d", partial))
  file.rename(partial, file)
}
