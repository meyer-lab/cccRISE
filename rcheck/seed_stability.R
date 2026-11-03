# Seed-to-seed stability of upstream InteractionPrograms with R's own RNG (no shims).
# Upstream sets set.seed(1111) inside the function, so the seed is passed as `seed`.
# Cell types with fewer than min.cell = 3 cells are dropped (upstream errors on them).
#
#   Rscript rcheck/seed_stability.R rcheck/out/check rcheck/out/seeds
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(WGCNA); library(flashClust); library(dynamicTreeCut)
  library(qlcMatrix); library(pbapply); library(jsonlite)
})
pboptions(type = "none")
args <- commandArgs(TRUE); dir <- args[1]; outdir <- args[2]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
`%notin%` <- Negate(`%in%`)
source("rcheck/upstream/scriabin/R/interaction_programs.R")
expr <- as(readMM(file.path(dir, "expr.mtx")), "CsparseMatrix")
features <- read.csv(file.path(dir, "features.csv"), header = FALSE)$V1
meta <- read.csv(file.path(dir, "meta.csv")); lr <- read.csv(file.path(dir, "lr.csv"))
dimnames(expr) <- list(features, meta$cell); rownames(meta) <- meta$cell
seu <- CreateSeuratObject(counts = expr, meta.data = meta[, c("sample", "celltype")], assay = "RNA")
seu <- SetAssayData(seu, layer = "data", new.data = expr)
for (s in unique(meta$sample)) for (seed in 1:5) {
  obj <- subset(seu, subset = sample == s)
  tt <- table(obj$celltype); obj <- obj[, obj$celltype %in% names(tt)[tt >= 3]]
  if (file.exists(file.path(outdir, sprintf("r_%s_seed%d.json", s, seed)))) next
  ip <- suppressWarnings(suppressMessages(InteractionPrograms(obj, assay = "RNA", slot = "data", database = "custom",
        ligands = lr$ligand, recepts = lr$receptor, cell_types = "celltype", return.mat = TRUE, seed = seed)))
  write_json(ip$modules, file.path(outdir, sprintf("r_%s_seed%d.json", s, seed)))
}
