# Upstream InteractionPrograms with no randomness: the full CCIM of every
# sample (iterate.threshold above its size, so no subsampling), on all cells
# and with single cells left out (0-based indices from the command line).
#
#   Rscript rcheck/cell_drop.R rcheck/out/drop rcheck/out/drop/r 0 37 74
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(WGCNA); library(flashClust); library(dynamicTreeCut)
  library(qlcMatrix); library(pbapply); library(jsonlite)
})
pboptions(type = "none")
args <- commandArgs(TRUE); dir <- args[1]; outdir <- args[2]; drops <- as.integer(args[-(1:2)])
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
`%notin%` <- Negate(`%in%`)
source("rcheck/upstream/scriabin/R/interaction_programs.R")
expr <- as(readMM(file.path(dir, "expr.mtx")), "CsparseMatrix")
features <- read.csv(file.path(dir, "features.csv"), header = FALSE)$V1
meta <- read.csv(file.path(dir, "meta.csv")); lr <- read.csv(file.path(dir, "lr.csv"))
dimnames(expr) <- list(features, meta$cell); rownames(meta) <- meta$cell
seu <- CreateSeuratObject(counts = expr, meta.data = meta[, c("sample", "celltype")], assay = "RNA")
seu <- SetAssayData(seu, layer = "data", new.data = expr)
for (s in unique(meta$sample)) for (d in c(-1L, drops)) {
  file <- file.path(outdir, sprintf("%s_drop%d.json", s, d))
  if (file.exists(file)) next
  cells <- meta$cell[meta$sample == s]
  if (d >= 0) cells <- cells[-(d + 1)]
  t0 <- Sys.time()
  ip <- suppressWarnings(suppressMessages(InteractionPrograms(seu[, cells], assay = "RNA", slot = "data",
        database = "custom", ligands = lr$ligand, recepts = lr$receptor,
        iterate.threshold = length(cells) + 1, return.mat = TRUE)))
  write_json(ip$modules, file)
  cat(s, "drop", d, length(ip$modules), "modules", format(Sys.time() - t0), "\n")
}
