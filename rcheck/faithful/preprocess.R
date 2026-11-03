# Authors' preprocessing for the faithful Scriabin run (Wilk et al. 2023,
# Methods "Pre-processing of scRNA-seq data" and the interaction-programs
# vignette): SCTransform on the whole dataset regressing % mitochondrial and
# % ribosomal reads, then ALRA denoising (SeuratWrappers::RunALRA, the
# vignette's recommendation for droplet data). ALRA runs per sample: on the
# whole dataset its dense intermediates exceeded a 56 GB cap (the vignette's
# examples are 2-14k cells; Scriabin builds every CCIM and TOM per sample
# anyway). Optionally the raw UMIs are
# first downsampled per cell with scuttle::downsampleMatrix (the paper's
# technical-noise model, Fig. 2d-e) to `prop` of their depth.
#
# Writes <out>/sct_lr.rds and <out>/alra_lr.rds (LR genes x cells: the only
# rows InteractionPrograms reads), <out>/sct_full.rds, and with `full` also
# <out>/alra_full.rds (all genes, for AddModuleScore's control genes).
# Checkpoints: sct_full.rds and <out>/alra/<sample>.rds.
#
#   Rscript rcheck/faithful/preprocess.R rcheck/out/faithful/input rcheck/out/faithful/depth1 1 full
#   Rscript rcheck/faithful/preprocess.R rcheck/out/faithful/input rcheck/out/faithful/depth0.52 0.5166
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(SeuratWrappers); library(dplyr)
})
args <- commandArgs(TRUE)
indir <- args[1]; out <- args[2]; prop <- as.numeric(args[3]); full <- isTRUE(args[4] == "full")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
options(future.globals.maxSize = Inf)  # SCTransform runs sequentially; lift the transfer guard

lr <- scriabin::LoadLR(species = "human", database = "OmniPath")
meta <- read.csv(file.path(indir, "meta.csv"), check.names = FALSE)
rownames(meta) <- meta$cell
if (file.exists(file.path(out, "sct_full.rds"))) {
  sct <- readRDS(file.path(out, "sct_full.rds"))
  log("loaded SCT data")
} else {
  genes <- read.csv(file.path(indir, "genes.csv"), header = FALSE)$V1
  nnz_p <- readBin(file.path(indir, "indptr.bin"), "integer", n = nrow(meta) + 1, size = 8)
  nnz <- nnz_p[length(nnz_p)]
  counts <- sparseMatrix(
    i = readBin(file.path(indir, "indices.bin"), "integer", n = nnz, size = 4) + 1L,
    p = nnz_p, x = as.numeric(readBin(file.path(indir, "data.bin"), "integer", n = nnz, size = 4)),
    dims = c(length(genes), nrow(meta)), dimnames = list(genes, meta$cell))
  log("counts", nrow(counts), "x", ncol(counts))
  if (prop < 1) {
    set.seed(20261009)
    counts <- scuttle::downsampleMatrix(counts, prop = prop, bycol = TRUE)
    log("downsampled to", prop, "; mean UMI/cell", mean(Matrix::colSums(counts)))
  }
  seu <- CreateSeuratObject(counts = counts, meta.data = meta[, -1])
  rm(counts); invisible(gc())
  seu <- SCTransform(seu, vars.to.regress = c("percent.mt", "percent.ribo"), verbose = FALSE)
  log("SCTransform done")
  sct <- GetAssayData(seu, assay = "SCT", layer = "data")
  rm(seu); invisible(gc())
  saveRDS(sct, file.path(out, "sct_full.rds"), compress = FALSE)
}
lr_genes <- intersect(union(lr$source_genesymbol, lr$target_genesymbol), rownames(sct))
saveRDS(sct[lr_genes, ], file.path(out, "sct_lr.rds"))

dir.create(file.path(out, "alra"), showWarnings = FALSE)
ranks <- c()
for (s in sort(unique(meta$sample))) {
  file <- file.path(out, "alra", paste0(s, ".rds"))
  cells <- meta$cell[meta$sample == s]
  if (!file.exists(file)) {
    obj <- CreateSeuratObject(counts = sct[, cells], assay = "SCT")
    obj <- SetAssayData(obj, layer = "data", new.data = sct[, cells])
    set.seed(1)  # rsvd's random projections; the vignette sets no seed
    obj <- suppressWarnings(suppressMessages(RunALRA(obj)))
    saveRDS(list(alra = GetAssayData(obj, assay = "alra", layer = "data"),
                 k = Tool(obj, slot = "RunALRA")$k), file)
    log("ALRA", s, length(cells), "cells, rank", Tool(obj, slot = "RunALRA")$k)
  }
  ranks[s] <- readRDS(file)$k
}
rm(sct); invisible(gc())
parts <- lapply(sort(unique(meta$sample)), function(s) readRDS(file.path(out, "alra", paste0(s, ".rds")))$alra)
saveRDS(do.call(cbind, lapply(parts, function(m) m[lr_genes, ]))[, meta$cell], file.path(out, "alra_lr.rds"))
if (full) {
  alra <- do.call(cbind, parts)[, meta$cell]
  saveRDS(alra, file.path(out, "alra_full.rds"), compress = FALSE)
  density <- Matrix::nnzero(alra) / length(alra)
} else density <- NA
writeLines(jsonlite::toJSON(list(prop = prop, n_lr_genes = length(lr_genes),
  alra_rank = as.list(ranks), alra_density = density), auto_unbox = TRUE),
  file.path(out, "preprocess.json"))
saveRDS(meta[, c("sample", "celltype")], file.path(out, "meta.rds"))
log("done")
