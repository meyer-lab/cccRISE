# Upstream NICHES (msraredon/NICHES) on cccRISE's inputs, per sample:
# CellToSystem scores each cell as a sender and SystemToCell as a receiver of
# the listed LR pairs, against the mean of the sample's other side (blend
# mean, raw output). Both modes are deterministic. Run by
# `cccrise.runners.run_niches`, which writes the inputs into <dir>:
#   expression.mtx  tokens x cells, cccRISE's normalized expression
#   tokens.txt      the ligands and receptors (a complex is one min-aggregated
#                   token, so NICHES treats it as a single feature)
#   cells.csv       cell, sample, celltype
#   pairs.csv       ligand, receptor
# and reads <dir>/{c2s,s2c}.mtx (LR pair x cell), {c2s,s2c}_{rows,cells}.txt
# and versions.yaml.
#   Rscript niches_system.R <dir> <NICHES source> <R library> <workers>
#
# R setup (the versions the results were made with, recorded in each run's
# versions.yaml): R 4.5.2 with Seurat 5.6.0, Matrix 1.7.4 and dplyr in
# <R library>. NICHES is not installed: its R/ files are sourced from a clone
# at 5d932b8 (`run_niches.NICHES_COMMIT`, checked):
#   git clone https://github.com/msraredon/NICHES <NICHES source>
#   git -C <NICHES source> checkout 5d932b8
args <- commandArgs(TRUE)
dir <- args[1]
.libPaths(c(normalizePath(args[3]), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(parallel)
})
for (f in list.files(file.path(args[2], "R"), full.names = TRUE)) source(f)

meta <- read.csv(file.path(dir, "cells.csv"), row.names = 1)
data <- as(readMM(file.path(dir, "expression.mtx")), "CsparseMatrix")
dimnames(data) <- list(readLines(file.path(dir, "tokens.txt")), rownames(meta))
pairs <- read.csv(file.path(dir, "pairs.csv"))
database <- data.frame(lig = pairs$ligand, rec = pairs$receptor)

run_sample <- function(sample) {
  cells <- rownames(meta)[meta$sample == sample]
  m <- data[, cells, drop = FALSE]
  obj <- CreateSeuratObject(counts = m, meta.data = meta[cells, , drop = FALSE], assay = "IP")
  obj <- SetAssayData(obj, layer = "data", new.data = m)
  res <- suppressWarnings(suppressMessages(RunNICHES(
    obj, assay = "IP", LR.database = "custom", custom_LR_database = database,
    species = "human", cell_types = "celltype", meta.data.to.map = NULL,
    min.cells.per.ident = 1, CellToCell = FALSE, SystemToCell = TRUE,
    CellToSystem = TRUE, blend = "mean", output_format = "raw")))
  list(s2c = as(res$SystemToCell$SystemToCellMatrix, "CsparseMatrix"),
       c2s = as(res$CellToSystem$CellToSystemMatrix, "CsparseMatrix"))
}

samples <- sort(unique(meta$sample))
results <- mclapply(samples, run_sample, mc.cores = as.integer(args[4]), mc.preschedule = FALSE)
failed <- vapply(results, inherits, logical(1), "try-error")
if (any(failed)) stop("NICHES failed on ", paste(samples[failed], collapse = ", "), ": ", results[failed][[1]])

for (mode in c("s2c", "c2s")) {
  pieces <- lapply(results, `[[`, mode)
  rows <- sort(unique(unlist(lapply(pieces, rownames))))
  combined <- do.call(cbind, lapply(pieces, function(piece) {
    full <- Matrix(0, length(rows), ncol(piece), sparse = TRUE, dimnames = list(rows, colnames(piece)))
    full[rownames(piece), ] <- piece
    full
  }))
  writeMM(combined, file.path(dir, paste0(mode, ".mtx")))
  writeLines(rows, file.path(dir, paste0(mode, "_rows.txt")))
  writeLines(sub("^System—|—System$", "", colnames(combined)), file.path(dir, paste0(mode, "_cells.txt")))
}
writeLines(c(paste0("r: '", getRversion(), "'"),
             paste0("seurat: '", packageVersion("Seurat"), "'"),
             paste0("matrix: '", packageVersion("Matrix"), "'")),
           file.path(dir, "versions.yaml"))
