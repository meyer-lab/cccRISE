# panc8 inDrop raw counts, for cccRISE's own preprocessing
# (rcheck/authors/panc8_cccrise_inputs.py): the cells of meta.rds
# (rcheck/authors/panc8_prep.R). Writes
# singlecell_cache/authors/panc8/raw/{counts.mtx (genes x cells), genes.txt, cells.txt}.
#   R_LIBS=singlecell_cache/authors/panc8/rlib Rscript rcheck/authors/panc8_export_counts.R
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })
data("panc8", package = "panc8.SeuratData")
panc8 <- UpdateSeuratObject(panc8)
meta <- readRDS("singlecell_cache/authors/panc8/meta.rds")
counts <- GetAssayData(panc8, assay = "RNA", layer = "counts")[, rownames(meta)]
out <- "singlecell_cache/authors/panc8/raw"; dir.create(out, showWarnings = FALSE)
writeMM(counts, file.path(out, "counts.mtx"))
writeLines(rownames(counts), file.path(out, "genes.txt"))
writeLines(colnames(counts), file.path(out, "cells.txt"))
cat(dim(counts), "max", max(counts), "\n")
