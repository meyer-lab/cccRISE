# Inputs for reproducing the Scriabin paper's own robustness analyses on panc8
# (Wilk et al. 2024, Nat Biotechnol; rcheck/authors/panc8_paper_*.R):
#
# - inDrop cells, preprocessed as the interaction-programs vignette
#   (SCTransform, then SeuratWrappers::RunALRA), for Extended Data Fig. 8f;
# - Fluidigm C1 cells, raw counts and meta, for Fig. 2d-e (the deep "ground
#   truth" that is downsampled to inDrop depth).
#
# SCTransform runs as vst.flavor = "v1": the paper's Reporting Summary lists
# Seurat 4.1 with sctransform 0.3.3, whose default flavor is v1 (Seurat 5's is
# v2). Writes, under singlecell_cache/authors/panc8_paper/: indrop_alra.rds,
# indrop_meta.rds, c1_counts.rds, c1_meta.rds.
#   R_LIBS=singlecell_cache/authors/panc8/rlib Rscript rcheck/authors/panc8_paper_prep.R
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({ library(Seurat); library(Matrix); library(SeuratWrappers) })
options(future.globals.maxSize = Inf)
out <- "singlecell_cache/authors/panc8_paper"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
data("panc8", package = "panc8.SeuratData")
panc8 <- suppressWarnings(UpdateSeuratObject(panc8))
meta_of <- function(obj) data.frame(celltype = as.character(obj$celltype),
                                    donor = as.character(obj$replicate), row.names = colnames(obj))

c1 <- subset(panc8, cells = colnames(panc8)[panc8$tech == "fluidigmc1"])
saveRDS(as(GetAssayData(c1, layer = "counts"), "CsparseMatrix"), file.path(out, "c1_counts.rds"))
saveRDS(meta_of(c1), file.path(out, "c1_meta.rds"))
cat("C1:", ncol(c1), "cells, mean counts", round(mean(colSums(GetAssayData(c1, layer = "counts")))), "\n")

id <- subset(panc8, cells = colnames(panc8)[panc8$tech == "indrop"])
set.seed(1111)  # ALRA's randomized SVD is otherwise unseeded
id <- SCTransform(id, vst.flavor = "v1", verbose = FALSE)
id <- RunALRA(id)
alra <- GetAssayData(id, assay = "alra", layer = "data")
cat("inDrop ALRA:", dim(alra), "density", round(mean(alra > 0), 3), "\n")
saveRDS(as(alra, "CsparseMatrix"), file.path(out, "indrop_alra.rds"))
saveRDS(meta_of(id), file.path(out, "indrop_meta.rds"))
