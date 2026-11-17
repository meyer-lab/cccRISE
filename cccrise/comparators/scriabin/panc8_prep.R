# The Fluidigm C1 pancreas islets of panc8 (panc8.SeuratData 3.0.2, the
# dataset of the Scriabin vignette), raw counts and meta: the deep "ground
# truth" Fig. 2d-e downsamples to inDrop depth. From PR #275's
# rcheck/authors/panc8_paper_prep.R, C1 part only. Writes c1_counts.rds and
# c1_meta.rds under singlecell_cache/authors/panc8_paper/. No randomness.
#   Rscript cccrise/comparators/scriabin/panc8_prep.R
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })
out <- "singlecell_cache/authors/panc8_paper"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
data("panc8", package = "panc8.SeuratData")
panc8 <- suppressWarnings(UpdateSeuratObject(panc8))
c1 <- subset(panc8, cells = colnames(panc8)[panc8$tech == "fluidigmc1"])
counts <- as(GetAssayData(c1, layer = "counts"), "CsparseMatrix")
saveRDS(counts, file.path(out, "c1_counts.rds"))
saveRDS(data.frame(celltype = as.character(c1$celltype), donor = as.character(c1$replicate),
                   row.names = colnames(c1)), file.path(out, "c1_meta.rds"))
cat("C1:", ncol(c1), "cells, mean counts", round(mean(colSums(counts))), "\n")
