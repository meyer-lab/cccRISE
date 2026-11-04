# panc8 inputs with the authors' preprocessing, as Scriabin's
# interaction-programs vignette (rcheck/upstream/scriabin/vignettes/
# interaction-programs.Rmd): panc8.SeuratData 3.0.2, inDrop cells only,
# SCTransform (defaults), SeuratWrappers::RunALRA. Writes, under
# singlecell_cache/authors/panc8/: alra_data.rds (all genes x cells, the ALRA
# assay's data), meta.rds (sample = inDrop donor, celltype) and
# meta_bytype.rds (the vignette's split: sample = celltype).
#   R_LIBS=singlecell_cache/authors/panc8/rlib Rscript rcheck/authors/panc8_prep.R
# (panc8.SeuratData installed into that library from seurat.nygenome.org.)
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({ library(Seurat); library(Matrix); library(SeuratWrappers); library(dplyr) })
options(future.globals.maxSize = Inf)
out <- "singlecell_cache/authors/panc8"
data("panc8", package = "panc8.SeuratData")
panc8 <- UpdateSeuratObject(panc8)
panc_id <- subset(panc8, cells = colnames(panc8)[panc8$tech == "indrop"])
set.seed(1111)
panc_id <- SCTransform(panc_id, verbose = FALSE)
panc_id <- RunALRA(panc_id)
alra <- GetAssayData(panc_id, assay = "alra", layer = "data")
cat("ALRA:", dim(alra), "density", mean(alra > 0), "\n")
saveRDS(as(alra, "CsparseMatrix"), file.path(out, "alra_data.rds"))
donor <- if ("replicate" %in% colnames(panc_id@meta.data)) panc_id$replicate else panc_id$orig.ident
meta <- data.frame(sample = as.character(donor), celltype = as.character(panc_id$celltype),
                   row.names = colnames(panc_id))
saveRDS(meta, file.path(out, "meta.rds"))
saveRDS(transform(meta, sample = celltype), file.path(out, "meta_bytype.rds"))
print(table(meta$sample)); print(sort(table(meta$celltype)))
