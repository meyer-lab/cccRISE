# panc8 (authors' preprocessing) for the port, exactly as upstream
# InteractionPrograms builds its inputs: OmniPath (curation >= 7, via the
# stub's verbatim LoadLR) filtered to genes in the ALRA assay, in upstream's
# order, and the LR genes' ALRA values. Writes
# singlecell_cache/authors/panc8/port/{lr_expr.mtx (cells x genes), genes.txt,
# lr.csv (ligand, receptor, lr_name), meta.csv (cell, donor, celltype)}.
#   Rscript rcheck/authors/panc8_export_port.R
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({ library(Matrix); library(dplyr) })
O <- "singlecell_cache/authors/panc8"; P <- file.path(O, "port"); dir.create(P, showWarnings = FALSE)
data <- readRDS(file.path(O, "alra_data.rds")); meta <- readRDS(file.path(O, "meta.rds"))
lit <- scriabin::LoadLR(species = "human", database = "OmniPath")
lit <- lit %>% filter(source_genesymbol %in% rownames(data), target_genesymbol %in% rownames(data))
lr <- data.frame(ligand = lit$source_genesymbol, receptor = lit$target_genesymbol)
lr$lr_name <- paste(lr$ligand, lr$receptor, sep = "=")
cat(nrow(lr), "pairs;", sum(duplicated(lr$lr_name)), "duplicate names\n")
genes <- sort(unique(c(lr$ligand, lr$receptor)))
writeMM(t(data[genes, ]), file.path(P, "lr_expr.mtx"))
writeLines(genes, file.path(P, "genes.txt"))
write.csv(lr, file.path(P, "lr.csv"), row.names = FALSE)
write.csv(data.frame(cell = rownames(meta), donor = meta$sample, celltype = meta$celltype),
          file.path(P, "meta.csv"), row.names = FALSE)
