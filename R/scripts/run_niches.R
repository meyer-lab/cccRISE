#' Run NICHES (msraredon/NICHES) on an exported dataset.
#'
#' NICHES computes a per-cell-pair ligand-receptor signal matrix; we use its
#' bundled "fantom5" human LR database (NICHES does not consume the shared
#' complex-aware Jin/CellChat table directly).
#'
#' Usage: Rscript run_niches.R <data_dir> <group_by_column> <out_dir>

suppressPackageStartupMessages({
  library(NICHES)
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[[1]]
group_by <- args[[2]]
out_dir <- args[[3]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("lib_load_data.R")

seurat_obj <- load_exported_seurat(data_dir)
seurat_obj <- NormalizeData(seurat_obj)
Idents(seurat_obj) <- seurat_obj[[group_by, drop = TRUE]]

niches_out <- RunNICHES(
  object = seurat_obj,
  LR.database = "fantom5",
  species = "human",
  assay = "RNA",
  cell_types = group_by,
  min.cells.per.ident = 10,
  min.cells.per.gene = 10,
  meta.data.to.map = intersect(c(group_by, "condition"), colnames(seurat_obj[[]])),
  CellToCell = TRUE,
  CellToSystem = FALSE,
  SystemToCell = FALSE
)

cell_to_cell <- niches_out[["CellToCell"]]

signal_matrix <- GetAssayData(cell_to_cell, layer = "counts")
writeMM(signal_matrix, file.path(out_dir, "celltocell_signal_matrix.mtx"))
writeLines(rownames(signal_matrix), file.path(out_dir, "celltocell_lr_features.txt"))
writeLines(colnames(signal_matrix), file.path(out_dir, "celltocell_edge_ids.txt"))
write.csv(cell_to_cell[[]], file.path(out_dir, "celltocell_edge_metadata.csv"))

saveRDS(niches_out, file.path(out_dir, "niches_object.rds"))
cat("NICHES done ->", out_dir, "\n")
