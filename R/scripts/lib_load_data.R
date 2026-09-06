#' Shared helper for loading an exported dataset (see export_dataset.py)
#' into a Seurat object.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

#' Read a dataset exported by scripts/export_dataset.py into a Seurat object.
#'
#' @param data_dir directory containing matrix.mtx.gz, genes.tsv.gz,
#'   barcodes.tsv.gz, metadata.csv
#' @param min_cells,min_features passed to CreateSeuratObject
load_exported_seurat <- function(data_dir, min_cells = 3, min_features = 1) {
  counts <- ReadMtx(
    mtx = file.path(data_dir, "matrix.mtx.gz"),
    cells = file.path(data_dir, "barcodes.tsv.gz"),
    features = file.path(data_dir, "genes.tsv.gz"),
    feature.column = 2
  )
  meta <- read.csv(file.path(data_dir, "metadata.csv"), row.names = 1)
  stopifnot(identical(rownames(meta), colnames(counts)))

  obj <- CreateSeuratObject(
    counts = counts,
    meta.data = meta,
    min.cells = min_cells,
    min.features = min_features
  )
  obj
}
