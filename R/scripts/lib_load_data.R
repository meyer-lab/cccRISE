#' Shared helpers for loading an exported dataset (see export_dataset.py)
#' into a Seurat object, and for loading the shared ligand-receptor database.

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

#' Load the shared CellChat "Human-2020-Jin" ligand-receptor table exported
#' alongside the dataset. Multi-subunit complexes use "&" as a separator
#' (matching the Python-side convention in cellcommunicationpf2/import_data.py).
load_lr_pairs <- function(data_dir) {
  read.csv(file.path(data_dir, "lr_pairs.csv"), stringsAsFactors = FALSE)
}

#' Reduce the (possibly multi-subunit) LR table to simple one-gene-ligand /
#' one-gene-receptor pairs, for tools that don't support receptor complexes.
#' Complexes (subunits joined by "&") are dropped.
simple_lr_pairs <- function(lr_df) {
  keep <- !grepl("&", lr_df$ligand_symbol) & !grepl("&", lr_df$receptor_symbol)
  unique(lr_df[keep, c("ligand_symbol", "receptor_symbol")])
}
