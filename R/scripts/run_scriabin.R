#' Run Scriabin (BlishLab/scriabin) on an exported dataset.
#'
#' Scriabin's GenerateInteractome() computes a single-cell-resolution
#' ligand-receptor edge score for every (sender cell, receiver cell) pair, so
#' its cost is O(n_send * n_rec) -- keep n_send/n_rec to tens of cells, not
#' thousands, unless you are prepared to let it run for a long time.
#'
#' Requires patch_scriabin.R (sourced below) -- see that file for the two
#' upstream bugs it works around.
#'
#' Usage: Rscript run_scriabin.R <data_dir> <group_by_column> <send_group> <rec_group> <out_dir> [n_send] [n_rec] [seed]
#' Example: Rscript run_scriabin.R ../data/balf_covid_small celltype Macrophages T ../output/scriabin 25 25 0

suppressPackageStartupMessages({
  library(scriabin)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[[1]]
group_by <- args[[2]]
send_group <- args[[3]]
rec_group <- args[[4]]
out_dir <- args[[5]]
n_send <- if (length(args) >= 6) as.integer(args[[6]]) else 25L
n_rec <- if (length(args) >= 7) as.integer(args[[7]]) else 25L
seed <- if (length(args) >= 8) as.integer(args[[8]]) else 0L
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("lib_load_data.R")
source("patch_scriabin.R")
set.seed(seed)

# database = "custom": supply the shared LR resource ourselves, for
# comparability with run_cellchat.R. ligands/recepts are picked up by
# GenerateInteractome() from this (calling) environment -- see
# patch_scriabin.R for why that lexical lookup is how upstream intends it.
lr_shared <- simple_lr_pairs(load_lr_pairs(data_dir))
ligands <- lr_shared$ligand_symbol
recepts <- lr_shared$receptor_symbol

seurat_obj <- load_exported_seurat(data_dir)
seurat_obj <- NormalizeData(seurat_obj)

group_vec <- seurat_obj[[group_by, drop = TRUE]]
send_pool <- colnames(seurat_obj)[group_vec == send_group]
rec_pool <- colnames(seurat_obj)[group_vec == rec_group]
stopifnot(length(send_pool) > 0, length(rec_pool) > 0)

send_cells <- sample(send_pool, min(n_send, length(send_pool)))
rec_cells <- sample(rec_pool, min(n_rec, length(rec_pool)))

# GenerateInteractome() looks up ranked_genes[[cell_id]] purely to flag
# whether a ligand/receptor was among that cell's top variable genes; we
# don't use that annotation downstream, so supply an inert placeholder
# (an empty named vector per cell) rather than reverse-engineering the
# undocumented per-cell ranked-gene format the function expects.
ranked_genes <- setNames(
  rep(list(character(0)), ncol(seurat_obj)),
  colnames(seurat_obj)
)

interactome <- GenerateInteractome(
  seu = seurat_obj,
  ranked_genes = ranked_genes,
  cluster_results = NULL,
  use_clusters = FALSE,
  send_cells = send_cells,
  rec_cells = rec_cells,
  cell.type.calls = group_by,
  assay.use = "RNA",
  slot.use = "data",
  group.by = group_by,
  database = "custom"  # shared LR resource, for comparability with run_cellchat.R
)

write.csv(interactome, file.path(out_dir, "scriabin_cell_pair_edges.csv"), row.names = FALSE)
saveRDS(interactome, file.path(out_dir, "scriabin_cell_pair_edges.rds"))
cat("Scriabin done ->", out_dir, "\n")
cat(nrow(interactome), "nonzero cell-pair x LR edges across",
    length(send_cells), "sender and", length(rec_cells), "receiver cells\n")
