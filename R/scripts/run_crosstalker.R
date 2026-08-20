#' Run CrossTalkeR (CostaLab/CrossTalkeR) on an exported dataset.
#'
#' CrossTalkeR does differential LR-network analysis/visualization between
#' two conditions, given per-condition LR score tables in a CellPhoneDB/LIANA
#' -like format. We generate those two tables ourselves with CellChat
#' (computeCommunProb) run separately on each condition subset, then hand
#' them to CrossTalkeR::generate_report(report = FALSE) to get the merged
#' differential network object without needing pandoc/Rmd rendering.
#'
#' Usage: Rscript run_crosstalker.R <data_dir> <group_by_column> <condition_column> <cond1> <cond2> <out_dir>
#' Example: Rscript run_crosstalker.R ../data/balf_covid celltype condition Control "Severe COVID-19" ../output/crosstalker

suppressPackageStartupMessages({
  library(CrossTalkeR)
  library(CellChat)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[[1]]
group_by <- args[[2]]
condition_col <- args[[3]]
cond1 <- args[[4]]
cond2 <- args[[5]]
out_dir <- args[[6]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("lib_load_data.R")

seurat_obj <- load_exported_seurat(data_dir)
seurat_obj <- NormalizeData(seurat_obj)

run_cellchat_lr_table <- function(obj) {
  cellchat <- createCellChat(object = obj, group.by = group_by, assay = "RNA")
  cellchat@DB <- CellChatDB.human
  cellchat <- subsetData(cellchat)
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  cellchat <- computeCommunProb(cellchat, type = "triMean")
  cellchat <- filterCommunication(cellchat, min.cells = 10)
  subsetCommunication(cellchat)
}

to_crosstalker_format <- function(df_lr) {
  data.frame(
    source = df_lr$source,
    target = df_lr$target,
    gene_A = df_lr$ligand,
    gene_B = df_lr$receptor,
    type_gene_A = "Ligand",
    type_gene_B = "Receptor",
    MeanLR = df_lr$prob
  )
}

conditions <- list(cond1 = cond1, cond2 = cond2)
lrpaths <- c()
for (nm in names(conditions)) {
  cond_val <- conditions[[nm]]
  sub_obj <- subset(seurat_obj, cells = colnames(seurat_obj)[
    seurat_obj[[condition_col, drop = TRUE]] == cond_val
  ])
  df_lr <- run_cellchat_lr_table(sub_obj)
  ct_df <- to_crosstalker_format(df_lr)
  path <- file.path(out_dir, paste0("lr_table_", nm, ".csv"))
  write.csv(ct_df, path, row.names = FALSE)
  lrpaths[nm] <- path
}

data <- generate_report(
  lrpaths = lrpaths,
  out_path = paste0(out_dir, "/"),
  threshold = 0,
  report = FALSE,
  comparison = list(c("cond1", "cond2"))
)

saveRDS(data, file.path(out_dir, "crosstalker_object.rds"))
for (tbl_name in names(data@tables)) {
  tbl <- data@tables[[tbl_name]]
  if (is.data.frame(tbl)) {
    write.csv(tbl, file.path(out_dir, paste0("table_", tbl_name, ".csv")), row.names = FALSE)
  }
}
cat("CrossTalkeR done ->", out_dir, "\n")
cat("Tables written:", paste(names(data@tables), collapse = ", "), "\n")
