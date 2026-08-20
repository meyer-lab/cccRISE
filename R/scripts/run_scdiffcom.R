#' Run scDiffCom (CyrilLagger/scDiffCom, CRAN) on an exported dataset.
#'
#' scDiffCom is a two-condition differential CCC test with permutation-based
#' significance -- directly relevant to Reviewer #1 comment 6/7 (no null
#' model / no differential-CCC baseline in the original submission).
#'
#' Usage: Rscript run_scdiffcom.R <data_dir> <group_by_column> <condition_column> <cond1> <cond2> <out_dir> [iterations]
#' Example: Rscript run_scdiffcom.R ../data/balf_covid celltype condition Control "Severe COVID-19" ../output/scdiffcom 1000

suppressPackageStartupMessages({
  library(scDiffCom)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[[1]]
group_by <- args[[2]]
condition_col <- args[[3]]
cond1 <- args[[4]]
cond2 <- args[[5]]
out_dir <- args[[6]]
iterations <- if (length(args) >= 7) as.integer(args[[7]]) else 1000L
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("lib_load_data.R")

seurat_obj <- load_exported_seurat(data_dir)
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- subset(seurat_obj, cells = colnames(seurat_obj)[
  seurat_obj[[condition_col, drop = TRUE]] %in% c(cond1, cond2)
])

scdiffcom_obj <- run_interaction_analysis(
  seurat_object = seurat_obj,
  LRI_species = "human",
  seurat_celltype_id = group_by,
  seurat_condition_id = list(
    column_name = condition_col,
    cond1_name = cond1,
    cond2_name = cond2
  ),
  iterations = iterations
)

cci_detected <- GetTableCCI(scdiffcom_obj, type = "detected", simplified = TRUE)
write.csv(cci_detected, file.path(out_dir, "cci_detected.csv"), row.names = FALSE)

cci_all <- GetTableCCI(scdiffcom_obj, type = "raw", simplified = TRUE)
write.csv(cci_all, file.path(out_dir, "cci_all.csv"), row.names = FALSE)

saveRDS(scdiffcom_obj, file.path(out_dir, "scdiffcom_object.rds"))
cat("scDiffCom done ->", out_dir, "\n")
