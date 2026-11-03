# Convert export_dataset.py's MatrixMarket files to the .rds inputs that
# rcheck/faithful/ip_runs.R and merge_sig.R read: lr_data.rds (LR features x
# cells, dgCMatrix), full_data.rds (every gene, if exported) and meta.rds
# (rownames = cells; sample, celltype).
# Usage: Rscript rcheck/shared/to_rds.R <dir>
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages(library(Matrix))
dir <- commandArgs(TRUE)[1]
meta <- read.csv(file.path(dir, "meta.csv"))
rownames(meta) <- meta$cell
saveRDS(meta[, c("sample", "celltype")], file.path(dir, "meta.rds"))
conv <- function(mtx, feats, rds) {
  m <- as(readMM(file.path(dir, mtx)), "CsparseMatrix")
  dimnames(m) <- list(read.csv(file.path(dir, feats), header = FALSE)$V1, meta$cell)
  saveRDS(m, file.path(dir, rds))
  cat(rds, nrow(m), "x", ncol(m), "\n")
}
conv("lr_expr.mtx", "lr_features.csv", "lr_data.rds")
if (file.exists(file.path(dir, "expr.mtx"))) conv("expr.mtx", "features.csv", "full_data.rds")
