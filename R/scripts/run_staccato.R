#' Run STACCato (daiqile96/STACCato) on an exported dataset.
#'
#' STACCato's core `staccato()` Tucker-decomposes a 4D communication tensor
#' (sample x LR-pair x sender-celltype x receiver-celltype) with an optional
#' sample-level covariate. The upstream package does not export a tensor
#' builder (its tutorial builds one externally via LIANA), so this script
#' builds a simple pseudobulk product tensor -- mean ligand expression in the
#' sender cell type times mean receptor expression in the receiver cell type,
#' per sample -- from the same shared LR table used by run_cellchat.R.
#'
#' NB: `staccato()` is defined but not exported from the installed package
#' (an upstream NAMESPACE bug -- see R/README.md); we call it via `:::`.
#'
#' Usage: Rscript run_staccato.R <data_dir> <group_by_column> <condition_column> <out_dir> [core_rank]
#' Example: Rscript run_staccato.R ../data/balf_covid celltype condition ../output/staccato 3,10,5,5

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(rTensor)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[[1]]
group_by <- args[[2]]
condition_col <- args[[3]]
out_dir <- args[[4]]
core_shape <- if (length(args) >= 5) as.integer(strsplit(args[[5]], ",")[[1]]) else NULL
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("lib_load_data.R")

seurat_obj <- load_exported_seurat(data_dir)
seurat_obj <- NormalizeData(seurat_obj)
lr <- simple_lr_pairs(load_lr_pairs(data_dir))

expr <- GetAssayData(seurat_obj, assay = "RNA", layer = "data")
meta <- seurat_obj[[]]

samples <- sort(unique(meta$sample))
celltypes <- sort(unique(meta[[group_by]]))
lr <- lr[lr$ligand_symbol %in% rownames(expr) & lr$receptor_symbol %in% rownames(expr), ]
stopifnot(nrow(lr) > 0)

cat(sprintf(
  "Building tensor: %d samples x %d LR pairs x %d senders x %d receivers\n",
  length(samples), nrow(lr), length(celltypes), length(celltypes)
))

# Mean expression per gene, per (sample, celltype) pseudobulk group
group_key <- paste(meta$sample, meta[[group_by]], sep = "___")
group_means <- sapply(split(seq_len(ncol(expr)), group_key), function(idx) {
  Matrix::rowMeans(expr[, idx, drop = FALSE])
})  # genes x (sample___celltype)

get_mean <- function(gene, sample, celltype) {
  key <- paste(sample, celltype, sep = "___")
  if (!(key %in% colnames(group_means)) || !(gene %in% rownames(group_means))) return(0)
  group_means[gene, key]
}

tsr <- array(
  0,
  dim = c(length(samples), nrow(lr), length(celltypes), length(celltypes)),
  dimnames = list(samples, lr$ligand_symbol, celltypes, celltypes)
)
for (si in seq_along(samples)) {
  for (ci in seq_along(celltypes)) {
    lig_vec <- sapply(lr$ligand_symbol, get_mean, sample = samples[si], celltype = celltypes[ci])
    for (cj in seq_along(celltypes)) {
      rec_vec <- sapply(lr$receptor_symbol, get_mean, sample = samples[si], celltype = celltypes[cj])
      tsr[si, , ci, cj] <- lig_vec * rec_vec
    }
  }
}
tsr <- as.tensor(tsr)

# Sample-level covariate: one numeric column per non-reference condition level
cond_by_sample <- meta[!duplicated(meta$sample), c("sample", condition_col)]
cond_by_sample <- cond_by_sample[match(samples, cond_by_sample$sample), ]
X_covar1 <- model.matrix(~cond_by_sample[[condition_col]])
colnames(X_covar1)[1] <- "Intercept"

if (is.null(core_shape)) {
  core_shape <- pmin(c(ncol(X_covar1), 5L, 5L, 5L), dim(tsr))
}
cat("Core (Tucker) rank:", paste(core_shape, collapse = "x"), "\n")

fit <- STACCato:::staccato(
  tsr = tsr,
  X_covar1 = X_covar1,
  lr.names = lr$ligand_symbol,
  sender.names = celltypes,
  receiver.names = celltypes,
  core_shape = core_shape
)

write.csv(as.data.frame(fit$W$W1), file.path(out_dir, "covariate_factor_W1.csv"))
write.csv(as.data.frame(fit$W$W2), file.path(out_dir, "lr_factor_W2.csv"))
write.csv(as.data.frame(fit$W$W3), file.path(out_dir, "sender_factor_W3.csv"))
write.csv(as.data.frame(fit$W$W4), file.path(out_dir, "receiver_factor_W4.csv"))
saveRDS(fit, file.path(out_dir, "staccato_fit.rds"))
cat("STACCato done ->", out_dir, "\n")
