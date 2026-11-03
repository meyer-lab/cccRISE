# NICHES N2 / N4 in the R-shared arm (CONCORDANCE_PLAN.md §5.2): upstream
# RunNICHES (pinned) per sample on cccRISE's inputs (lr_data.rds: the LR genes
# and min-aggregated complex pseudo-genes, so the LR database lists them as
# single features), SystemToCell and CellToSystem (blend mean, raw output),
# run twice per sample: N2 asks whether the two runs are identical.
# One checkpoint per sample: <out>/<sample>.rds (s2c, c2s: LR x cell sparse;
# identical: logical), written atomically.
#   Rscript rcheck/shared/niches_runs.R <data.rds> <meta.rds> <lr.csv> <out> [workers]
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(parallel)
})
up <- "rcheck/upstream"
for (f in list.files(file.path(up, "niches/R"), full.names = TRUE)) source(f)
args <- commandArgs(TRUE)
data <- readRDS(args[1]); meta <- readRDS(args[2]); lr <- read.csv(args[3]); out <- args[4]
workers <- if (length(args) >= 5) as.integer(args[5]) else 4
dir.create(out, showWarnings = FALSE, recursive = TRUE)
db <- data.frame(lig = lr$ligand, rec = lr$receptor)
run <- function(obj) {
  res <- suppressWarnings(suppressMessages(RunNICHES(
    obj, assay = "IP", LR.database = "custom", custom_LR_database = db, species = "human",
    cell_types = "celltype", meta.data.to.map = NULL, min.cells.per.ident = 1,
    CellToCell = FALSE, SystemToCell = TRUE, CellToSystem = TRUE, blend = "mean",
    output_format = "raw")))
  list(s2c = as(res$SystemToCell$SystemToCellMatrix, "CsparseMatrix"),
       c2s = as(res$CellToSystem$CellToSystemMatrix, "CsparseMatrix"))
}
samples <- sort(unique(meta$sample))
todo <- samples[!file.exists(file.path(out, paste0(samples, ".rds")))]
cat(format(Sys.time(), "%H:%M:%S"), length(todo), "of", length(samples), "samples\n")
res <- mclapply(todo, function(s) {
  t0 <- Sys.time()
  cells <- rownames(meta)[meta$sample == s]
  m <- data[, cells, drop = FALSE]
  obj <- CreateSeuratObject(counts = m, meta.data = meta[cells, , drop = FALSE], assay = "IP")
  obj <- SetAssayData(obj, layer = "data", new.data = m)
  a <- tryCatch(run(obj), error = function(e) e)
  if (inherits(a, "error")) return(paste(s, "ERROR", conditionMessage(a)))
  b <- run(obj)
  same <- identical(a$s2c, b$s2c) && identical(a$c2s, b$c2s)
  partial <- file.path(out, paste0(s, ".rds.partial"))
  saveRDS(c(a, list(identical = same)), partial)
  file.rename(partial, file.path(out, paste0(s, ".rds")))
  paste(s, length(cells), "cells, identical:", same,
        format(round(difftime(Sys.time(), t0, units = "secs"))))
}, mc.cores = workers, mc.preschedule = FALSE)
for (r in res) cat(r, "\n")
