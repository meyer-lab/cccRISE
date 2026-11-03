# NICHES N1 in the R-shared arm: upstream RunNICHES CellToCell (blend mean,
# raw) per sample, once per seed. set.seed(seed) before RunNICHES has no
# effect: PrepNodeObject's subset() calls Seurat's WhichCells(seed = 1),
# resetting the RNG, so upstream's pairing is one fixed draw. With RESEED=1
# the seed is set again just before RunCellToCell's sender / receiver draws
# (the only random step), to measure the draw's effect. One checkpoint per
# (sample, seed): <out>/<sample>_s<seed>.rds with the edge matrix (LR x edges,
# sparse) and the edges' sending / receiving cells and types.
#   Rscript rcheck/shared/niches_c2c.R <data.rds> <meta.rds> <lr.csv> <out> <workers> <seed> [<seed> ...]
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({ library(Seurat); library(Matrix); library(dplyr); library(parallel) })
for (f in list.files("rcheck/upstream/niches/R", full.names = TRUE)) source(f)
args <- commandArgs(TRUE)
data <- readRDS(args[1]); meta <- readRDS(args[2]); lr <- read.csv(args[3]); out <- args[4]
workers <- as.integer(args[5]); seeds <- as.integer(args[-(1:5)])
dir.create(out, showWarnings = FALSE, recursive = TRUE)
db <- data.frame(lig = lr$ligand, rec = lr$receptor)
.c2c_seed <- NULL
if (Sys.getenv("RESEED") == "1") {
  # seed right before the sender / receiver draws (RunCellToCell's own
  # subset() calls reset the RNG again); the pinned file text, patched once
  src <- readLines("rcheck/upstream/niches/R/RunCellToCell.R")
  at <- which(src == "  lig.data <- list()")
  stopifnot(length(at) == 1)
  src <- append(src, "  set.seed(.c2c_seed)", after = at - 1)
  eval(parse(text = src), envir = globalenv())
}
jobs <- expand.grid(sample = sort(unique(meta$sample)), seed = seeds, stringsAsFactors = FALSE)
jobs <- jobs[!file.exists(file.path(out, sprintf("%s_s%d.rds", jobs$sample, jobs$seed))), ]
res <- mclapply(seq_len(nrow(jobs)), function(k) {
  s <- jobs$sample[k]; seed <- jobs$seed[k]
  cells <- rownames(meta)[meta$sample == s]
  m <- data[, cells, drop = FALSE]
  obj <- CreateSeuratObject(counts = m, meta.data = meta[cells, , drop = FALSE], assay = "IP")
  obj <- SetAssayData(obj, layer = "data", new.data = m)
  set.seed(seed); .c2c_seed <<- seed
  r <- tryCatch(suppressWarnings(suppressMessages(RunNICHES(
    obj, assay = "IP", LR.database = "custom", custom_LR_database = db, species = "human",
    cell_types = "celltype", meta.data.to.map = NULL, min.cells.per.ident = 1,
    CellToCell = TRUE, SystemToCell = FALSE, CellToSystem = FALSE, blend = "mean",
    output_format = "raw"))), error = function(e) e)
  if (inherits(r, "error")) return(paste(s, seed, "ERROR", conditionMessage(r)))
  md <- r$CellToCell$metadata
  partial <- file.path(out, sprintf("%s_s%d.rds.partial", s, seed))
  saveRDS(list(m = as(r$CellToCell$CellToCellMatrix, "CsparseMatrix"),
               edges = md[, intersect(c("SendingCell", "ReceivingCell", "SendingType", "ReceivingType"), colnames(md))]),
          partial)
  file.rename(partial, file.path(out, sprintf("%s_s%d.rds", s, seed)))
  paste(s, seed, ncol(r$CellToCell$CellToCellMatrix), "edges")
}, mc.cores = workers, mc.preschedule = FALSE)
for (x in res) cat(x, "\n")
