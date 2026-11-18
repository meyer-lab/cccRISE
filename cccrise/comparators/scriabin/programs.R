# Upstream Scriabin's interaction programs on cccRISE's inputs, per sample, at
# one seed (Appendix Figure S8A: `cccrise-scriabin-seeds` merges them,
# significance.R tests the merged programs in each sample, and the runner
# compares the significant ones between seeds). From PR #275's rcheck/faithful/ip_runs.R, its `programs`
# mode on the full cells. README.md has the R setup.
#
# InteractionPrograms (upstream.R) runs with every argument at its default
# except cell_types and the LR list: the custom database of cccRISE's LR pairs
# (lr.csv). FindAllInteractionPrograms calls InteractionPrograms on each
# sample of a SplitObject and only then merges, so running samples one at a
# time (in parallel, one checkpoint per sample) gives the same per-sample
# programs; the merge is `comparators.scriabin_programs` (upstream's, exactly).
#
# Random numbers: InteractionPrograms subsamples the cells of a sample above
# iterate.threshold = 300 cells after set.seed(seed), its `seed` argument
# (upstream's default 1111); smaller samples use the full cell-cell
# interaction matrix and are deterministic. The caller's set.seed never
# reaches it.
#
# Only deviation: in samples that will be subsampled, cells of a type with
# fewer than min.cell = 3 cells are dropped first, because upstream's
# `sample_n` errors on them.
#
# <in_dir>: export_inputs.py's output; writes <out>/<sample>.json (cell count,
# LR pair count, modules) and <sample>.rds (similarity matrix, modules,
# connectivity, cells; the checkpoint, and significance.R's input).
#   Rscript cccrise/comparators/scriabin/programs.R <in_dir> <out> <seed> [workers]
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(WGCNA); library(flashClust); library(dynamicTreeCut)
  library(qlcMatrix); library(pbapply); library(jsonlite); library(parallel)
})
pboptions(type = "none")
source("cccrise/comparators/scriabin/upstream.R")
args <- commandArgs(TRUE)
in_dir <- args[1]; out <- args[2]; seed <- as.integer(args[3])
workers <- if (length(args) >= 4) as.integer(args[4]) else 6
dir.create(out, showWarnings = FALSE, recursive = TRUE)
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

meta <- read.csv(file.path(in_dir, "meta.csv"), row.names = 1)
data <- as(readMM(file.path(in_dir, "lr_expr.mtx")), "CsparseMatrix")
dimnames(data) <- list(read.csv(file.path(in_dir, "lr_features.csv"), header = FALSE)$V1,
                       rownames(meta))
lr <- read.csv(file.path(in_dir, "lr.csv"))

sample_object <- function(cells) {
  m <- data[, cells, drop = FALSE]
  obj <- CreateSeuratObject(counts = m, meta.data = meta[cells, , drop = FALSE], assay = "IP")
  SetAssayData(obj, layer = "data", new.data = m)
}

drop_small_types <- function(cells) {
  if (length(cells) <= 300) return(cells)
  tt <- table(meta[cells, "celltype"])
  cells[meta[cells, "celltype"] %in% names(tt)[tt >= 3]]
}

save_programs <- function(ip, cells, file) {
  json <- sub("\\.rds$", ".json", file)
  write_json(list(n_cells = length(cells), n_lr = nrow(ip$cor_mat), modules = ip$modules),
             paste0(json, ".partial"), auto_unbox = TRUE, na = "null")
  # TOM dropped: the significance test and the merge never read it
  saveRDS(list(cor_mat = ip$cor_mat, modules = ip$modules, connectivity = ip$connectivity,
               cells = cells), paste0(file, ".partial"))
  # flush both before the .rds (the checkpoint) appears
  system2("sync", c("-d", paste0(json, ".partial"), paste0(file, ".partial")))
  file.rename(paste0(json, ".partial"), json)
  file.rename(paste0(file, ".partial"), file)
}

samples <- sort(unique(meta$sample))
todo <- samples[!file.exists(file.path(out, paste0(samples, ".rds")))]
log(length(todo), "of", length(samples), "samples to run; seed", seed)
res <- mclapply(todo, function(s) {
  t0 <- Sys.time()
  cells <- drop_small_types(rownames(meta)[meta$sample == s])
  ip <- tryCatch(suppressWarnings(suppressMessages(InteractionPrograms(
    sample_object(cells), assay = "IP", slot = "data", cell_types = "celltype",
    database = "custom", ligands = lr$ligand, recepts = lr$receptor,
    return.mat = TRUE, seed = seed))), error = function(e) e)
  if (inherits(ip, "error")) return(paste(s, "ERROR", conditionMessage(ip)))
  save_programs(ip, cells, file.path(out, paste0(s, ".rds")))
  paste(s, length(cells), "cells,", length(ip$modules), "modules,",
        format(round(difftime(Sys.time(), t0, units = "secs"))))
}, mc.cores = workers, mc.preschedule = FALSE)
for (r in res) log(r)
