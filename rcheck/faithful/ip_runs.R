# Upstream InteractionPrograms per sample on the faithful inputs
# (`preprocess.R`), with every argument at its default except those the
# vignette's comparative example sets (cell_types; assay = the denoised one).
# FindAllInteractionPrograms calls InteractionPrograms on each sample of a
# SplitObject and only then merges, so running samples one at a time (in
# parallel, one checkpoint per sample) gives the same per-sample results;
# `merge_sig.R` then runs upstream's merge unchanged on them.
#
# Only deviation: in samples that will be subsampled (> iterate.threshold =
# 300 cells), cells of a type with fewer than min.cell = 3 cells are dropped
# first, because upstream's `sample_n` errors on them.
#
#   programs: Rscript ip_runs.R programs <data.rds> <meta.rds> <out> <seed> [frac] [workers]
#             frac < 1 first keeps that fraction of each sample's cells (set.seed(seed))
#   drop:     Rscript ip_runs.R drop <data.rds> <meta.rds> <out> <sample> <idx> [<idx> ...]
#             full CCIM (no subsampling), all cells and each 0-based cell left out
#
# Optional environment (unset: the faithful arm, OmniPath via the stub):
#   IP_LR_CSV    an LR table (ligand, receptor columns) passed as the custom
#                database: the R-shared arm, on cccRISE's LR pairs
#                (rcheck/shared/export_dataset.py)
#   IP_SETTINGS  JSON of module-discovery settings for rcheck/patched_ip.R:
#                r2_cutoff, max_soft_power (null = no cap), min_size,
#                tree_cut_quantile, n_iterate (S7)
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(WGCNA); library(flashClust); library(dynamicTreeCut)
  library(qlcMatrix); library(pbapply); library(jsonlite); library(parallel)
})
pboptions(type = "none")
`%notin%` <- Negate(`%in%`)
source("rcheck/upstream/scriabin/R/interaction_programs.R")
lr_args <- list()
if (nzchar(Sys.getenv("IP_LR_CSV"))) {
  lr_tab <- read.csv(Sys.getenv("IP_LR_CSV"))
  lr_args <- list(database = "custom", ligands = lr_tab$ligand, recepts = lr_tab$receptor)
}
cfg <- list(r2_cutoff = 0.6, max_soft_power = 3)
if (nzchar(Sys.getenv("IP_SETTINGS"))) {
  source("rcheck/patched_ip.R")
  cfg <- modifyList(cfg, fromJSON(Sys.getenv("IP_SETTINGS")))
  if (is.null(cfg$max_soft_power)) cfg$max_soft_power <- Inf
  lr_args <- c(lr_args, Filter(Negate(is.null), list(
    r2_cutoff = cfg$r2_cutoff, max.soft.power = cfg$max_soft_power, min.size = cfg$min_size,
    tree.cut.quantile = cfg$tree_cut_quantile, n.iterate = cfg$n_iterate)))
}
args <- commandArgs(TRUE)
mode <- args[1]; data <- readRDS(args[2]); meta <- readRDS(args[3]); out <- args[4]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

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

run_ip <- function(cells, seed, ...) {
  suppressWarnings(suppressMessages(do.call(InteractionPrograms, c(list(
    sample_object(cells), assay = "IP", slot = "data", cell_types = "celltype",
    return.mat = TRUE, seed = seed), lr_args, list(...)))))
}

save_ip <- function(ip, cells, file) {
  # TOM dropped: significance testing and the merge never read it
  saveRDS(list(cor_mat = ip$cor_mat, modules = ip$modules,
               connectivity = ip$connectivity, cells = cells), file)
  # the soft power upstream picked (it is not returned): same call, same rule
  sp <- suppressWarnings(suppressMessages(
    pickSoftThreshold.fromSimilarity(ip$cor_mat, RsquaredCut = cfg$r2_cutoff, verbose = 0)))
  est <- sp$powerEstimate
  write_json(list(n_cells = length(cells), n_lr = nrow(ip$cor_mat),
                  power_estimate = est,
                  soft_power = if (is.na(est) || est > cfg$max_soft_power) 1 else est,
                  max_r2_power_le3 = max(sp$fitIndices$SFT.R.sq[sp$fitIndices$Power <= 3]),
                  modules = ip$modules),
             sub("\\.rds$", ".json", file), auto_unbox = TRUE, na = "null")
}

if (mode == "programs") {
  seed <- as.integer(args[5])
  frac <- if (length(args) >= 6) as.numeric(args[6]) else 1
  workers <- if (length(args) >= 7) as.integer(args[7]) else 6
  samples <- sort(unique(meta$sample))
  todo <- samples[!file.exists(file.path(out, paste0(samples, ".rds")))]
  log(length(todo), "of", length(samples), "samples to run; seed", seed, "frac", frac)
  res <- mclapply(todo, function(s) {
    t0 <- Sys.time()
    cells <- rownames(meta)[meta$sample == s]
    if (frac < 1) {
      set.seed(seed)
      cells <- sort(sample(cells, round(frac * length(cells))))
    }
    cells <- drop_small_types(cells)
    ip <- tryCatch(run_ip(cells, seed), error = function(e) e)
    if (inherits(ip, "error")) return(paste(s, "ERROR", conditionMessage(ip)))
    save_ip(ip, cells, file.path(out, paste0(s, ".rds")))
    paste(s, length(cells), "cells,", length(ip$modules), "modules,",
          format(round(difftime(Sys.time(), t0, units = "secs"))))
  }, mc.cores = workers, mc.preschedule = FALSE)
  for (r in res) log(r)
} else if (mode == "drop") {
  s <- args[5]; drops <- as.integer(args[-(1:5)])
  cells_all <- rownames(meta)[meta$sample == s]
  for (d in c(-1L, drops)) {
    file <- file.path(out, sprintf("%s_drop%d.rds", s, d))
    if (file.exists(file)) next
    cells <- if (d >= 0) cells_all[-(d + 1)] else cells_all
    ip <- run_ip(cells, 1111, iterate.threshold = length(cells) + 1)
    save_ip(ip, cells, file)
    log(s, "drop", d, length(ip$modules), "modules")
  }
}
