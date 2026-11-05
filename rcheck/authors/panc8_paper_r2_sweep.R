# The Scriabin paper's soft-power analysis (Wilk et al. 2024, Extended Data
# Fig. 8f; vignettes/fig_s11_ip.png upstream): "The interaction program
# discovery workflow was repeated on 35 random subsamples of the inDrop panc8
# dataset, using 19 different R^2 thresholds", read out as the recommended
# softPower, % programs failing significance, % programs of only one ligand or
# receptor, and program size. One subsample per call:
#
# 1. `n_cells` random inDrop cells (set.seed(index)), the vignette's
#    preprocessing (panc8_paper_prep.R), upstream InteractionPrograms with its
#    defaults and cell_types = "celltype" -> the LR similarity matrix.
# 2. For R^2 = 0.05, 0.10, ..., 0.95: WGCNA's recommended power, and the power
#    used: "uncapped" (recommended; 1 if none), as the paper's panel shows
#    powers up to 7, and "capped" (upstream today: 1 if none or above 3).
# 3. For each distinct power: modules (upstream's step) and each module's
#    significance at the Methods' 10,000 replicates.
#
# Not stated in the paper, so chosen here: the subsample size (n_cells), and
# random (not cell-type-stratified) subsampling.
#
#   Rscript rcheck/authors/panc8_paper_r2_sweep.R <n_cells> <index> <out_dir> [n_replicate]
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(qlcMatrix); library(pbapply)
})
pboptions(type = "none")
`%notin%` <- Negate(`%in%`)
source("rcheck/upstream_fixes.R"); source_upstream_ip()
source("rcheck/authors/_ip_steps.R")
args <- commandArgs(TRUE)
n_cells <- as.integer(args[1]); index <- as.integer(args[2]); out <- args[3]
n_rep <- if (length(args) >= 4) as.integer(args[4]) else 10000L
file <- file.path(out, sprintf("n%d_sub%02d.rds", n_cells, index))
if (file.exists(file)) quit(save = "no")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
CUTS <- seq(0.05, 0.95, by = 0.05)
t0 <- Sys.time()

src <- "singlecell_cache/authors/panc8_paper"
data <- readRDS(file.path(src, "indrop_alra.rds")); meta <- readRDS(file.path(src, "indrop_meta.rds"))
set.seed(index)
cells <- sort(sample(colnames(data), n_cells))
if (n_cells > 300) {  # upstream's per-type sample_n errors on types under min.cell = 3
  tt <- table(meta[cells, "celltype"])
  cells <- cells[meta[cells, "celltype"] %in% names(tt)[tt >= 3]]
}
obj <- CreateSeuratObject(counts = data[, cells], meta.data = meta[cells, , drop = FALSE], assay = "IP")
obj <- SetAssayData(obj, layer = "data", new.data = data[, cells])
rm(data); invisible(gc())

ip <- suppressWarnings(suppressMessages(InteractionPrograms(
  obj, assay = "IP", slot = "data", cell_types = "celltype", return.mat = TRUE, softPower = 1)))
m_cor <- ip$cor_mat
# the copied module step must give upstream's modules exactly
stopifnot(identical(unname(lapply(modules_at(m_cor, 1), sort)), unname(lapply(ip$modules, sort))))

est <- power_estimates(m_cor, CUTS)
powers <- data.frame(r2 = CUTS, recommended = est,
                     uncapped = ifelse(is.na(est), 1, est),
                     capped = ifelse(is.na(est) | est > 3, 1, est))
mods <- list(); fail <- list()
for (p in sort(unique(c(powers$uncapped, powers$capped)))) {
  key <- as.character(p)
  mods[[key]] <- if (p == 1) ip$modules else modules_at(m_cor, p)
  set.seed(index * 100 + p)
  fail[[key]] <- vapply(mods[[key]], function(m) module_failure(m_cor, m, n_rep), numeric(1))
}
save_atomic(list(n_cells = n_cells, index = index, cells = cells, n_lr = ncol(m_cor),
                 powers = powers, modules = mods, failure = fail, n_replicate = n_rep,
                 seconds = as.numeric(difftime(Sys.time(), t0, units = "secs"))), file)
cat(sprintf("n%d sub%02d: %d cells, %d LR, powers %s, %.0f s\n", n_cells, index, length(cells),
            ncol(m_cor), paste(names(mods), collapse = ","),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))
