# Panel B of figureR_scriabin, the read-out the Scriabin paper does not make:
# whether the same interaction programs come back on the same C1 subsample
# (panc8_noise.R's subsample `index`: same seeds, so the same cells and the
# same draw 1). From PR #275's rcheck/authors/panc8_paper_noise.R.
#
# - Arms: GT (full depth) and two independent downsampling draws to inDrop
#   coverage, each raw and ALRA-denoised (panc8_steps.R's `arms`).
# - Programs: upstream InteractionPrograms at its defaults on each arm. At
#   300 cells it uses the full CCIM (no cell subsampling), so the programs of
#   two draws differ only through the downsampling draw.
# - Then upstream's significance test of each arm's programs in its own
#   similarity matrix, as the vignette applies it (significance.R; one
#   sample, so a program is kept if upstream keeps it with p < 0.05), its
#   random programs seeded set.seed(index).
#
# Writes <out_dir>/sub<index>.rds: per arm, the LR pair count, the programs
# and the kept programs with their p.
#   Rscript cccrise/comparators/scriabin/panc8_programs.R <index> <out_dir>
source("cccrise/comparators/scriabin/panc8_steps.R")
source("cccrise/comparators/scriabin/significance.R")
args <- commandArgs(TRUE)
index <- as.integer(args[1]); out <- args[2]
file <- file.path(out, sprintf("sub%02d.rds", index))
if (file.exists(file)) quit(save = "no")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), sprintf("sub%02d", index), ..., "\n")

sub <- c1_subsample(index)
data <- arms(sub, index, draws = 1:2)
log("preprocessed")

programs <- lapply(names(data), function(arm) {
  obj <- data[[arm]]
  ip <- suppressWarnings(suppressMessages(InteractionPrograms(
    obj, assay = assay_of(obj), slot = "data", cell_types = "celltype", return.mat = TRUE)))
  kept <- significant_programs(ip, seed = index)
  kept <- kept[kept$pval < 0.05, ]
  log(arm, length(ip$modules), "programs,", nrow(kept), "significant")
  list(n_lr = ncol(ip$cor_mat), modules = ip$modules, kept = kept$name, pval = kept$pval)
})
names(programs) <- names(data)
save_atomic(list(index = index, cells = sub$cells, programs = programs), file)
log("done")
