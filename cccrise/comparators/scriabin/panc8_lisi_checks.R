# Why did PR #275's Fig. 2e (panel A) sit above the paper's? Reruns the LISI
# arm of panc8_noise.R on one C1 subsample, changing one thing at a time:
#
# - base:            PR #275's setting (CCA, 10,000 sampled GT pairs,
#                    Seurat's seeds); panc8_noise.R integrates all 90,000
# - gt_sct_seed<s>:  the GT CCIM's only random step, SCTransform's hard-coded
#                    gene subsample (seed.use 1448145 -> s), GT side only
# - hardcoded<s>:    every hard-coded seed R can reach at once: SCTransform
#                    (GT and downsampled) and RunPCA (42 -> s). RunCCA's
#                    set.seed(42) is set too but is inert here: above 10^6
#                    cross-product entries (>= 1,000 pairs a side) Seurat 5
#                    solves CCA with RSpectra::svds, whose start vector comes
#                    from Spectra's own generator at seed 0 (C++), not R's
# - cca_<n>:         the cap on GT pairs entering integration (n pairs;
#                    90,000 = every pair of 300 cells, the paper's setting
#                    and panc8_noise.R's).
#                    Seurat 5's matrix-free RSpectra CCA makes all 90,000
#                    feasible (Seurat 4, the paper's, formed the dense
#                    90,000 x 90,000 cross-product)
# - none_<n>:        no integration: the two CCIMs merged, PCA, LISI
#
# The global-RNG draws (cells, downsampling, ALRA, pairs) are the ones
# panc8_noise.R makes; its 12 subsamples are already 12 independent
# global draws. Per setting and arm, writes each GT pair's LISI to
# <out_dir>/sub<index>/<arm>__<setting>.rds (resumable). `settings`: a comma
# list of the settings to run (default: all).
#
#   Rscript cccrise/comparators/scriabin/panc8_lisi_checks.R <index> <out_dir> [settings]
source("cccrise/comparators/scriabin/panc8_steps.R")
args <- commandArgs(TRUE)
index <- as.integer(args[1]); out <- file.path(args[2], sprintf("sub%02d", index))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
ALT_SEEDS <- c(1, 2)
CAPS <- c(2500, 5000, 20000, 90000)
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), sprintf("sub%02d", index), ..., "\n")

SETTINGS <- c(
  list(base = list()),
  setNames(lapply(ALT_SEEDS, function(s) list(gt_sct = s)), paste0("gt_sct_seed", ALT_SEEDS)),
  setNames(lapply(ALT_SEEDS, function(s) list(gt_sct = s, down_sct = s, pca = s, cca = s)),
           paste0("hardcoded", ALT_SEEDS)),
  setNames(lapply(CAPS, function(n) list(cap = n)), paste0("cca_", CAPS)),
  setNames(lapply(c(10000, 90000), function(n) list(cap = n, reduction = "none")),
           paste0("none_", c(10000, 90000)))
)
if (length(args) > 2) SETTINGS <- SETTINGS[strsplit(args[3], ",")[[1]]]
opt <- function(s, key, default) if (is.null(s[[key]])) default else s[[key]]

sub <- c1_subsample(index)
down1 <- downsample(sub$counts, sub$prop, index, 1)
prep_cache <- list()
prepared <- function(which, alra, sct_seed) {
  key <- paste(which, alra, sct_seed)
  if (is.null(prep_cache[[key]])) {
    m <- if (which == "gt") sub$counts else down1
    prep_cache[[key]] <<- preprocess(m, sub$meta, index, alra = alra, sct_seed = sct_seed)
  }
  prep_cache[[key]]
}
ccim_cache <- list()
ccim_of <- function(which, alra, sct_seed, label) {
  key <- paste(which, alra, sct_seed)
  if (is.null(ccim_cache[[key]])) ccim_cache[[key]] <<- ccim(prepared(which, alra, sct_seed), label)
  ccim_cache[[key]]
}

for (arm in c("down1", "down1_alra")) {
  for (name in names(SETTINGS)) {
    file <- file.path(out, sprintf("%s__%s.rds", arm, name))
    if (file.exists(file)) next
    s <- SETTINGS[[name]]
    cap <- opt(s, "cap", 10000); reduction <- opt(s, "reduction", "cca")
    gt <- ccim_of("gt", FALSE, opt(s, "gt_sct", 1448145), "GT")
    dn <- ccim_of("down1", arm == "down1_alra", opt(s, "down_sct", 1448145), arm)
    pairs <- sort(head(pair_order(colnames(gt), index), cap))
    started <- Sys.time()
    lisi <- if (reduction == "none") {
      unintegrated(gt[, pairs], dn[, pairs], pca_seed = opt(s, "pca", 42))
    } else {
      integrate(gt[, pairs], dn[, pairs], pca_seed = opt(s, "pca", 42), cca_seed = opt(s, "cca", 42))
    }
    minutes <- as.numeric(difftime(Sys.time(), started, units = "mins"))
    save_atomic(list(index = index, arm = arm, setting = name, cap = length(pairs),
                     reduction = reduction, lisi = lisi, minutes = minutes), file)
    log(arm, name, "pairs", length(pairs), "frac>1", round(mean(lisi > 1 + 1e-6), 4),
        "mean", round(mean(lisi), 3), sprintf("(%.1f min)", minutes))
  }
}
log("done")
