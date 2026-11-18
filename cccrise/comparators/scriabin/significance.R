# Upstream Scriabin's significance test of interaction programs
# (InteractionProgramSignificance, sourced by upstream.R), applied as the
# vignette (interaction-programs.Rmd) and the paper's Methods apply it before
# programs are used. Each program's similarities are tested against
# n.replicate random programs of its size (one-sided Mann-Whitney); a program
# failing more than 5% of the tests in every sample is dropped, as is one
# with only one distinct ligand or receptor. The vignette then keeps the
# programs whose smallest p is below 0.05.
#
# significant_programs(ip, seed): the programs of `ip` (one sample's
# similarity matrix, programs and connectivity) that upstream keeps, with
# their p (the share of failed tests), at n.replicate = 500, the vignette's
# (upstream's default and the paper's Methods use 10,000, ~20 times the
# compute: ~110 CPU-hours for S8A; at 300 cells, 500 and 10,000 keep almost
# the same programs). Upstream draws its random programs from the
# global RNG without seeding it; here set.seed(seed) comes first, so reruns
# reproduce. Upstream errors when it keeps no program (ldply of an empty
# list); that is returned as none kept.
#
# As a script (Appendix Figure S8A): one seed's merged programs,
# <seed_dir>/merged.json (`cccrise-scriabin-seeds --stage merge`), tested in
# each merged sample's similarity matrix (<seed_dir>/<sample>.rds,
# programs.R), one checkpoint per sample in
# <seed_dir>/significance/<sample>.json. Upstream computes each sample's p
# on its own (an lapply over samples) and drops a program only if every
# sample fails it, so testing the samples one at a time, in parallel, and
# keeping a program if any sample keeps it with p < 0.05 (the runner) is
# upstream's multi-sample test followed by the vignette's filter.
#   Rscript cccrise/comparators/scriabin/significance.R <seed_dir> <seed> [workers]
suppressPackageStartupMessages(library(stringr))  # str_split, unqualified upstream

significant_programs <- function(ip, seed, n.replicate = 500) {
  set.seed(seed)
  sig <- tryCatch(
    suppressWarnings(suppressMessages(InteractionProgramSignificance(ip, n.replicate = n.replicate))),
    error = function(e) if (grepl("id variables not found", conditionMessage(e))) NULL else stop(e))
  if (is.null(sig)) return(data.frame(name = character(0), pval = numeric(0)))
  kept <- unique(sig[, c("name", "sample_pval")])
  data.frame(name = kept$name, pval = kept$sample_pval)[order(kept$name), ]
}

if (sys.nframe() == 0) {
  suppressPackageStartupMessages({
    library(Seurat); library(dplyr); library(tibble); library(purrr); library(pbapply)
    library(WGCNA); library(jsonlite); library(parallel)
  })
  pboptions(type = "none")
  source("cccrise/comparators/scriabin/upstream.R")
  args <- commandArgs(TRUE)
  dir <- args[1]; seed <- as.integer(args[2])
  workers <- if (length(args) >= 3) as.integer(args[3]) else 6
  log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
  merged <- fromJSON(file.path(dir, "merged.json"), simplifyVector = FALSE)
  programs <- lapply(merged$programs, unlist)
  samples <- unlist(merged$samples)
  out <- file.path(dir, "significance")
  dir.create(out, showWarnings = FALSE)
  todo <- samples[!file.exists(file.path(out, paste0(samples, ".json")))]
  log(length(todo), "of", length(samples), "samples to test;", length(programs), "programs; seed", seed)
  res <- mclapply(todo, function(s) {
    t0 <- Sys.time()
    x <- readRDS(file.path(dir, paste0(s, ".rds")))
    kept <- significant_programs(list(cor_mat = x$cor_mat, tom = NULL, modules = programs,
                                      connectivity = x$connectivity), seed)
    file <- file.path(out, paste0(s, ".json"))
    write_json(kept, paste0(file, ".partial"), dataframe = "rows", digits = NA)
    file.rename(paste0(file, ".partial"), file)
    paste(s, nrow(kept), "of", length(programs), "kept,",
          format(round(difftime(Sys.time(), t0, units = "mins"), 1)))
  }, mc.cores = workers, mc.preschedule = FALSE)
  for (r in res) log(if (inherits(r, "try-error")) paste("ERROR", r) else r)
}
