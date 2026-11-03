# Upstream's multi-sample steps on per-sample InteractionPrograms results
# (`ip_runs.R programs`), each code path unchanged:
#
#   merge:        FindAllInteractionPrograms itself, with InteractionPrograms
#                 replaced by a loader of the cached per-sample result (its
#                 merge loop is the code under test). -> merged.json
#   significance: InteractionProgramSignificance (n.replicate = 10000, its
#                 default and the paper's) once per sample, in parallel. Upstream
#                 keeps a program significant in >= 1 sample, so the union of
#                 the per-sample calls is its multi-sample output.
#                 -> significance/<sample>.rds, kept.json
#   score:        ScoreInteractionPrograms on the full denoised matrix.
#                 -> scores_ligands.csv, scores_receptors.csv
#
#   Rscript merge_sig.R merge <progdir>
#   Rscript merge_sig.R significance <progdir> [workers] [n.replicate]
#   Rscript merge_sig.R score <progdir> <full.rds> <meta.rds>
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(ade4); library(pbapply); library(stringr); library(jsonlite)
  library(parallel)
})
pboptions(type = "none")
`%notin%` <- Negate(`%in%`)
source("rcheck/upstream/scriabin/R/interaction_programs.R")
args <- commandArgs(TRUE)
mode <- args[1]; dir <- args[2]
log <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
samples <- sub("\\.rds$", "", list.files(dir, pattern = "\\.rds$"))
load_ip <- function(s) readRDS(file.path(dir, paste0(s, ".rds")))

if (mode == "merge") {
  # A light two-row object; the cached loader ignores its expression
  cells <- unlist(lapply(samples, function(s) load_ip(s)$cells))
  smp <- rep(samples, sapply(samples, function(s) length(load_ip(s)$cells)))
  light <- CreateSeuratObject(counts = Matrix(1, 2, length(cells), sparse = TRUE,
                              dimnames = list(c("g1", "g2"), cells)),
                              meta.data = data.frame(sample = smp, row.names = cells))
  InteractionPrograms <- function(object, ...) {
    ip <- load_ip(unique(object$sample))
    list(cor_mat = ip$cor_mat, tom = NULL, modules = ip$modules, connectivity = ip$connectivity)
  }
  t0 <- Sys.time()
  ip_all <- FindAllInteractionPrograms(light, group.by = "sample")
  log(length(ip_all$modules), "merged programs from",
      sum(sapply(samples, function(s) length(load_ip(s)$modules))), "modules",
      format(round(difftime(Sys.time(), t0, units = "secs"))))
  write_json(ip_all$modules, file.path(dir, "merged.json"))
} else if (mode == "significance") {
  workers <- if (length(args) >= 3) as.integer(args[3]) else 6
  n_rep <- if (length(args) >= 4) as.integer(args[4]) else 10000
  merged <- lapply(fromJSON(file.path(dir, "merged.json"), simplifyVector = FALSE), unlist)
  sig <- file.path(dir, "significance"); dir.create(sig, showWarnings = FALSE)
  todo <- samples[!file.exists(file.path(sig, paste0(samples, ".rds")))]
  log(length(todo), "samples to test,", length(merged), "programs,", n_rep, "replicates")
  res <- mclapply(todo, function(s) {
    t0 <- Sys.time()
    ip <- load_ip(s)
    ip_s <- list(cor_mat = setNames(list(ip$cor_mat), s), tom = setNames(list(NULL), s),
                 modules = merged, connectivity = setNames(list(ip$connectivity), s))
    df <- tryCatch(InteractionProgramSignificance(ip_s, n.replicate = n_rep),
                   error = function(e) e)
    # no program significant in this sample: upstream's data.frame step errors
    if (inherits(df, "error")) df <- data.frame(name = character(), lr_pair = character())
    saveRDS(df, file.path(sig, paste0(s, ".rds")))
    paste(s, length(unique(df$name)), "significant,",
          format(round(difftime(Sys.time(), t0, units = "secs"))))
  }, mc.cores = workers, mc.preschedule = FALSE)
  for (r in res) log(r)
  if (all(file.exists(file.path(sig, paste0(samples, ".rds"))))) {
    dfs <- lapply(samples, function(s) readRDS(file.path(sig, paste0(s, ".rds"))))
    keep <- unique(unlist(lapply(dfs, function(d) unique(as.character(d$name)))))
    keep <- names(merged)[names(merged) %in% keep]
    write_json(merged[keep], file.path(dir, "kept.json"))
    sig_tab <- sapply(dfs, function(d) names(merged) %in% d$name)
    dimnames(sig_tab) <- list(names(merged), samples)
    write.csv(sig_tab, file.path(dir, "significant_in.csv"))
    log(length(keep), "of", length(merged), "programs kept")
  }
} else if (mode == "score") {
  full <- readRDS(args[3]); meta <- readRDS(args[4])
  kept <- lapply(fromJSON(file.path(dir, "kept.json"), simplifyVector = FALSE), unlist)
  mods <- data.frame(name = rep(names(kept), lengths(kept)), lr_pair = unlist(kept))
  seu <- CreateSeuratObject(counts = full, meta.data = meta, assay = "dn")
  seu <- SetAssayData(seu, layer = "data", new.data = full)
  rm(full); invisible(gc())
  seu <- ScoreInteractionPrograms(seu, mods)
  write.csv(t(as.matrix(GetAssayData(seu, assay = "IPligands", layer = "data"))),
            file.path(dir, "scores_ligands.csv"))
  write.csv(t(as.matrix(GetAssayData(seu, assay = "IPreceptors", layer = "data"))),
            file.path(dir, "scores_receptors.csv"))
  log("scored", ncol(seu), "cells on", length(kept), "programs")
}
