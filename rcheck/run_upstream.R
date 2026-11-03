# Run the pinned upstream Scriabin / NICHES R code on the inputs written by
# rcheck/export_inputs.py, with Python's random draws fed in through shims.
#
# Usage:
#   Rscript rcheck/run_upstream.R <dir> programs      # Scriabin IPs + merge, NICHES modes
#   Rscript rcheck/run_upstream.R <dir> significance  # needs <dir>/random_sets.json
#   Rscript rcheck/run_upstream.R <dir> scores        # module scores + exported controls
#
# Upstream files are source()d into the global environment, so a global
# `sample_n` / `sample` defined here shadows dplyr's / base's for upstream code
# only (package namespaces, e.g. Seurat's, keep their own).
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(dplyr); library(tidyr); library(tibble)
  library(purrr); library(WGCNA); library(flashClust); library(dynamicTreeCut)
  library(qlcMatrix); library(ade4); library(pbapply); library(stringr); library(jsonlite)
})
pboptions(type = "none")
args <- commandArgs(TRUE)
dir <- args[1]; stage <- args[2]
up <- "rcheck/upstream"
`%notin%` <- Negate(`%in%`)
for (f in c("interaction_programs.R")) source(file.path(up, "scriabin/R", f))

expr <- readMM(file.path(dir, "expr.mtx"))
features <- read.csv(file.path(dir, "features.csv"), header = FALSE)$V1
meta <- read.csv(file.path(dir, "meta.csv"))
lr <- read.csv(file.path(dir, "lr.csv"))
expr <- as(expr, "CsparseMatrix")
dimnames(expr) <- list(features, meta$cell)
rownames(meta) <- meta$cell
seu <- CreateSeuratObject(counts = expr, meta.data = meta[, c("sample", "celltype")], assay = "RNA")
seu <- SetAssayData(seu, layer = "data", new.data = expr)
stopifnot(identical(rownames(seu), features))

write_mat <- function(m, file) write.csv(as.matrix(m), file, quote = TRUE)

if (stage == "programs") {
  draws <- lapply(fromJSON(file.path(dir, "subsamples.json"), simplifyVector = FALSE), function(reps) lapply(reps, unlist))
  out <- list()
  for (s in names(draws)) {
    obj <- subset(seu, subset = sample == s)
    # sample_n shim: per type, return Python's picks for the next subsample
    picks <- draws[[s]]; counter <- list()
    assign("sample_n", envir = globalenv(), function(tbl, size) {
      ct <- setNames(obj$celltype, colnames(obj))
      t <- ct[[tbl$cell[1]]]
      k <- (counter[[t]] %||% 0) + 1; counter[[t]] <<- k
      want <- picks[[k]][ct[picks[[k]]] == t]
      stopifnot(length(want) == size || length(want) == sum(ct == t))
      tbl[match(want, tbl$cell), , drop = FALSE]
    })
    ip <- InteractionPrograms(obj, assay = "RNA", slot = "data", database = "custom",
                              ligands = lr$ligand, recepts = lr$receptor,
                              cell_types = "celltype", return.mat = TRUE)
    rm(sample_n, envir = globalenv())
    write_mat(ip$cor_mat, file.path(dir, paste0("r_sim_", s, ".csv")))
    write.csv(ip$connectivity, file.path(dir, paste0("r_con_", s, ".csv")))
    write_json(ip$modules, file.path(dir, paste0("r_modules_", s, ".json")))
    out[[s]] <- ip
  }
  # FindAllInteractionPrograms' merge, on the per-sample results above
  mod_list <- unlist(lapply(out, function(x) x$modules), recursive = FALSE)
  m_merge <- local({
    lrsum <- t(do.call(rbind, lapply(seq_along(mod_list), function(x) unique(unlist(mod_list)) %in% mod_list[[x]])))
    lrsum[lrsum == T] <- 1; colnames(lrsum) <- names(mod_list); rownames(lrsum) <- unique(unlist(mod_list))
    d <- as.matrix(dist.binary(t(lrsum), method = 1)); d <- 1 - d^2
    merge_num <- 1
    while (max(d[d < 1]) > 0.15) {
      tomerge <- rownames(which(d == max(d[d < 1]), arr.ind = T))
      new_mod <- unique(unlist(mod_list[tomerge]))
      mod_list <- mod_list[names(mod_list) %notin% tomerge]
      mod_list[[length(mod_list) + 1]] <- new_mod
      names(mod_list)[length(mod_list)] <- paste0("merge_", merge_num); merge_num <- merge_num + 1
      lrsum <- t(do.call(rbind, lapply(seq_along(mod_list), function(x) unique(unlist(mod_list)) %in% mod_list[[x]])))
      lrsum[lrsum == T] <- 1; colnames(lrsum) <- names(mod_list); rownames(lrsum) <- unique(unlist(mod_list))
      d <- as.matrix(dist.binary(t(lrsum), method = 1)); d <- 1 - d^2
    }
    names(mod_list) <- paste0("IP-", seq_along(mod_list)); mod_list
  })
  write_json(m_merge, file.path(dir, "r_merged.json"))

  # NICHES, per sample, all three non-spatial modes and blends
  for (f in list.files(file.path(up, "niches/R"), full.names = TRUE)) source(f)
  db <- data.frame(lig = lr$ligand_niches, rec = lr$receptor_niches)
  for (s in names(draws)) {
    obj <- subset(seu, subset = sample == s)
    for (blend in c("mean", "sum", "mean.adj")) {
      res <- RunNICHES(obj, assay = "RNA", LR.database = "custom", custom_LR_database = db,
                       species = "human", cell_types = "celltype", meta.data.to.map = NULL, min.cells.per.ident = 1,
                       CellToCell = blend == "mean", SystemToCell = TRUE, CellToSystem = TRUE,
                       blend = blend, output_format = "raw")
      write_mat(res$SystemToCell$SystemToCellMatrix, file.path(dir, sprintf("r_s2c_%s_%s.csv", s, blend)))
      write_mat(res$CellToSystem$CellToSystemMatrix, file.path(dir, sprintf("r_c2s_%s_%s.csv", s, blend)))
      if (blend == "mean") {
        write_mat(res$CellToCell$CellToCellMatrix, file.path(dir, sprintf("r_c2c_%s.csv", s)))
        write.csv(res$CellToCell$metadata[, c("SendingType", "ReceivingType")], file.path(dir, sprintf("r_c2c_meta_%s.csv", s)))
      }
    }
  }
}

if (stage == "significance") {
  sets <- fromJSON(file.path(dir, "random_sets.json"), simplifyVector = FALSE)
  mods <- lapply(fromJSON(file.path(dir, "r_merged.json")), unlist)
  m_cor_list <- list(); con_list <- list()
  for (s in names(sets)) {
    m <- as.matrix(read.csv(file.path(dir, paste0("r_sim_", s, ".csv")), row.names = 1, check.names = FALSE))
    m_cor_list[[s]] <- m
    con_list[[s]] <- read.csv(file.path(dir, paste0("r_con_", s, ".csv")), row.names = 1)
  }
  # sample shim: Python's random LR sets, in upstream's call order
  # (sample-major, then module, then replicate)
  queue <- unlist(lapply(names(sets), function(s) lapply(sets[[s]], function(per_mod) per_mod)), recursive = FALSE)
  queue <- unlist(queue, recursive = FALSE); qi <- 0
  assign("sample", envir = globalenv(), function(x, size, ...) { qi <<- qi + 1; v <- unlist(queue[[qi]]); stopifnot(length(v) == size); x[v + 1] })
  n_rep <- max(sapply(unlist(sets, recursive = FALSE), length))
  sig <- InteractionProgramSignificance(list(m_cor_list, NULL, mods, con_list), n.replicate = n_rep)
  message("random sets consumed: ", qi, " of ", length(queue))
  rm(sample, envir = globalenv())
  write.csv(sig, file.path(dir, "r_significance.csv"), row.names = FALSE)
}

if (stage == "scores") {
  mods <- lapply(fromJSON(file.path(dir, "r_merged.json")), unlist)
  scored <- ScoreInteractionPrograms(seu, mods, return.assay = FALSE)
  md <- scored@meta.data
  write.csv(md[, grepl("^(ligands|receptors)_", colnames(md))], file.path(dir, "r_scores.csv"))
  # Seurat's control draws, by re-running .AddModuleScore's selection with the
  # same seed; checked against `scored` before export
  l_mods <- lapply(mods, function(x) unique(sapply(str_split(x, "="), `[[`, 1)))
  r_mods <- lapply(mods, function(x) unique(sapply(str_split(x, "="), `[[`, 2)))
  data <- GetAssayData(seu, layer = "data")
  controls <- function(features) {
    set.seed(1)
    data.avg <- Matrix::rowMeans(data); data.avg <- data.avg[order(data.avg)]
    data.cut <- ggplot2::cut_number(x = data.avg + rnorm(n = length(data.avg))/1e30, n = 24, labels = FALSE, right = FALSE)
    names(data.cut) <- names(data.avg)
    lapply(features, function(f) {
      f <- intersect(f, rownames(data))
      unique(unlist(lapply(f, function(g) names(sample(data.cut[which(data.cut == data.cut[g])], size = 100, replace = FALSE)))))
    })
  }
  ctrl <- list(ligands = controls(l_mods), receptors = controls(r_mods))
  for (side in names(ctrl)) for (i in seq_along(mods)) {
    feats <- intersect(list(ligands = l_mods, receptors = r_mods)[[side]][[i]], rownames(data))
    redo <- Matrix::colMeans(data[feats, , drop = FALSE]) - Matrix::colMeans(data[ctrl[[side]][[i]], , drop = FALSE])
    stopifnot(isTRUE(all.equal(unname(redo), md[[paste0(side, "_", names(mods)[i])]], tolerance = 1e-12)))
  }
  write_json(ctrl, file.path(dir, "r_controls.json"))
}
