# Combine niches_runs.R's per-sample outputs: <dir>/s2c.mtx, c2s.mtx (LR x cell),
# s2c_rows.txt / c2s_rows.txt, s2c_cells.txt / c2s_cells.txt, identical.csv (N2).
#   Rscript rcheck/shared/niches_export.R <dir>
.libPaths(c(normalizePath("rcheck/lib"), .libPaths()))
suppressPackageStartupMessages(library(Matrix))
dir <- commandArgs(TRUE)[1]
fs <- list.files(dir, pattern = "\\.rds$", full.names = TRUE)
xs <- lapply(fs, readRDS)
write.csv(data.frame(sample = sub("\\.rds$", "", basename(fs)), identical = sapply(xs, `[[`, "identical")),
          file.path(dir, "identical.csv"), row.names = FALSE)
for (mode in c("s2c", "c2s")) {
  rows <- sort(unique(unlist(lapply(xs, function(x) rownames(x[[mode]])))))
  m <- do.call(cbind, lapply(xs, function(x) { y <- x[[mode]]; z <- Matrix(0, length(rows), ncol(y), sparse = TRUE,
    dimnames = list(rows, colnames(y))); z[rownames(y), ] <- y; z }))
  writeMM(m, file.path(dir, paste0(mode, ".mtx")))
  writeLines(rows, file.path(dir, paste0(mode, "_rows.txt")))
  writeLines(sub("^System—|—System$", "", colnames(m)), file.path(dir, paste0(mode, "_cells.txt")))
}
