# Dump each per-sample InteractionPrograms result's averaged LR similarity
# (cor_mat) for the S6 locus analysis: sims/<sample>.f64 (raw float64,
# column-major, n x n) and sims/<sample>.txt (its LR names, in order).
# Usage: Rscript rcheck/shared/export_sims.R <run_dir> [<run_dir> ...]
for (dir in commandArgs(TRUE)) {
  out <- file.path(dir, "sims"); dir.create(out, showWarnings = FALSE)
  for (f in list.files(dir, pattern = "\\.rds$")) {
    s <- sub("\\.rds$", "", f)
    if (file.exists(file.path(out, paste0(s, ".txt")))) next
    m <- readRDS(file.path(dir, f))$cor_mat
    writeBin(as.vector(m), file.path(out, paste0(s, ".f64")), size = 8)
    writeLines(rownames(m), file.path(out, paste0(s, ".txt")))
  }
  cat(dir, length(list.files(out, pattern = "\\.txt$")), "samples\n")
}
