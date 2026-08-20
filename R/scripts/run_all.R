#' Run all six R-based CCC methods on one exported dataset.
#'
#' Usage: Rscript run_all.R <data_dir> <out_root> [group_by] [condition_col] [cond1] [cond2]
#' Example:
#'   Rscript run_all.R ../data/balf_covid_small ../output/balf_covid_small \
#'     celltype condition Control "Severe COVID-19"
#'
#' Runs from within R/scripts/ (each sub-script sources "lib_load_data.R"
#' with a relative path). See ../README.md for the one-command wrapper.

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[[1]]
out_root <- args[[2]]
group_by <- if (length(args) >= 3) args[[3]] else "celltype"
condition_col <- if (length(args) >= 4) args[[4]] else "condition"
cond1 <- if (length(args) >= 5) args[[5]] else "Control"
cond2 <- if (length(args) >= 6) args[[6]] else "Severe COVID-19"

run_rscript <- function(name, script, script_args) {
  cat("\n==== ", name, " ====\n", sep = "")
  tryCatch(
    {
      # shQuote() every argument -- system2() does NOT shell-quote args
      # containing spaces (e.g. "Severe COVID-19"), so without this an
      # unquoted space silently splits into two argv entries and shifts
      # every positional argument after it.
      status <- system2("Rscript", shQuote(c(script, script_args)))
      if (status != 0) stop("Rscript exited with status ", status)
    },
    error = function(e) cat("!! ", name, " failed: ", conditionMessage(e), "\n", sep = "")
  )
}

run_rscript("CellChat", "run_cellchat.R",
  c(data_dir, group_by, file.path(out_root, "cellchat")))

run_rscript("NICHES", "run_niches.R",
  c(data_dir, group_by, file.path(out_root, "niches")))

run_rscript("scDiffCom", "run_scdiffcom.R",
  c(data_dir, group_by, condition_col, cond1, cond2, file.path(out_root, "scdiffcom")))

run_rscript("CrossTalkeR", "run_crosstalker.R",
  c(data_dir, group_by, condition_col, cond1, cond2, file.path(out_root, "crosstalker")))

run_rscript("STACCato", "run_staccato.R",
  c(data_dir, group_by, condition_col, file.path(out_root, "staccato")))

run_rscript("Scriabin", "run_scriabin.R", {
  meta <- read.csv(file.path(data_dir, "metadata.csv"))
  celltypes <- sort(unique(meta[[group_by]]))
  send_group <- celltypes[[1]]
  rec_group <- celltypes[[min(2, length(celltypes))]]
  c(data_dir, group_by, send_group, rec_group, file.path(out_root, "scriabin"), "25", "25", "0")
})

cat("\nAll methods attempted. See ", out_root, " for output.\n", sep = "")
