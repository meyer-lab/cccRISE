#' Post-install patches for known upstream package bugs.
#'
#' Run this once after every `uvr sync` (a fresh sync rebuilds .uvr/library/
#' from scratch, so any direct edit there is lost). See R/README.md.
#'
#' Usage: Rscript apply_patches.R

# run from within R/scripts/ (see run_all.R / README)
lib <- normalizePath(file.path(getwd(), "..", ".uvr", "library"))

patch_staccato_namespace <- function(lib) {
  # STACCato's upstream NAMESPACE (daiqile96/STACCato@main) contains a single
  # stale line, `export(str_split_one)`, re-exporting a stringr function that
  # no longer exists in current stringr (>=1.5). No package R/ source ever
  # defines str_split_one, so loadNamespace() fails outright with
  # "undefined exports: str_split_one". str_split_one is unused by every
  # other function in the package (confirmed by grepping the compiled
  # namespace), so it is safe to just drop the stale export.
  ns_file <- file.path(lib, "STACCato", "NAMESPACE")
  meta_file <- file.path(lib, "STACCato", "Meta", "nsInfo.rds")
  if (!file.exists(ns_file)) {
    message("STACCato not installed, skipping patch")
    return(invisible())
  }

  lines <- readLines(ns_file)
  writeLines(lines[!grepl("^export\\(str_split_one\\)$", lines)], ns_file)

  ns_info <- readRDS(meta_file)
  ns_info$exports <- setdiff(ns_info$exports, "str_split_one")
  saveRDS(ns_info, meta_file)

  message("Patched STACCato NAMESPACE (dropped stale str_split_one export)")
}

patch_staccato_namespace(lib)
