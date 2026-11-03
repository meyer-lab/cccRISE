# Extra packages for the faithful Scriabin run (rcheck/faithful/), into the
# isolated rcheck/lib: ALRA denoising via SeuratWrappers (as the vignette),
# scuttle::downsampleMatrix (the paper's noise model) and LISI (its metric).
lib <- normalizePath("rcheck/lib")
.libPaths(c(lib, .libPaths()))
bioc <- "https://bioconductor.org/packages/3.22"
options(repos = c(CRAN = "https://cloud.r-project.org",
                  BioCsoft = file.path(bioc, "bioc"),
                  BioCann = file.path(bioc, "data/annotation"),
                  BioCexp = file.path(bioc, "data/experiment")), Ncpus = 8)
# rsvd pinned to the version the ALAD faithful run used (ALRA's randomized
# SVD); CRAN's 2.0.0 is not checked against it.
if (!requireNamespace("rsvd", quietly = TRUE) || packageVersion("rsvd") != "1.0.5")
  remotes::install_version("rsvd", "1.0.5", lib = lib, upgrade = "never", dependencies = FALSE)
cran <- c("R.utils")
bio <- c("scuttle")
miss <- setdiff(c(cran, bio), rownames(installed.packages()))
if (length(miss)) install.packages(miss, lib = lib, dependencies = c("Depends", "Imports", "LinkingTo"))
deps <- c("Depends", "Imports", "LinkingTo")
remotes::install_github("KlugerLab/ALRA@f34d465", lib = lib, dependencies = deps, upgrade = "never")
remotes::install_github("satijalab/seurat-wrappers@8728711", lib = lib, dependencies = deps, upgrade = "never")
remotes::install_github("immunogenomics/LISI@a917556", lib = lib, dependencies = deps, upgrade = "never")
for (p in c("rsvd", cran, bio, "ALRA", "SeuratWrappers", "lisi"))
  cat(p, requireNamespace(p, quietly = TRUE), as.character(tryCatch(packageVersion(p), error = function(e) NA)), "\n")
