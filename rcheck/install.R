# Isolated library for the R-vs-Python comparator checks (rcheck/lib, git-excluded).
# Plain install.packages with Bioconductor 3.22 repos (the R 4.5 release),
# so the system library's BiocManager configuration is left untouched.
lib <- normalizePath("rcheck/lib")
.libPaths(c(lib, .libPaths()))
bioc <- "https://bioconductor.org/packages/3.22"
options(repos = c(CRAN = "https://cloud.r-project.org",
                  BioCsoft = file.path(bioc, "bioc"),
                  BioCann = file.path(bioc, "data/annotation"),
                  BioCexp = file.path(bioc, "data/experiment")), Ncpus = 8)
need <- c("Seurat", "qlcMatrix", "WGCNA", "BisRNA", "gtools", "permute", "plotrix")
missing <- setdiff(need, rownames(installed.packages()))
cat("missing:", missing, "\n")
if (length(missing)) install.packages(missing, lib = lib, dependencies = c("Depends", "Imports", "LinkingTo"))
for (p in need) cat(p, requireNamespace(p, quietly = TRUE), as.character(tryCatch(packageVersion(p), error = function(e) NA)), "\n")
