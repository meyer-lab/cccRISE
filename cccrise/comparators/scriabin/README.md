# Upstream Scriabin runs

The R scripts behind the manuscript's Scriabin results, run with the authors'
own code: upstream Scriabin (BlishLab/scriabin) at `bce695b`. Only these
scripts are kept here. `upstream.R` sources the upstream functions they call
(GenerateCCIM, InteractionPrograms, LoadLR) from a clone of the repository,
with the text patches its header lists, so the scriabin package itself is
never installed.

| Script | Does | Result |
|---|---|---|
| `upstream.R` | Loads the pinned upstream functions from the clone | (all) |
| `export_inputs.py` | cccRISE's expression, cells and LR pairs for upstream | Appendix Fig. S8A |
| `programs.R` | InteractionPrograms per sample at one seed | Appendix Fig. S8A (`cccrise-scriabin-seeds`) |
| `panc8_prep.R` | panc8's 638 Fluidigm C1 islet cells, raw counts and meta | `figureR_scriabin` |
| `panc8_steps.R` | Shared steps of the paper's Fig. 2d-e test; lists every random draw and who seeds it | `figureR_scriabin` |
| `panc8_noise.R` | One 300-cell subsample: the paper's LISI read-out (panel A) and the programs (panel B) | `figureR_scriabin` |
| `panc8_lisi_checks.R` | Panel A's read-out with one setting changed at a time (pair cap, hard-coded seeds, no integration) | reviewer response |

S8A's merge of the per-sample programs is upstream's
FindAllInteractionPrograms merge, ported exactly to Python
(`comparators.scriabin_programs`), because the R merge takes days per seed on
IBD. S8A's cccRISE side is the seed refits of the production workflow, not
anything here.

## R setup

R 4.5.2, with the versions the results were made with (Bioconductor 3.22):

```r
options(repos = c(CRAN = "https://cloud.r-project.org",
                  BioCsoft = "https://bioconductor.org/packages/3.22/bioc"))
install.packages(c("Seurat", "qlcMatrix", "WGCNA", "flashClust", "dynamicTreeCut",
                   "plyr", "jsonlite", "scuttle", "R.utils"))
# Seurat 5.6.0, SeuratObject 5.4.0, sctransform 0.4.3, WGCNA 1.74,
# dynamicTreeCut 1.63-1, qlcMatrix 0.9.9, RSpectra 0.16.2, plyr 1.8.9, scuttle 1.20.0
remotes::install_version("rsvd", "1.0.5")                     # ALRA's randomized SVD
remotes::install_github("KlugerLab/ALRA@f34d465")
remotes::install_github("satijalab/seurat-wrappers@8728711")  # RunALRA
remotes::install_github("immunogenomics/LISI@a917556")         # lisi 1.0
# figureR_scriabin's data: panc8 (the Scriabin vignette's islets)
install.packages("https://seurat.nygenome.org/src/contrib/panc8.SeuratData_3.0.2.tar.gz",
                 repos = NULL, type = "source")
```

Then the upstream code, from the repository root:

```sh
git clone https://github.com/BlishLab/scriabin rcheck/upstream/scriabin
git -C rcheck/upstream/scriabin checkout bce695b
```

`SCRIABIN_SOURCE` points `upstream.R` at a clone elsewhere. Point `R_LIBS` at
the library holding the packages above. The scripts run from the repository
root with one BLAS thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`), as the
results were made. Thread count changes floating-point summation order, and
so the last bits of the results.

## Run

Appendix Fig. S8A (seeds 1111, upstream's own, and 2-10; per dataset):

```sh
uv run python -m cccrise.comparators.scriabin.export_inputs singlecell_cache/shared/alad --dataset alad
for seed in 1111 2 3 4 5 6 7 8 9 10; do
  Rscript cccrise/comparators/scriabin/programs.R singlecell_cache/shared/alad rcheck/out/shared/alad/s1/seed$seed $seed
done
uv run cccrise-scriabin-seeds --dataset alad
```

`figureR_scriabin` (12 subsamples; all 90,000 cell pairs of a subsample enter
integration, ~1.5 h for the raw arm and ~4 h for the ALRA arm per subsample,
~7 GB each):

```sh
Rscript cccrise/comparators/scriabin/panc8_prep.R
for i in $(seq 1 12); do
  Rscript cccrise/comparators/scriabin/panc8_noise.R $i rcheck/out/scriabin_reproduction/noise
done
Rscript cccrise/comparators/scriabin/panc8_lisi_checks.R 1 rcheck/out/scriabin_reproduction/lisi_checks
for a in downsampling downsampling_programs lisi_checks; do
  uv run cccrise-scriabin-reproduction --analysis $a
done
uv run cccrise-fbuild R_scriabin
```

## Random numbers (`figureR_scriabin`)

| Step | Draw | Seeded by |
|---|---|---|
| 300 of the 638 C1 cells | `base::sample` | caller, `set.seed(index)` |
| Read downsampling to inDrop depth | `scuttle::downsampleMatrix` | caller, `set.seed(1000 * index + draw)` |
| ALRA's randomized SVD | `rsvd` | caller, `set.seed(index)` |
| GT pairs entering integration (`panc8_lisi_checks.R`'s caps only) | `base::sample` | caller, `set.seed(index)` |
| SCTransform's 2,000-gene step-1 fit | `sample` | hard-coded, `seed.use = 1448145` |
| RunPCA | irlba start | hard-coded, `seed.use = 42` |
| CCA (anchor finding) | RSpectra `svds` start | hard-coded in C++ (Spectra `SimpleRandom`, seed 0); RunCCA's `set.seed(42)` only reaches irlba, which Seurat uses below 10^6 entries |
| Anchor kNN | Annoy | hard-coded (RcppAnnoy's default seed) |
| InteractionPrograms' cell subsampling | `sample` | hard-coded, `set.seed(1111)`; only above 300 cells, so never here |
| GenerateCCIM, ScaleData, IntegrateData weights, WGCNA programs, LISI (exact kd-tree kNN) | none | deterministic |

At 300 cells InteractionPrograms uses the full CCIM. So in panel B, the
programs of two downsampling draws of the same cells differ only through
`downsampleMatrix`'s draw. The GT CCIM's only random step is SCTransform's
hard-coded gene subsample.
