# R/ — reviewer-requested comparator methods

This folder is an [`uvr`](https://github.com/nbafrank/uvr)-managed R project
(the R equivalent of `uv`) that installs and runs every R-implemented method
the reviewers named in `manuscript/review-response.qmd` (see below), so
their output can be compared against the Python CCC-RISE analysis in
`cellcommunicationpf2/`.

Of the methods reviewers asked us to compare against, these are implemented
in R:

| Method | Source | Reviewer ask |
|---|---|---|
| [CellChat](https://github.com/jinworks/CellChat) | `jinworks/CellChat` | R1#1 (tensor/multi-condition baseline), R2#1, R2#3 |
| [NICHES](https://github.com/msraredon/NICHES) | `msraredon/NICHES` | R1#1 (single-cell-resolution baseline) |
| [Scriabin](https://github.com/BlishLab/scriabin) | `BlishLab/scriabin` | R1#1, R1#2 (single-cell-resolution concordance) |
| [scDiffCom](https://cran.r-project.org/package=scDiffCom) | CRAN | R1#6, R1#7 (permutation-based differential CCC) |
| [CrossTalkeR](https://github.com/CostaLab/CrossTalkeR) | `CostaLab/CrossTalkeR` | R1#7 (differential LR-network analysis) |

Tensor-cell2cell, CellPhoneDB, and DeepCOLOR (the other named tools) are
Python packages and are out of scope for this folder.

STACCato (`daiqile96/STACCato`, R1#1) is no longer run from here -- its
supervised Tucker decomposition is re-implemented in Python/TensorLy as
`cellcommunicationpf2.comparators.staccato`, which avoids the R runtime and
the upstream NAMESPACE bug documented below. See that function's docstring
for what it reproduces and how it differs from the original package.

## Setup

```bash
curl -fsSL https://raw.githubusercontent.com/nbafrank/uvr/main/install.sh | sh   # installs uvr to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

cd R
uvr sync                    # installs everything from uvr.lock into .uvr/library/
```

`uvr sync` builds ~450 packages (CellChat, Seurat, NICHES, scriabin and
their transitive Bioconductor/CRAN dependencies). Expect ~15-20 minutes on
a clean cache. It needs a handful of system libraries beyond a bare R
install:

```bash
sudo apt-get install -y libglpk-dev libfontconfig1-dev libfreetype6-dev libharfbuzz-dev \
    gdal-bin jags libcairo2-dev libgdal-dev libgeos-dev libgmp3-dev libgsl0-dev \
    libmpfr-dev libopenmpi-dev libproj-dev
```

Some packages (CelliD, a scriabin dependency) require C++14/17; if your
system R defaults to C++11 for packages that request it explicitly (visible
as `#error "*** C++14 compiler required"` during `uvr sync`), add to
`~/.R/Makevars`:

```
CXX_STD = CXX17
CXX11STD = -std=gnu++17
CXX14STD = -std=gnu++17
```

To run any script, activate the project library first:

```bash
cd R
source .uvr/activate    # sets R_LIBS_USER for this shell
cd scripts               # scripts assume this working directory
```

## Upstream bugs patched here

One of the five packages has a bug in its current GitHub `HEAD` that blocks
a stock run. It's documented and fixed by code committed in this folder
rather than by hand-editing installed files (which `uvr sync` would
overwrite):

- **Scriabin**: `GenerateInteractome()` (a) passes the removed `slot=`
  argument to `SeuratObject::GetAssayData()` (Seurat >=5 requires `layer=`),
  and (b) indexes its bundled LR resource one level too shallow (it's
  actually nested by species: `lr_resources.rds$human$<database>`).
  `scripts/patch_scriabin.R` defines a corrected copy of the function and
  installs it into the `scriabin` namespace at runtime via
  `assignInNamespace()`. `scripts/run_scriabin.R` sources this before calling
  `GenerateInteractome()`; nothing to run manually.

## Exporting a dataset

The R packages read a Seurat object built from a plain 10x-style
MatrixMarket export, not the project's `.h5ad` file directly (avoids adding
an HDF5/AnnData R dependency). `scripts/export_dataset.py` builds this
export using the repo's own Python loaders (`cellcommunicationpf2.import_data`),
so both sides of the comparison start from identical QC-filtered raw counts:

```bash
cd /path/to/cellcommunication-Pf2
source .venv/bin/activate
python3 R/scripts/export_dataset.py --dataset balf_covid --out R/data/balf_covid
```

`--dataset` currently supports `balf_covid` (the COVID-19 BAL cohort,
bundled in the repo at `cellcommunicationpf2/data/`) and `alad` (the lung
transplant cohort — requires your own copy of `BAL-scRNAseq-raw.h5ad`,
which isn't distributed with the repo; pass `--dataset alad` after placing
that file where `cellcommunicationpf2.import_data.import_alad` expects it).

Scriabin's per-cell-pair computation is O(n_send × n_rec); for a full-scale
run (63k cells for `balf_covid`) most other tools are fine, but Scriabin
itself is only invoked on a small sampled subset (see below) regardless of
export size. If you also want the *other* four methods to run fast for a
smoke test, downsample the export itself:

```bash
python3 R/scripts/export_dataset.py --dataset balf_covid \
    --out R/data/balf_covid_small --cells-per-sample 60 --seed 0
```

This writes, into `--out`:
- `matrix.mtx.gz`, `genes.tsv.gz`, `barcodes.tsv.gz` — counts, genes x cells
- `metadata.csv` — per-cell obs (`sample`, `condition`, `celltype`, ...)
- `lr_pairs.csv` — the project's own CellChat "Human-2020-Jin" LR database,
  reused by CellChat, scDiffCom (via its own bundled database instead, see
  below), CrossTalkeR, and Scriabin (multi-subunit complexes, joined with
  `&`, are dropped for the tools that only accept single-gene
  ligands/receptors)

## Running the methods

```bash
cd R
source .uvr/activate
cd scripts
Rscript run_all.R ../data/balf_covid_small ../output/balf_covid_small \
    celltype condition Control "Severe COVID-19"
```

This calls each `run_<method>.R` script in turn and continues past any
individual failure (each is also directly runnable; see the header comment
in each file for its own CLI). Arguments, in order: exported dataset dir,
output root dir, cell-type/group column, condition column, and the two
condition levels used by the differential tools (scDiffCom, CrossTalkeR).

Individual scripts and what they need beyond the shared arguments:

| Script | Extra behavior |
|---|---|
| `run_cellchat.R` | Standard CellChat pipeline; LR-level + pathway-level communication tables |
| `run_niches.R` | Per-cell-pair signal matrix (NICHES' own bundled fantom5 LR database) |
| `run_scdiffcom.R` | Two-condition permutation test; needs `iterations` (default 1000, pass a smaller value for a quick smoke test) |
| `run_crosstalker.R` | Runs CellChat separately per condition, then feeds both into CrossTalkeR's differential network builder. Downloads KEGG pathway annotations on first run — needs internet access. |
| `run_scriabin.R` | Samples `n_send`/`n_rec` cells (default 25 each) from two named groups — keep small, this is O(n_send × n_rec) |

## Output format

Every script writes plain CSV (and one `.rds` copy of the full R object, for
follow-up work in R) to its `out_dir`. CSVs are directly importable in
Python with `pandas.read_csv`, so e.g. `cellchat/lr_communication.csv` can be
loaded next to the CCC-RISE factors in `cellcommunicationpf2/data/*.nc` for
comparison.

## Adding a package

```bash
uvr add <package>            # CRAN
uvr add <package> --bioc     # Bioconductor
uvr add <owner>/<repo>       # GitHub
```

`uvr add` updates `uvr.toml` and `uvr.lock`; commit both.
