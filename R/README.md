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
| [scDiffCom](https://cran.r-project.org/package=scDiffCom) | CRAN | R1#6, R1#7 (permutation-based differential CCC) |
| [CrossTalkeR](https://github.com/CostaLab/CrossTalkeR) | `CostaLab/CrossTalkeR` | R1#7 (differential LR-network analysis) |

Tensor-cell2cell, CellPhoneDB, and DeepCOLOR (the other named tools) are
Python packages and are out of scope for this folder.

Three other reviewer-requested methods are no longer run from here -- each
is re-implemented in Python instead, under
`cellcommunicationpf2/comparators/` (one file per method), which avoids an R
runtime and, for STACCato and Scriabin, upstream bugs their R packages had
(each module's docstring documents the specific bug it sidesteps). See each
module's docstring for what it reproduces, how it differs from the original
package, and how to compare its output against CCC-RISE:

| Method | Source | Reviewer ask | Python re-implementation |
|---|---|---|---|
| STACCato | `daiqile96/STACCato` | R1#1 (tensor/multi-condition baseline) | `cellcommunicationpf2.comparators.staccato` |
| Scriabin | `BlishLab/scriabin` | R1#1, R1#2 (single-cell-resolution concordance) | `cellcommunicationpf2.comparators.scriabin` |
| NICHES | `msraredon/NICHES` | R1#1 (single-cell-resolution baseline) | `cellcommunicationpf2.comparators.niches` |

Scriabin and NICHES are re-implemented against this project's own shared,
complex-aware LR table (the same one CCC-RISE uses) rather than either
method's bundled LR database, so the comparison isolates the algorithmic
approach rather than database choice.

## Setup

```bash
curl -fsSL https://raw.githubusercontent.com/nbafrank/uvr/main/install.sh | sh   # installs uvr to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

cd R
uvr sync                    # installs everything from uvr.lock into .uvr/library/
```

`uvr sync` builds ~300 packages (CellChat, Seurat, scDiffCom, CrossTalkeR and
their transitive Bioconductor/CRAN dependencies). Expect ~10-15 minutes on
a clean cache. It needs a handful of system libraries beyond a bare R
install:

```bash
sudo apt-get install -y libglpk-dev libfontconfig1-dev libfreetype6-dev libharfbuzz-dev \
    gdal-bin jags libcairo2-dev libgdal-dev libgeos-dev libgmp3-dev libgsl0-dev \
    libmpfr-dev libopenmpi-dev libproj-dev
```

To run any script, activate the project library first:

```bash
cd R
source .uvr/activate    # sets R_LIBS_USER for this shell
cd scripts               # scripts assume this working directory
```

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

For a full-scale run (63k cells for `balf_covid`) these three tools are
fine, but if you want a faster smoke test, downsample the export itself:

```bash
python3 R/scripts/export_dataset.py --dataset balf_covid \
    --out R/data/balf_covid_small --cells-per-sample 60 --seed 0
```

This writes, into `--out`:
- `matrix.mtx.gz`, `genes.tsv.gz`, `barcodes.tsv.gz` — counts, genes x cells
- `metadata.csv` — per-cell obs (`sample`, `condition`, `celltype`, ...)
- `lr_pairs.csv` — the project's own CellChat "Human-2020-Jin" LR database,
  exported for reference (none of the three remaining R scripts read it
  directly: CellChat/CrossTalkeR use `CellChatDB.human`, scDiffCom uses its
  own bundled database). The Python comparators in
  `cellcommunicationpf2/comparators/` load this same table directly via
  `import_ligand_receptor_pairs()` instead of from this file.

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
| `run_scdiffcom.R` | Two-condition permutation test; needs `iterations` (default 1000, pass a smaller value for a quick smoke test) |
| `run_crosstalker.R` | Runs CellChat separately per condition, then feeds both into CrossTalkeR's differential network builder. Downloads KEGG pathway annotations on first run — needs internet access. |

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
