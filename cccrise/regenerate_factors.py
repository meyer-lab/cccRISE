"""Augment saved CCC-RISE factor `.nc` files with per-cell projections.

Loads the raw single-cell data for a dataset, projects it onto the already-
computed RISE/CP factors (as `load_factors_to_anndata(..., do_embedding=True)`
already does for figure generation), and writes the per-cell projections,
sender/receiver CP embeddings, PaCMAP coordinates, and cell metadata back
into the factors `.nc` file (as float16) via `add_cell_projections`. This
lets downstream consumers (e.g. the Shiny explorer app) drive per-cell
visualizations directly from the `.nc` file, without re-loading or
re-projecting the raw single-cell data themselves.

Usage:
    python -m cccrise.regenerate_factors covid alad
    python -m cccrise.regenerate_factors covid --alad-raw-path /opt/BAL-scRNAseq-raw.h5ad
"""

import argparse
from pathlib import Path

from .factors import (
    ALAD_FACTORS_PATH,
    COVID19_FACTORS_PATH,
    add_cell_projections,
    load_factors_netcdf,
    load_factors_to_anndata,
    save_factors_netcdf,
)
from .import_data import import_alad, import_balf_covid

COVID_OBS_COLUMNS = ["celltype", "condition", "sample"]
ALAD_OBS_COLUMNS = ["celltype", "broad_cell_type", "dsco_id", "ALADstatus"]


def regenerate_covid(random_state: int = 0) -> None:
    print("Loading raw COVID-19 BALF data...")
    raw = import_balf_covid(gene_threshold=0.001, normalize=True)
    _regenerate(COVID19_FACTORS_PATH, raw, COVID_OBS_COLUMNS, random_state)


def regenerate_alad(raw_path: str | Path, random_state: int = 0) -> None:
    print("Loading raw ALAD data...")
    raw = import_alad(filename=raw_path, gene_threshold=0.001, normalize=True)
    _regenerate(ALAD_FACTORS_PATH, raw, ALAD_OBS_COLUMNS, random_state)


def _regenerate(factors_path, raw, obs_columns, random_state) -> None:
    print(f"Projecting cells onto factors from {factors_path}...")
    projected = load_factors_to_anndata(
        factors_path, adata=raw, do_embedding=True, random_state=random_state
    )

    # Load fully into memory and close the source file before writing back to
    # the same path -- netCDF4 can't have a file open for read while we
    # overwrite it.
    with load_factors_netcdf(factors_path) as ds:
        ds = ds.load()
    ds = add_cell_projections(ds, projected, obs_columns=obs_columns)

    print(f"Saving augmented factors to {factors_path}...")
    tmp_path = Path(factors_path).with_suffix(".tmp.nc")
    save_factors_netcdf(ds, tmp_path)
    tmp_path.replace(factors_path)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=["covid", "alad"],
        help="Which dataset(s) to regenerate.",
    )
    parser.add_argument(
        "--alad-raw-path",
        default="./BAL-scRNAseq-raw.h5ad",
        help="Path to the raw ALAD .h5ad file (default: %(default)s).",
    )
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    if "covid" in args.datasets:
        regenerate_covid(random_state=args.random_state)
    if "alad" in args.datasets:
        regenerate_alad(args.alad_raw_path, random_state=args.random_state)


if __name__ == "__main__":
    main()
