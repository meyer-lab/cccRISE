"""Export one of this repo's AnnData datasets to a directory that the R
methods in ../R can load without needing an h5ad/HDF5 reader in R.

Writes 10x-style MatrixMarket output (matrix.mtx.gz + genes.tsv.gz +
barcodes.tsv.gz), readable in R with `Seurat::ReadMtx()` or in Python with
`scipy.io.mmread()` / `scanpy.read_mtx()` -- so both sides of the comparison
start from the identical, QC-filtered raw-count matrix.

Usage
-----
    uv run R/scripts/export_dataset.py --dataset balf_covid --out R/data/balf_covid
    uv run R/scripts/export_dataset.py --dataset balf_covid --out R/data/balf_covid_small \
        --cells-per-sample 300 --seed 0
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.io import mmwrite

from cellcommunicationpf2.import_data import (
    import_alad,
    import_balf_covid,
    import_ligand_receptor_pairs,
)

DATASETS = {
    "balf_covid": lambda: import_balf_covid(normalize=False),
    "alad": lambda: import_alad(normalize=False),
}


def downsample_per_sample(adata, condition_name: str, n_per_sample: int, seed: int):
    rng = np.random.default_rng(seed)
    keep = []
    for idx in adata.obs.groupby(condition_name, observed=True).indices.values():
        idx = np.asarray(idx)
        if len(idx) > n_per_sample:
            idx = rng.choice(idx, size=n_per_sample, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return adata[keep].copy()


def gzip_file(path: Path) -> None:
    with open(path, "rb") as f_in, gzip.open(f"{path}.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--condition-col",
        default=None,
        help="obs column identifying the sample/condition (defaults per dataset)",
    )
    parser.add_argument(
        "--cells-per-sample",
        type=int,
        default=None,
        help="if set, randomly subsample to at most this many cells per sample "
        "(recommended for pairwise-cell methods like Scriabin on full-size data)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    default_condition = {"balf_covid": "sample", "alad": "dsco_id"}
    condition_col = args.condition_col or default_condition[args.dataset]

    print(f"Loading dataset '{args.dataset}' (raw, QC-filtered counts)...")
    adata = DATASETS[args.dataset]()

    if args.cells_per_sample is not None:
        n_before = adata.n_obs
        adata = downsample_per_sample(
            adata, condition_col, args.cells_per_sample, args.seed
        )
        print(
            f"Downsampled {n_before} -> {adata.n_obs} cells "
            f"(<= {args.cells_per_sample} per '{condition_col}')"
        )

    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"Writing MatrixMarket matrix ({adata.n_obs} cells x {adata.n_vars} genes)..."
    )
    X = scipy.sparse.csr_matrix(adata.X)
    mm_path = args.out / "matrix.mtx"
    mmwrite(mm_path, X.T.tocsr())  # genes x cells, matching 10x convention
    gzip_file(mm_path)

    genes = pd.DataFrame({"gene_id": adata.var_names, "gene_symbol": adata.var_names})
    genes["feature_type"] = "Gene Expression"
    genes_path = args.out / "genes.tsv"
    genes.to_csv(genes_path, sep="\t", header=False, index=False)
    gzip_file(genes_path)

    barcodes_path = args.out / "barcodes.tsv"
    pd.Series(adata.obs_names).to_csv(barcodes_path, header=False, index=False)
    gzip_file(barcodes_path)

    meta_path = args.out / "metadata.csv"
    cast(pd.DataFrame, adata.obs).to_csv(meta_path)

    lr = import_ligand_receptor_pairs()
    lr_cols = [
        "interaction_name",
        "pathway_name",
        "ligand",
        "receptor",
        "ligand_symbol",
        "receptor_symbol",
        "annotation",
        "evidence",
    ]
    lr[lr_cols].to_csv(args.out / "lr_pairs.csv", index=False)

    print(f"Done. Exported dataset to {args.out}/")
    print("  matrix.mtx.gz, genes.tsv.gz, barcodes.tsv.gz : counts (genes x cells)")
    print(
        f"  metadata.csv                                 : per-cell obs, condition column = '{condition_col}'"
    )
    print(
        "  lr_pairs.csv                                 : CellChat Human-2020-Jin ligand-receptor database"
    )


if __name__ == "__main__":
    main()
