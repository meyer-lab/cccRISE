"""panc8 inDrop through cccRISE's own preprocessing, for upstream R on
cccRISE's inputs: raw counts (rcheck/authors/panc8_export_counts.R) ->
cccRISE's normalization, as ``vsparse.load_and_normalize`` (cells > 10
counts; genes with total counts > 0.001 x cells; each cell scaled to the
median depth, each gene by its total, log10(1000x + 1)) -> CellChatDB LR
pairs (cccRISE's table; pairs whose genes are all measured; one per
lr_name), complexes as min-aggregated pseudo-genes
(`inputs.with_complex_pseudogenes`). Writes singlecell_cache/shared/panc8/ in
rcheck/shared/export_dataset.py's layout (lr_expr.mtx features x cells,
lr_features.csv, meta.csv, lr.csv), then the .rds inputs ip_runs.R reads
(rcheck/shared/to_rds.R: lr_data.rds, meta.rds; plus meta_bytype.rds, the
vignette's split with sample = celltype).

  uv run python rcheck/authors/panc8_cccrise_inputs.py
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from cccrise.comparators.inputs import (
    COMPLEX_SEP,
    unique_lr_pairs,
    with_complex_pseudogenes,
)
from cccrise.import_data import import_ligand_receptor_pairs

PANC8 = Path("singlecell_cache/authors/panc8")
OUT = Path("singlecell_cache/shared/panc8")


def normalize(counts: sp.csr_matrix) -> sp.csr_matrix:
    """cccRISE's normalization (`vsparse.load_and_normalize`'s arithmetic)."""
    x = counts.astype(np.float64).tocsr()
    depth = np.asarray(x.sum(axis=1)).ravel()
    depth = depth / np.median(depth)
    x = sp.diags(1.0 / depth) @ x
    gene_sum = np.asarray(x.sum(axis=0)).ravel()
    x = (x @ sp.diags(1.0 / np.where(gene_sum > 0, gene_sum, 1.0))).tocsr()
    x.data = np.log10(1000.0 * x.data + 1.0)
    return x


def main() -> None:
    raw = PANC8 / "raw"
    counts = (
        scipy.io.mmread(raw / "counts.mtx").T.tocsr().astype(np.float64)
    )  # cells x genes
    genes = pd.Index((raw / "genes.txt").read_text().split())
    cells = pd.Index((raw / "cells.txt").read_text().split())
    keep_c = np.asarray(counts.sum(axis=1)).ravel() > 10.0
    counts, cells = counts[keep_c], cells[keep_c]
    keep_g = np.asarray(counts.sum(axis=0)).ravel() > 0.001 * counts.shape[0]
    counts, genes = counts[:, keep_g], genes[keep_g]
    X = normalize(counts).tocsc()
    lr = import_ligand_receptor_pairs().astype(
        {"ligand": str, "receptor": str, "interaction_symbol": str}
    )
    lr = lr.drop_duplicates("interaction_symbol")
    measured = set(genes)
    ok = [
        all(g in measured for g in f"{a}{COMPLEX_SEP}{b}".split(COMPLEX_SEP))
        for a, b in zip(lr.ligand, lr.receptor, strict=True)
    ]
    lr = lr[ok].copy()
    lr["lr_name"] = lr.ligand + "=" + lr.receptor
    lr = unique_lr_pairs(lr)[["interaction_symbol", "ligand", "receptor", "lr_name"]]
    entities = pd.unique(pd.concat([lr.ligand, lr.receptor]))
    lr_genes = np.unique(np.concatenate([e.split(COMPLEX_SEP) for e in entities]))
    cols = np.sort(genes.get_indexer(lr_genes))
    X_lr, names = with_complex_pseudogenes(X[:, cols], genes[cols], lr, agg="min")
    pos = np.sort(names.get_indexer(entities))
    OUT.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(OUT / "lr_expr.mtx", sp.csc_matrix(X_lr)[:, pos].T.tocoo())
    pd.Series(names[pos]).to_csv(OUT / "lr_features.csv", index=False, header=False)
    meta = pd.read_csv(PANC8 / "port" / "meta.csv").set_index("cell").loc[cells]
    pd.DataFrame(
        {
            "cell": cells,
            "sample": meta.donor.to_numpy(),
            "celltype": meta.celltype.to_numpy(),
        }
    ).to_csv(OUT / "meta.csv", index=False)
    lr.to_csv(OUT / "lr.csv", index=False)
    (OUT / "dataset.txt").write_text("panc8\n")
    subprocess.run(["Rscript", "rcheck/shared/to_rds.R", str(OUT)], check=True)
    bytype = (
        f"m <- readRDS('{OUT}/meta.rds'); "
        f"saveRDS(transform(m, sample = celltype), '{OUT}/meta_bytype.rds')"
    )
    subprocess.run(["Rscript", "-e", bytype], check=True)
    print(
        f"{len(cells)} cells, {len(genes)} genes kept; {len(lr)} LR pairs, {len(pos)} LR features"
    )


if __name__ == "__main__":
    main()
