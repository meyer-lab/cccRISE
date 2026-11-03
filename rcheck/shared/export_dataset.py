"""Write a registry dataset's inputs for the R-shared arm (upstream R on
cccRISE's inputs; `CONCORDANCE_PLAN.md` §3.2).

Writes to OUT: ``lr_expr.mtx`` (features x cells: the LR pairs' single-gene
ligands/receptors and their ``min``-aggregated complex pseudo-genes, cccRISE's
normalized expression), ``lr_features.csv``, ``meta.csv`` (cell, sample,
celltype) and ``lr.csv`` (the fit's LR pairs, one per ``ligand=receptor``);
with ``--full``, also ``expr.mtx`` / ``features.csv`` over every gene (for
``AddModuleScore``'s control genes). `to_rds.R` then converts them.

Usage: uv run python rcheck/shared/export_dataset.py OUT --dataset alad [--full]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import dataset_inputs

from cccrise.comparators.inputs import with_complex_pseudogenes


def main(out: Path, dataset: str, full: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    inp = dataset_inputs(dataset)
    # LR features from the LR genes' columns only: appending pseudo-genes to
    # the whole matrix copies it (IBD: 2e9 nonzeros, OOM at 64G)
    entities = pd.unique(pd.concat([inp.lr["ligand"], inp.lr["receptor"]]))
    genes = np.unique(np.concatenate([e.split("&") for e in entities]))
    cols = np.sort(inp.genes.get_indexer(genes)[inp.genes.get_indexer(genes) >= 0])
    X_lr, names = with_complex_pseudogenes(
        inp.X[:, cols], inp.genes[cols], inp.lr, agg="min"
    )
    keep = names.get_indexer(entities)
    keep = np.sort(keep[keep >= 0])
    scipy.io.mmwrite(out / "lr_expr.mtx", X_lr[:, keep].T.tocoo())
    pd.Series(names[keep]).to_csv(out / "lr_features.csv", index=False, header=False)
    if full:
        X, all_names = with_complex_pseudogenes(inp.X, inp.genes, inp.lr, agg="min")
        scipy.io.mmwrite(out / "expr.mtx", X.T.tocoo())
        pd.Series(all_names).to_csv(out / "features.csv", index=False, header=False)
    inp.obs.rename_axis("cell").reset_index().to_csv(out / "meta.csv", index=False)
    inp.lr.to_csv(out / "lr.csv", index=False)
    (out / "dataset.txt").write_text(dataset + "\n")
    print(
        f"{dataset}: {X_lr.shape[0]} cells, {len(inp.samples)} samples, "
        f"{len(keep)} LR features, {len(inp.lr)} LR pairs, full={full}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--dataset", required=True, choices=["alad", "ibd"])
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    main(args.out, args.dataset, args.full)
