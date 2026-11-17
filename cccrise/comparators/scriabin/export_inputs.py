"""cccRISE's inputs for upstream Scriabin (`programs.R`, Appendix Figure S8A),
from PR #275's ``rcheck/shared/export_dataset.py``: a registry dataset's
normalized expression of the LR genes, its cells' sample and coarse cell type,
and its production fit's LR pairs.

Writes to OUT:

- ``lr_expr.mtx`` (features x cells) and ``lr_features.csv``: each ligand and
  receptor of the fit's LR pairs, a single gene as itself and a complex as one
  feature, the minimum over its subunits (0 if a subunit is not among the
  fit's genes), so upstream treats a complex the way it treats a gene;
- ``meta.csv``: cell, sample, celltype (the dataset's coarse labels);
- ``lr.csv``: the fit's LR pairs in its order, one per ``ligand=receptor``
  (Scriabin's pair label; a second pair naming the same genes is dropped).

Usage: uv run python -m cccrise.comparators.scriabin.export_inputs OUT --dataset alad
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.sparse import csc_matrix, hstack

from ...factors import load_factors_netcdf
from ...import_data import import_ligand_receptor_pairs
from ...runners.datasets import DATASETS
from ..niches_programs import COMPLEX_SEP


def fit_lr_pairs(lr_order: np.ndarray) -> pd.DataFrame:
    """The fit's LR pairs (`lr_order`, its ``lr_pair`` coordinate) with their
    ligand and receptor, one per ``ligand=receptor`` name."""
    table = import_ligand_receptor_pairs()
    table = table.astype({c: str for c in ("interaction_symbol", "ligand", "receptor")})
    order = pd.DataFrame({"interaction_symbol": lr_order.astype(str)})
    out = order.merge(
        table.drop_duplicates("interaction_symbol"),
        on="interaction_symbol",
        how="left",
        validate="one_to_one",
    )
    assert out["ligand"].notna().all(), "fit LR pairs missing from the LR table"
    out["lr_name"] = out["ligand"] + "=" + out["receptor"]
    out = out[["interaction_symbol", "ligand", "receptor", "lr_name"]]
    return out.drop_duplicates("lr_name").reset_index(drop=True)


def lr_features(X, genes: pd.Index, lr: pd.DataFrame) -> tuple[csc_matrix, list[str]]:
    """(cells x features) float64 expression of the ligands and receptors in
    `lr`: the single genes among `genes`, in their order, then the complexes."""
    entities = pd.unique(pd.concat([lr["ligand"], lr["receptor"]]))
    singles = [g for g in genes if g in set(entities)]
    complexes = [e for e in entities if COMPLEX_SEP in e and e not in set(genes)]
    columns = csc_matrix(X[:, genes.get_indexer(singles)], dtype=np.float64)
    extra = np.zeros((X.shape[0], len(complexes)))
    for j, entity in enumerate(complexes):
        position = genes.get_indexer(entity.split(COMPLEX_SEP))
        if (position >= 0).all():
            extra[:, j] = csc_matrix(X[:, position]).toarray().min(axis=1)
    return hstack([columns, csc_matrix(extra)], format="csc"), singles + complexes


def main(out: Path, dataset: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[dataset]
    fit = load_factors_netcdf(spec.factors_path())
    genes = pd.Index(fit.coords["gene"].values.astype(str))
    adata = spec.load()
    lr = fit_lr_pairs(fit.coords["lr_pair"].values)
    X, names = lr_features(adata[:, genes].X, genes, lr)
    scipy.io.mmwrite(out / "lr_expr.mtx", X.T.tocoo())
    pd.Series(names).to_csv(out / "lr_features.csv", index=False, header=False)
    pd.DataFrame(
        {
            "cell": adata.obs_names,
            "sample": adata.obs[spec.condition_column].astype(str).to_numpy(),
            "celltype": adata.obs[spec.celltype_column].astype(str).to_numpy(),
        }
    ).to_csv(out / "meta.csv", index=False)
    lr.to_csv(out / "lr.csv", index=False)
    print(
        f"{dataset}: {X.shape[0]} cells, {len(names)} LR features, {len(lr)} LR pairs"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--dataset", required=True, choices=["alad", "ibd"])
    args = parser.parse_args()
    main(args.out, args.dataset)
