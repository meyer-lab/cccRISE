"""Shared inputs for the single-cell comparators (Scriabin, NICHES).

The comparators run on exactly what cccRISE sees, not on each method's own
preprocessing (`CONCORDANCE_PLAN.md` decision 3, the "R-shared" arm):

- expression is cccRISE's normalized, uncentred ``adata.X`` (the registry
  loader, ``normalize=True``), restricted to the fit's genes;
- the LR universe is the pairs the cccRISE fit scored (`fit_lr_pairs`), in
  the fit's order;
- multi-subunit complexes are aggregated as cccRISE aggregates them
  (``min`` across subunits, as `ccc_rise.add_complexes_to_expression`: a
  complex with any subunit missing is 0). Upstream NICHES multiplies
  subunits (``agg="product"``, kept for the R checks); upstream Scriabin
  drops complexes, whose names are not genes, so they enter it as
  pseudo-genes (`with_complex_pseudogenes`).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csc_array, hstack, issparse

from ..factors import load_factors_netcdf
from ..import_data import import_ligand_receptor_pairs

COMPLEX_SEP = "&"
AGGREGATORS = ("min", "mean", "gmean", "product")


def fit_lr_pairs(
    factors_ds: xr.Dataset, lr_pairs: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The LR pairs the cccRISE fit in `factors_ds` scored, in its ``lr_pair``
    order, with ``ligand``/``receptor``/``interaction_symbol`` columns and an
    ``lr_name`` of the form ``ligand=receptor`` (Scriabin's pair label)."""
    table = import_ligand_receptor_pairs() if lr_pairs is None else lr_pairs
    table = table.astype({c: str for c in ("interaction_symbol", "ligand", "receptor")})
    # The table repeats some pairs (e.g. GH1^GHR under two pathways); same genes
    table = table.drop_duplicates("interaction_symbol")
    order = pd.DataFrame(
        {"interaction_symbol": factors_ds.coords["lr_pair"].values.astype(str)}
    )
    out = order.merge(table, on="interaction_symbol", how="left", validate="one_to_one")
    assert out["ligand"].notna().all(), "fit LR pairs missing from the LR table"
    out["lr_name"] = out["ligand"] + "=" + out["receptor"]
    return out[["interaction_symbol", "ligand", "receptor", "lr_name"]]


def _subunit_index(
    entities: Sequence[str], gene_index: dict[str, int], complex_sep: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(entities x max subunits) gene indices and a mask of real subunits, and
    whether every subunit of each entity is a known gene."""
    split = [e.split(complex_sep) for e in entities]
    width = max((len(s) for s in split), default=1)
    idx = np.zeros((len(split), width), dtype=np.int64)
    mask = np.zeros((len(split), width), dtype=bool)
    valid = np.zeros(len(split), dtype=bool)
    for i, subunits in enumerate(split):
        found = [gene_index.get(s) for s in subunits]
        if all(f is not None for f in found):
            valid[i] = True
            idx[i, : len(found)] = cast(list[int], found)
            mask[i, : len(found)] = True
    return idx, mask, valid


def _aggregate(vals: np.ndarray, mask: np.ndarray, agg: str) -> np.ndarray:
    """Reduce (cells, entities, subunits) `vals` over masked subunits."""
    if agg == "min":
        return np.where(mask, vals, np.inf).min(axis=2)
    if agg == "product":
        return np.where(mask, vals, 1.0).prod(axis=2)
    counts = np.maximum(mask.sum(axis=1), 1)
    if agg == "mean":
        return np.where(mask, vals, 0.0).sum(axis=2) / counts
    if agg == "gmean":
        with np.errstate(divide="ignore"):
            logs = np.where(mask, np.log(vals), 0.0)
        return np.exp(logs.sum(axis=2) / counts)
    raise ValueError(f"agg must be one of {AGGREGATORS}, not {agg!r}")


def entity_expression(
    X: Any,
    gene_names: pd.Index | Sequence[str],
    entities: pd.Series | Sequence[str],
    agg: str = "min",
    complex_sep: str = COMPLEX_SEP,
) -> np.ndarray:
    """(cells x entities) float64 expression of ligands or receptors, each
    possibly a `complex_sep`-joined complex aggregated with `agg`
    (`AGGREGATORS`). An entity with any subunit missing is 0."""
    entities = [str(e) for e in entities]
    gene_index = {g: i for i, g in enumerate(gene_names)}
    idx, mask, valid = _subunit_index(entities, gene_index, complex_sep)
    out = np.zeros((X.shape[0], len(entities)))
    if not valid.any():
        return out
    used = np.unique(idx[mask])
    local = np.zeros(len(gene_index), dtype=np.int64)
    local[used] = np.arange(len(used))
    sub = X[:, used]
    expr = np.asarray(sub.toarray() if issparse(sub) else sub, np.float64)
    vals = expr[:, local[idx[valid]]]  # (cells, valid entities, subunits)
    out[:, valid] = _aggregate(vals, mask[valid], agg)
    return out


def lr_expression(
    X: Any,
    gene_names: pd.Index | Sequence[str],
    lr_pairs: pd.DataFrame,
    agg: str = "min",
) -> tuple[np.ndarray, np.ndarray]:
    """(cells x LR pairs) ligand and receptor expression: column l is pair
    l's ligand (sender role) and its receptor (receiver role)."""
    lig = entity_expression(X, gene_names, lr_pairs["ligand"], agg)
    rec = entity_expression(X, gene_names, lr_pairs["receptor"], agg)
    return lig, rec


def with_complex_pseudogenes(
    X: Any,
    gene_names: pd.Index | Sequence[str],
    lr_pairs: pd.DataFrame,
    agg: str = "min",
) -> tuple[Any, pd.Index]:
    """`X` with one extra column per complex ligand/receptor of `lr_pairs`
    (its aggregated expression), so Scriabin's module scores and upstream R
    can treat a complex as a feature the way they treat a gene."""
    names = pd.Index(gene_names)
    entities = pd.unique(pd.concat([lr_pairs["ligand"], lr_pairs["receptor"]]))
    complexes = [e for e in entities if COMPLEX_SEP in e and e not in names]
    if not complexes:
        return X, names
    extra = entity_expression(X, names, complexes, agg)
    if issparse(X):
        combined = hstack([csc_array(X), csc_array(extra)], format="csc")
    else:
        combined = np.hstack([np.asarray(X), extra])
    return combined, names.append(pd.Index(complexes))


def unique_lr_pairs(lr_pairs: pd.DataFrame) -> pd.DataFrame:
    """`lr_pairs` with one row per ``lr_name`` (the first, in fit order).

    Two pairs can name the same genes: the LR table's ``PLG^F2R`` row has
    receptor F2RL3, so it duplicates ``PLG^F2RL3`` (IBD). Scriabin labels
    pairs ``ligand=receptor``, so a duplicate would be a second, identical
    column under the same name, which upstream mis-indexes."""
    return lr_pairs.drop_duplicates("lr_name").reset_index(drop=True)


@dataclass
class Inputs:
    """A dataset as the comparators see it: expression on the fit's genes,
    per-cell ``sample``/``celltype``, and the fit's LR pairs, one per
    ``ligand=receptor`` name (`unique_lr_pairs`; `fit_lr_pairs` keeps every
    pair in the fit's order, to align with its LR loadings)."""

    X: Any
    genes: pd.Index
    obs: pd.DataFrame
    lr: pd.DataFrame

    @property
    def samples(self) -> list[str]:
        return sorted(self.obs["sample"].unique())

    def rows(self, sample: str) -> np.ndarray:
        """Row indices of `sample`'s cells."""
        return np.nonzero(self.obs["sample"].to_numpy() == sample)[0]

    def lr_expression(
        self, rows: np.ndarray | None = None, agg: str = "min"
    ) -> tuple[np.ndarray, np.ndarray]:
        """`lr_expression` of the cells in `rows` (all cells when None)."""
        X = self.X if rows is None else self.X[rows]
        return lr_expression(X, self.genes, self.lr, agg)


def load_inputs(
    factors_path: Path,
    adata: anndata.AnnData,
    condition_column: str,
    celltype_column: str,
) -> Inputs:
    """`Inputs` for `adata` (normalized, as the registry loads it) and the
    cccRISE fit in `factors_path`."""
    ds = load_factors_netcdf(factors_path)
    genes = pd.Index(ds.coords["gene"].values.astype(str))
    # Slice X once: an AnnData view would re-slice it on every access
    X = adata[:, genes].X
    obs = pd.DataFrame(
        {
            "sample": adata.obs[condition_column].astype(str).to_numpy(),
            "celltype": adata.obs[celltype_column].astype(str).to_numpy(),
        },
        index=adata.obs_names,
    )
    return Inputs(X, genes, obs, unique_lr_pairs(fit_lr_pairs(ds)))
