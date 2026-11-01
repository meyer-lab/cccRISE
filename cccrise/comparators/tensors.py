"""Communication tensors of the annotation-pooled comparators.

Every comparator pools cells by (sample, cell type) before scoring, so each
tensor is ``(sample, sender type, receiver type, LR pair)``. They differ in
how cells are pooled and pairs scored:

- **Pseudobulk CPD** (`pseudobulk_tensor`): mean expression per (sample,
  type), the LR score the product of ligand and receptor, as cccRISE scores
  its eigen-states. A type absent from a sample is zero.
- **STACCato** (`staccato_tensor`), as its authors build it with LIANA
  (``rank_aggregate.by_sample`` and ``to_tensor_c2c(score_key="expr_prod")``):
  the same product of mean expressions, with LIANA's default filters. A pair
  scores zero in a (sample, sender, receiver) where its ligand or receptor
  (any subunit) is detected in under `STACCATO_EXPR_PROP` of the sender's or
  receiver's cells, or where either type has under `STACCATO_MIN_CELLS` cells
  in that sample. Unlike the authors (``how="outer_cells",
  outer_fraction=0.8``), every type is kept; absent entries are zero, as the
  authors' R code reads them.
- **Tensor-cell2cell** (`tensor_cell2cell_tensor`), as its paper scores it
  (Armingol et al. 2022; ``aggregate_single_cells(method="nn_cell_fraction")``
  and ``InteractionTensor(communication_score="expression_mean")``): the
  fraction of a type's cells expressing each gene, the LR score the mean of
  ligand and receptor. By default every type present in any sample is kept
  and the entries of a type absent from a sample are masked
  (``how="outer_cells"``, the package's option for such types). The paper
  itself kept only the types present in every sample (``how="inner"``, no
  mask), which on ALAD drops most immune and epithelial types; that
  construction is kept as a sensitivity check.

A complex's expression is the minimum over its subunits, and an LR pair is
kept when every gene it names was measured, as in Tensor-cell2cell. Pairs are
scored by the table's ``ligand``/``receptor`` columns and labelled by its
``interaction_symbol``, as cccRISE scores and labels its `lr_pair` mode.
"""

from dataclasses import dataclass

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

COMPLEX_SEP = "&"


@dataclass(frozen=True)
class PooledTensor:
    """A ``(sample, sender, receiver, lr)`` tensor and its labels. `mask` is
    1 where an entry is observed and 0 where it is missing; None when every
    entry is observed."""

    values: np.ndarray
    mask: np.ndarray | None
    samples: np.ndarray
    celltypes: np.ndarray
    lr_pairs: np.ndarray


def subunits(name: str) -> list[str]:
    return name.split(COMPLEX_SEP)


def scored_lr_pairs(lr_pairs: pd.DataFrame, genes) -> pd.DataFrame:
    """The ``ligand``/``receptor`` rows whose every (sub)unit is in `genes`,
    in table order, with their ``interaction_symbol`` as ``lr_pair``."""
    measured = set(genes)
    names = lr_pairs[["ligand", "receptor", "interaction_symbol"]].astype(str)
    keep = names[["ligand", "receptor"]].apply(
        lambda row: all(g in measured for side in row for g in subunits(side)),
        axis=1,
    )
    kept = names[keep].reset_index(drop=True)
    return kept.rename(columns={"interaction_symbol": "lr_pair"})


def group_profiles(
    adata: anndata.AnnData,
    sample_column: str,
    celltype_column: str,
    genes: list[str],
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per (sample, cell type), its cell count ``(S, T)`` and each gene's mean
    expression (``aggregation="mean"``) or fraction of expressing cells
    (``"fraction"``) ``(S, T, genes)``; zero where a type has no cells.
    Samples and types are in sorted order, and are returned too."""
    if aggregation not in ("mean", "fraction"):
        raise ValueError(
            f"aggregation must be 'mean' or 'fraction', got {aggregation!r}"
        )
    samples = np.unique(adata.obs[sample_column].astype(str))
    celltypes = np.unique(adata.obs[celltype_column].astype(str))
    s = pd.Index(samples).get_indexer(adata.obs[sample_column].astype(str))
    t = pd.Index(celltypes).get_indexer(adata.obs[celltype_column].astype(str))
    groups = sparse.csr_matrix(
        (np.ones(adata.n_obs), (s * len(celltypes) + t, np.arange(adata.n_obs))),
        shape=(len(samples) * len(celltypes), adata.n_obs),
    )
    X = sparse.csr_matrix(adata[:, genes].X)
    if aggregation == "fraction":
        X = (X > 0).astype(float)
    counts = np.asarray(groups.sum(axis=1)).ravel()
    sums = np.asarray((groups @ X).todense())
    profiles = sums / np.maximum(counts, 1)[:, None]
    shape = (len(samples), len(celltypes))
    return (
        counts.reshape(shape),
        profiles.reshape(*shape, len(genes)),
        samples,
        celltypes,
    )


def _complex_expression(profiles: np.ndarray, genes: list[str], names) -> np.ndarray:
    """``(S, T, len(names))``: each (complex) name's minimum over its subunits."""
    column = {g: i for i, g in enumerate(genes)}
    return np.stack(
        [profiles[..., [column[g] for g in subunits(n)]].min(axis=-1) for n in names],
        axis=-1,
    )


def pooled_tensor(
    adata: anndata.AnnData,
    sample_column: str,
    celltype_column: str,
    lr_pairs: pd.DataFrame,
    aggregation: str,
    score: str,
    mask_absent: bool,
) -> PooledTensor:
    """The pooled communication tensor of `adata` grouped by
    `celltype_column` within `sample_column`.

    `aggregation` is `group_profiles`'; `score` is ``"product"`` (ligand x
    receptor) or ``"mean"`` (their average). With `mask_absent`, the entries
    of a type absent from a sample, as sender or receiver, are masked.
    """
    pairs = scored_lr_pairs(lr_pairs, adata.var_names)
    genes = sorted(
        {
            g
            for n in pd.concat([pairs["ligand"], pairs["receptor"]])
            for g in subunits(n)
        }
    )
    counts, profiles, samples, celltypes = group_profiles(
        adata, sample_column, celltype_column, genes, aggregation
    )
    ligand = _complex_expression(profiles, genes, pairs["ligand"])[:, :, None, :]
    receptor = _complex_expression(profiles, genes, pairs["receptor"])[:, None, :, :]
    if score == "product":
        values = ligand * receptor
    elif score == "mean":
        values = (ligand + receptor) / 2
    else:
        raise ValueError(f"score must be 'product' or 'mean', got {score!r}")
    mask = None
    if mask_absent:
        present = counts > 0
        mask = (present[:, :, None] & present[:, None, :]).astype(float)
        mask = np.broadcast_to(mask[..., None], values.shape).copy()
        values = values * mask
    return PooledTensor(values, mask, samples, celltypes, pairs["lr_pair"].to_numpy())


def pseudobulk_tensor(adata, sample_column, celltype_column, lr_pairs) -> PooledTensor:
    """Pseudobulk CPD's and STACCato's tensor (module docstring)."""
    return pooled_tensor(
        adata,
        sample_column,
        celltype_column,
        lr_pairs,
        aggregation="mean",
        score="product",
        mask_absent=False,
    )


#: LIANA's defaults STACCato's authors build with: the least share of a type's
#: cells detecting a ligand or receptor (every subunit), and the fewest cells
#: a type needs in a sample to count there.
STACCATO_EXPR_PROP = 0.1
STACCATO_MIN_CELLS = 5


def staccato_tensor(
    adata,
    sample_column,
    celltype_column,
    lr_pairs,
    expr_prop: float = STACCATO_EXPR_PROP,
    min_cells: int = STACCATO_MIN_CELLS,
) -> PooledTensor:
    """STACCato's tensor (module docstring): `pseudobulk_tensor`'s scores,
    zero where LIANA's `expr_prop` and `min_cells` rules would filter."""
    pairs = scored_lr_pairs(lr_pairs, adata.var_names)
    genes = sorted(
        {
            g
            for n in pd.concat([pairs["ligand"], pairs["receptor"]])
            for g in subunits(n)
        }
    )
    counts, means, samples, celltypes = group_profiles(
        adata, sample_column, celltype_column, genes, "mean"
    )
    _, detected, _, _ = group_profiles(
        adata, sample_column, celltype_column, genes, "fraction"
    )
    ligand, receptor = (
        _complex_expression(profile, genes, pairs[side])
        for profile, side in ((means, "ligand"), (means, "receptor"))
    )
    ligand_detected, receptor_detected = (
        _complex_expression(detected, genes, pairs[side])
        for side in ("ligand", "receptor")
    )
    present = counts >= min_cells
    keep = (ligand_detected[:, :, None, :] >= expr_prop) & (
        receptor_detected[:, None, :, :] >= expr_prop
    )
    keep &= (present[:, :, None] & present[:, None, :])[..., None]
    values = np.where(keep, ligand[:, :, None, :] * receptor[:, None, :, :], 0.0)
    return PooledTensor(values, None, samples, celltypes, pairs["lr_pair"].to_numpy())


def tensor_cell2cell_tensor(
    adata, sample_column, celltype_column, lr_pairs, how: str = "outer_cells"
) -> PooledTensor:
    """Tensor-cell2cell's tensor (module docstring): every type, absent ones
    masked (``how="outer_cells"``), or only the types present in every
    sample, unmasked (``"inner"``)."""
    if how not in ("outer_cells", "inner"):
        raise ValueError(f"how must be 'outer_cells' or 'inner', got {how!r}")
    if how == "inner":
        present = pd.crosstab(adata.obs[sample_column], adata.obs[celltype_column])
        shared = present.columns[(present > 0).all()]
        adata = adata[adata.obs[celltype_column].isin(shared)]
    return pooled_tensor(
        adata,
        sample_column,
        celltype_column,
        lr_pairs,
        aggregation="fraction",
        score="mean",
        mask_absent=how == "outer_cells",
    )
