"""One pass over the cells, then any cell-type labelling's pseudobulk.

`benchmark.pseudobulk_tensor` averages every gene over the cells of each
(sample, cell type) and scores LR pairs on those means. When a labelling
only merges a finer annotation's types into groups, each group's mean is the
count-weighted average of its members' means, so one pass at the finest
annotation (`group_sums`: per (sample, fine type) cell counts and expression
sums of the genes LR pairs use) gives every such labelling's pseudobulk
(`labelled_matrices`) without touching the cells again. A (sample, group)
with no cells is zero, as `utils.pseudobulk_X` makes it.

`loading_sums` does the same for cccRISE's per-cell sender and receiver
loadings, so per-type profiles and active-cell counts can be formed for any
labelling too."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .agreement import sample_scaled

#: A cell is active in a component above this x the component's 99th
#: percentile over all cells (as `run_lr_concordance.ACTIVE_TAU`).
ACTIVE_TAU = 0.5


def lr_genes(lr_pairs: pd.DataFrame, complex_sep: str = "&") -> list[str]:
    """Every gene a ligand or receptor of `lr_pairs` names, complex subunits
    included, sorted."""
    names = pd.concat([lr_pairs["ligand"], lr_pairs["receptor"]]).astype(str)
    return sorted({g for name in names for g in name.split(complex_sep)})


def _indicator(samples: np.ndarray, types: np.ndarray, sample_order, type_order):
    """``(n_samples * n_types, n_cells)`` 0/1 matrix of each cell's group,
    groups in (sample, type) order."""
    s = pd.Index(sample_order).get_indexer(samples)
    t = pd.Index(type_order).get_indexer(types)
    if (s < 0).any() or (t < 0).any():
        raise ValueError("cells with a sample or type outside the given orders")
    rows = s * len(type_order) + t
    n = len(samples)
    return sparse.csr_matrix(
        (np.ones(n), (rows, np.arange(n))),
        shape=(len(sample_order) * len(type_order), n),
    )


def group_sums(
    X, samples: Sequence[str], types: Sequence[str], sample_order, type_order
) -> tuple[np.ndarray, np.ndarray]:
    """Per (sample, type): its cell count, ``(S, T)``, and the column sums of
    `X` (cells x genes, dense or sparse) over its cells, ``(S, T, genes)``."""
    samples, types = np.asarray(samples).astype(str), np.asarray(types).astype(str)
    ind = _indicator(samples, types, sample_order, type_order)
    shape = (len(sample_order), len(type_order))
    counts = np.asarray(ind.sum(axis=1)).reshape(shape)
    sums = np.asarray((ind @ X).todense() if sparse.issparse(X) else ind @ X)
    return counts, sums.reshape(*shape, -1)


def loading_sums(
    loadings: np.ndarray,
    samples: Sequence[str],
    types: Sequence[str],
    sample_order,
    type_order,
) -> tuple[np.ndarray, np.ndarray]:
    """Per (sample, type) and component: the sum of the sample-scaled
    loadings (`agreement.sample_scaled`) and the number of cells active in
    the component (> `ACTIVE_TAU` x its 99th percentile over all cells)."""
    scaled = sample_scaled(loadings, samples)
    active = (scaled > ACTIVE_TAU * np.percentile(scaled, 99, axis=0)).astype(float)
    _, sums = group_sums(scaled, samples, types, sample_order, type_order)
    _, n_active = group_sums(active, samples, types, sample_order, type_order)
    return sums, n_active


def merge_groups(values: np.ndarray, type_order, mapping: Mapping[str, str]):
    """`values` (S, T, ...) summed over the fine types each group of
    `mapping` (fine type -> group) holds; returns them and the sorted group
    names (``np.unique`` order, as `pseudobulk_X` uses)."""
    groups = np.unique([mapping[t] for t in type_order])
    member = np.zeros((len(type_order), len(groups)))
    for i, t in enumerate(type_order):
        member[i, np.searchsorted(groups, mapping[t])] = 1
    return np.einsum("st...,tg->sg...", values, member), groups


def labelled_matrices(
    counts: np.ndarray,
    sums: np.ndarray,
    type_order,
    genes: Sequence[str],
    mapping: Mapping[str, str],
) -> tuple[list[pd.DataFrame], np.ndarray]:
    """The pseudobulk mean matrices of the labelling `mapping` (fine type ->
    group): one genes x groups DataFrame per sample, as
    `utils.pseudobulk_X(type="mean")` returns them, and the group names."""
    group_counts, groups = merge_groups(counts, type_order, mapping)
    group_sums_, _ = merge_groups(sums, type_order, mapping)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(
            group_counts[..., None] > 0, group_sums_ / group_counts[..., None], 0.0
        )
    return [
        pd.DataFrame(means[s].T, index=list(genes), columns=groups)
        for s in range(means.shape[0])
    ], groups
