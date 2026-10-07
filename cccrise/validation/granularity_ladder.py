"""A cell-type granularity ladder, and how much of cccRISE a cell-type mean sees.

The annotations the comparators use come at two levels (coarse and fine), and
the fine one mostly splits macrophages. To ask what a finer annotation buys a
cell-type pseudobulk method, every coarse type is split evenly instead:

- `subtype_labels`: within each coarse type, bisecting k-means into `m`
  subtypes on that type's own expression (`type_components`: most variable
  genes, then principal components). RISE space is never used, so the ladder cannot be
  circular. A type is split at most ``n_cells // MIN_CELLS`` ways, and a
  cluster under `MIN_CELLS` is merged into its nearest neighbours.
- `random_split`: the same subtype sizes per coarse type, with cells
  assigned at random. It is the control for "more groups", whatever their
  content.

Two statistics need no fits:

- `representability`: the share of a per-cell loading's variance explained
  by its (sample x type) means -- exactly what a pseudobulk tensor at that
  granularity sees -- and by type alone.
- `sender_receiver_separation`: within a cell type, whether the cells
  carrying a component as senders are the ones carrying it as receivers. A
  cell-type method implies they are the same cells.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA

#: Smallest subtype the ladder makes.
MIN_CELLS = 100
N_GENES = 2000
N_PCS = 30
#: Types with fewer cells than this are left out of the separation table.
MIN_TYPE_CELLS = 100
TOP_FRACTION = 0.1


def _pcs(X, n_genes: int = N_GENES, n_pcs: int = N_PCS, seed: int = 0) -> np.ndarray:
    """Principal components of the `n_genes` most variable genes of `X`."""
    X = sparse.csr_array(X) if sparse.issparse(X) else np.asarray(X)
    mean = np.asarray(X.mean(axis=0)).ravel()
    sq = X.multiply(X) if sparse.issparse(X) else X**2
    var = np.asarray(sq.mean(axis=0)).ravel() - mean**2
    genes = np.argsort(-var)[: min(n_genes, X.shape[1])]
    dense = X[:, genes]
    dense = dense.toarray() if sparse.issparse(dense) else dense
    n_pcs = min(n_pcs, dense.shape[0] - 1, dense.shape[1])
    return PCA(n_components=n_pcs, random_state=seed).fit_transform(dense)


def n_subtypes(n_cells: int, m: int, min_cells: int = MIN_CELLS) -> int:
    """How many ways a type of `n_cells` is split at ladder step `m`."""
    return max(1, min(m, n_cells // min_cells))


def type_components(X, coarse: Sequence[str], seed: int = 0) -> dict[str, np.ndarray]:
    """Per coarse type: principal components of its own cells' most variable
    genes (`subtype_labels` clusters on these)."""
    coarse = np.asarray(coarse).astype(str)
    return {
        t: _pcs(X[np.flatnonzero(coarse == t)], seed=seed) for t in np.unique(coarse)
    }


def subtype_labels(
    components: dict[str, np.ndarray],
    coarse: Sequence[str],
    m: int,
    min_cells: int = MIN_CELLS,
    seed: int = 0,
) -> np.ndarray:
    """Per cell: ``"<coarse type> <i>"``, from bisecting k-means of each
    coarse type's `components` (`type_components`) into at most `m` subtypes."""
    coarse = np.asarray(coarse).astype(str)
    labels = coarse.astype(object).copy()
    for t in np.unique(coarse):
        cells = np.flatnonzero(coarse == t)
        k = n_subtypes(len(cells), m, min_cells)
        if k == 1:
            continue
        model = BisectingKMeans(n_clusters=k, random_state=seed, n_init=3)
        clusters = merge_small(
            model.fit_predict(components[t]), components[t], min_cells
        )
        labels[cells] = [f"{t} {c + 1}" for c in clusters]
    return labels.astype(str)


def merge_small(clusters: np.ndarray, points: np.ndarray, min_cells: int) -> np.ndarray:
    """Clusters renumbered 0.., after the smallest cluster under `min_cells`
    is repeatedly dissolved into the nearest remaining centroids (bisecting
    k-means can split off a handful of outlying cells)."""
    clusters = np.asarray(clusters).copy()
    while True:
        ids, counts = np.unique(clusters, return_counts=True)
        if len(ids) == 1 or counts.min() >= min_cells:
            break
        small = ids[np.argmin(counts)]
        keep = ids[ids != small]
        centroids = np.vstack([points[clusters == i].mean(axis=0) for i in keep])
        moving = clusters == small
        dist = ((points[moving, None, :] - centroids[None]) ** 2).sum(axis=2)
        clusters[moving] = keep[np.argmin(dist, axis=1)]
    return np.unique(clusters, return_inverse=True)[1]


def random_split(
    labels: Sequence[str], coarse: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    """`labels` shuffled among the cells of each coarse type: the same
    subtype sizes, with no expression content."""
    labels = np.asarray(labels).astype(str)
    coarse = np.asarray(coarse).astype(str)
    out = labels.copy()
    for t in np.unique(coarse):
        cells = np.flatnonzero(coarse == t)
        out[cells] = labels[rng.permutation(cells)]
    return out


def _group_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each row replaced by its group's mean row."""
    codes, _ = pd.factorize(groups)
    counts = np.bincount(codes)[:, None]
    sums = np.zeros((counts.shape[0], values.shape[1]))
    np.add.at(sums, codes, values)
    return (sums / counts)[codes]


def block_means(
    loadings: np.ndarray, samples: Sequence[str], types: Sequence[str]
) -> np.ndarray:
    """Each cell's loading replaced by its (sample x type) mean: what a
    cell-type pseudobulk method can represent of it."""
    blocks = np.char.add(
        np.char.add(np.asarray(samples).astype(str), "|"), np.asarray(types).astype(str)
    )
    return _group_means(np.asarray(loadings, float), blocks)


def _explained(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per column: 1 - within-group SS / total SS."""
    within = ((values - _group_means(values, groups)) ** 2).sum(axis=0)
    total = ((values - values.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - within / np.where(total > 0, total, 1)


def representability(
    loadings: np.ndarray, samples: Sequence[str], types: Sequence[str]
) -> pd.DataFrame:
    """Per component: variance of the per-cell loading explained by its
    (sample x type) means (``r2_sample_type``) and by type alone
    (``eta2_type``)."""
    loadings = np.asarray(loadings, float)
    samples = np.asarray(samples).astype(str)
    types = np.asarray(types).astype(str)
    blocks = np.char.add(np.char.add(samples, "|"), types)
    return pd.DataFrame(
        {
            "component": np.arange(1, loadings.shape[1] + 1),
            "r2_sample_type": _explained(loadings, blocks),
            "eta2_type": _explained(loadings, types),
            "n_types": len(np.unique(types)),
            "n_blocks": len(np.unique(blocks)),
        }
    )


def sender_receiver_separation(
    sender: np.ndarray, receiver: np.ndarray, types: Sequence[str]
) -> pd.DataFrame:
    """Per (component, type with >= `MIN_TYPE_CELLS` cells): Spearman of the
    sender and receiver loadings over the type's cells, and the Jaccard
    overlap of its top-`TOP_FRACTION` senders and receivers."""
    types = np.asarray(types).astype(str)
    rows = []
    for t in np.unique(types):
        cells = np.flatnonzero(types == t)
        if len(cells) < MIN_TYPE_CELLS:
            continue
        n_top = max(1, round(TOP_FRACTION * len(cells)))
        for c in range(sender.shape[1]):
            s, r = sender[cells, c], receiver[cells, c]
            top_s = set(np.argsort(-s)[:n_top])
            top_r = set(np.argsort(-r)[:n_top])
            rows.append(
                {
                    "component": c + 1,
                    "celltype": t,
                    "n_cells": len(cells),
                    "spearman": float(spearmanr(s, r).statistic),
                    "top_jaccard": len(top_s & top_r) / len(top_s | top_r),
                }
            )
    return pd.DataFrame(rows)


def sample_dominance(labels: Sequence[str], samples: Sequence[str]) -> pd.DataFrame:
    """Per subtype: cells, samples, and the share of cells from its largest
    sample."""
    frame = pd.DataFrame(
        {"label": np.asarray(labels).astype(str), "sample": np.asarray(samples)}
    )
    counts = frame.groupby(["label", "sample"]).size()
    by_label = counts.groupby(level="label")
    return pd.DataFrame(
        {
            "n_cells": by_label.sum(),
            "n_samples": by_label.size(),
            "top_sample_share": by_label.max() / by_label.sum(),
        }
    ).reset_index()
