"""Python ports of the WGCNA, dynamicTreeCut and R base-stats routines that
Scriabin's interaction-program discovery calls.

Each function reproduces one upstream routine, named in its docstring, closely
enough that `cccrise/tests/test_comparators_rcheck.py` can check it against
the R original on identical input:

- `pick_soft_threshold`: ``WGCNA::pickSoftThreshold.fromSimilarity``
  (WGCNA 1.74, ``R/Functions.R``), with ``scaleFreeFitIndex``.
- `signed_tom`: ``WGCNA::TOMsimilarity(adj, TOMType = "signed")``, the C
  routine ``tomSimilarityFromAdj`` (``src/networkFunctions.c``).
- `intramodular_connectivity`: ``WGCNA::intramodularConnectivity``.
- `hclust_complete`: ``stats::hclust(method = "complete")`` /
  ``flashClust``, returning R's ``merge``/``height``/``order``
  (``hcass2`` conventions).
- `cutree_height`: ``stats::cutree(tree, h = ...)``.
- `cutree_dynamic_tree`: ``dynamicTreeCut::cutreeDynamic(method = "tree")``,
  i.e. ``cutreeDynamicTree`` with ``deepSplit = FALSE`` (dynamicTreeCut
  1.63-1, ``R/cutreeDynamic.R``).
- `r_cut_breaks` / `cut_number`: base ``cut(x, n)`` and
  ``ggplot2::cut_number`` (as used by Seurat's ``AddModuleScore``).

Indices are 0-based in Python except where a routine's R output is itself an
index (``merge``, ``order``), which keep R's 1-based values so the two can be
compared directly.
"""

from dataclasses import dataclass

import numpy as np

#: ``pickSoftThreshold``'s default power grid.
DEFAULT_POWERS = (*range(1, 11), *range(12, 21, 2))


def r_cut_breaks(x: np.ndarray, n_breaks: int) -> np.ndarray:
    """Breaks of base R ``cut(x, breaks = n_breaks)``: equal-width over the
    range, with the outer two widened by 1/1000 of the range."""
    lo, hi = float(np.min(x)), float(np.max(x))
    dx = hi - lo
    if dx == 0:
        dx = abs(lo) if lo != 0 else 1.0
        return np.linspace(lo - dx / 1000, hi + dx / 1000, n_breaks + 1)
    breaks = np.linspace(lo, hi, n_breaks + 1)
    breaks[0], breaks[-1] = lo - dx / 1000, hi + dx / 1000
    return breaks


def _lm_r2_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """``summary(lm(y ~ x))``'s R^2 and slope."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum(resid**2) / ss_tot, float(coef[1])


def scale_free_fit_index(k: np.ndarray, n_breaks: int = 10) -> tuple[float, float]:
    """``WGCNA::scaleFreeFitIndex(k, nBreaks)`` (``removeFirst = FALSE``):
    R^2 and slope of log10 p(k) on log10 k over `n_breaks` connectivity bins."""
    breaks = r_cut_breaks(k, n_breaks)
    # cut() is right-closed: bin b holds breaks[b] < k <= breaks[b + 1]
    bins = np.clip(np.searchsorted(breaks, k, side="left") - 1, 0, n_breaks - 1)
    counts = np.bincount(bins, minlength=n_breaks).astype(float)
    sums = np.bincount(bins, weights=k, minlength=n_breaks)
    with np.errstate(invalid="ignore", divide="ignore"):
        dk = sums / counts
    mids_breaks = np.linspace(k.min(), k.max(), n_breaks + 1)
    mids = (mids_breaks[:-1] + mids_breaks[1:]) / 2
    dk = np.where(np.isnan(dk) | (dk == 0), mids, dk)
    p_dk = counts / len(k)
    r2, slope = _lm_r2_slope(np.log10(p_dk + 1e-9), np.log10(dk))
    return r2, slope


def pick_soft_threshold(
    similarity: np.ndarray,
    r2_cut: float = 0.85,
    powers: tuple[int, ...] = DEFAULT_POWERS,
    n_breaks: int = 10,
) -> tuple[int | None, np.ndarray]:
    """``pickSoftThreshold.fromSimilarity``: the lowest power whose scale-free
    fit R^2 exceeds `r2_cut` (None if none does), and the per-power
    (R^2, slope) table.

    Connectivity is ``colSums(sim^p, na.rm = TRUE) - 1`` with the diagonal set
    to 1, as upstream.
    """
    sim = np.array(similarity, dtype=float)
    np.fill_diagonal(sim, 1.0)
    table = np.empty((len(powers), 2))
    for i, p in enumerate(sorted(powers)):
        k = np.nansum(sim**p, axis=0) - 1.0
        table[i] = scale_free_fit_index(k, n_breaks)
    passed = np.nonzero(table[:, 0] > r2_cut)[0]
    return (sorted(powers)[passed[0]] if len(passed) else None), table


def signed_tom(adj: np.ndarray) -> np.ndarray:
    """``TOMsimilarity(adj, TOMType = "signed", TOMDenom = "min")``.

    With the diagonal of `adj` set to 1 and ``conn_i = sum_g |adj_ig|``,
    ``TOM_ij = |(adj @ adj)_ij - adj_ij| / (min(conn_i, conn_j) - |adj_ij|)``
    (0 where the denominator is 0); the diagonal is 1. NA adjacencies are 0.
    """
    a = np.nan_to_num(np.array(adj, dtype=float), nan=0.0)
    np.fill_diagonal(a, 1.0)
    conn = np.abs(a).sum(axis=0)
    num = np.abs(a @ a - a)
    den = np.minimum.outer(conn, conn) - np.abs(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        tom = np.where(den == 0, 0.0, num / den)
    np.fill_diagonal(tom, 1.0)
    return tom


def intramodular_connectivity(adj: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``intramodularConnectivity(adj, colors)``: (kTotal, kWithin) per node,
    with the diagonal zeroed; kWithin is 0 in modules of fewer than 3 nodes."""
    a = np.array(adj, dtype=float)
    np.fill_diagonal(a, 0.0)
    k_total = np.nansum(a, axis=0)
    k_within = np.zeros(len(labels))
    for lab in np.unique(labels):
        rows = labels == lab
        if rows.sum() >= 3:
            k_within[rows] = np.nansum(a[np.ix_(rows, rows)], axis=0)
    return np.column_stack([k_total, k_within])


@dataclass
class HClust:
    """R ``hclust`` object: 1-based ``merge`` (negative = singleton), ``height``
    and 1-based leaf ``order``."""

    merge: np.ndarray
    height: np.ndarray
    order: np.ndarray


def _row_nearest(d: np.ndarray, active: np.ndarray, k: int) -> tuple[float, int]:
    """Smallest distance from `k` to an active cluster of higher index, and
    the lowest such index (inf, -1 when none)."""
    row = np.where(active[k + 1 :], d[k, k + 1 :], np.inf)
    if row.size == 0:
        return np.inf, -1
    j = int(np.argmin(row))
    return float(row[j]), k + 1 + j


def _complete_linkage_steps(
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """hclust.f's agglomeration for complete linkage: at each step merge the
    closest pair of active clusters (ties: lowest i, then lowest j), keep the
    lower index for the merged cluster, and update by the max rule. Returns
    1-based (IA, IB, height) as hclust.f passes them to hcass2.

    Each cluster's nearest higher-index neighbour is cached (hclust.f's NN
    list) and a row is rescanned only when the merge can change it, so a
    step costs O(n) rather than O(n^2); the choice, ties included, is the
    full scan's (`cccrise/tests/test_comparators_wgcna.py`)."""
    n = dist.shape[0]
    d = np.array(dist, dtype=float)
    np.fill_diagonal(d, np.inf)
    active = np.ones(n, dtype=bool)
    nn_d, nn_j = np.empty(n), np.empty(n, dtype=int)
    for k in range(n):
        nn_d[k], nn_j[k] = _row_nearest(d, active, k)
    ia, ib, height = np.empty(n - 1, int), np.empty(n - 1, int), np.empty(n - 1)
    for step in range(n - 1):
        i = int(np.argmin(nn_d))  # lowest i among the smallest
        j = int(nn_j[i])
        ia[step], ib[step], height[step] = i + 1, j + 1, nn_d[i]
        merged = np.maximum(d[i], d[j])
        d[i, :], d[:, i] = merged, merged
        d[i, i] = np.inf
        active[j] = False
        d[j, :], d[:, j] = np.inf, np.inf
        nn_d[j], nn_j[j] = np.inf, -1
        # rows below i whose nearest was i or j, or that now tie/beat it via i
        lower = np.flatnonzero(
            active[:i] & ((nn_j[:i] == i) | (nn_j[:i] == j) | (merged[:i] <= nn_d[:i]))
        )
        # rows between i and j that pointed at j
        middle = i + 1 + np.flatnonzero(active[i + 1 : j] & (nn_j[i + 1 : j] == j))
        for k in (*lower, i, *middle):
            nn_d[k], nn_j[k] = _row_nearest(d, active, int(k))
    return ia, ib, height


def _hcass2(ia: np.ndarray, ib: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """hclust.f's ``HCASS2``: relabel clusters as R's ``merge`` matrix and
    derive the leaf ``order``."""
    n = len(ia) + 1
    iia, iib = ia.copy(), ib.copy()
    for i in range(n - 2):
        k = min(ia[i], ib[i])
        later = np.arange(i + 1, n - 1)
        iia[later[ia[later] == k]] = -(i + 1)
        iib[later[ib[later] == k]] = -(i + 1)
    iia, iib = -iia, -iib
    for i in range(n - 1):
        if iia[i] > 0 and iib[i] < 0:
            iia[i], iib[i] = iib[i], iia[i]
        if iia[i] > 0 and iib[i] > 0:
            iia[i], iib[i] = min(iia[i], iib[i]), max(iia[i], iib[i])
    order = [iia[n - 2], iib[n - 2]]
    for i in range(n - 2, 0, -1):
        j = order.index(i)
        order[j : j + 1] = [iia[i - 1], iib[i - 1]]
    return np.column_stack([iia, iib]), -np.asarray(order)


def hclust_complete(dist: np.ndarray) -> HClust:
    """``hclust(as.dist(dist), method = "complete")`` (and ``flashClust``,
    which reproduces it)."""
    ia, ib, height = _complete_linkage_steps(dist)
    merge, order = _hcass2(ia, ib)
    return HClust(merge=merge, height=height, order=order)


def cutree_height(tree: HClust, h: float) -> np.ndarray:
    """``cutree(tree, h = h)``: apply every merge with height <= `h`; groups are
    numbered 1.. by first appearance in observation order."""
    n = len(tree.order)
    n_merges = int(np.sum(tree.height <= h))
    members: list[list[int]] = []
    group = np.arange(n)
    for a, b in tree.merge[:n_merges]:
        cells = [
            *([-a - 1] if a < 0 else members[a - 1]),
            *([-b - 1] if b < 0 else members[b - 1]),
        ]
        members.append(cells)
        group[cells] = min(cells)
    _, first = np.unique(group, return_index=True)
    rank = {g: r + 1 for r, g in enumerate(group[np.sort(first)])}
    return np.array([rank[g] for g in group])


def _sorted_label_counts(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``-sort(-table(labels))``: labels by decreasing count, ties in
    increasing label order (``table`` sorts levels; ``sort`` is stable)."""
    values, counts = np.unique(labels, return_counts=True)
    idx = np.argsort(-counts, kind="stable")
    return values[idx], counts[idx]


def _cut_tree_static(tree: HClust, height: float, min_size: int) -> np.ndarray:
    """dynamicTreeCut's ``.cutTreeStatic``: static-cut groups of at least
    `min_size`, renumbered 1.. by decreasing size, others -1."""
    labels = cutree_height(tree, height)
    names, counts = _sorted_label_counts(labels)
    out = np.full(len(labels), -1)
    for i, name in enumerate(names[counts >= min_size]):
        out[labels == name] = i + 1
    return out


def _locate_cluster(labels: np.ndarray) -> np.ndarray | None:
    """``.locateCluster``: 1-based (start, end) runs of positive labels."""
    n = len(labels)
    shifted = np.concatenate([[labels[0]], labels[:-1]])
    diff = np.abs(labels - shifted)
    if labels[0] > 0:
        diff[0] = 1
    if labels[-1] > 0:
        diff[-1] = 1
    flags = np.nonzero(diff > 0)[0] + 1
    if len(flags) == 0:
        return None
    runs = []
    for i in range(len(flags) - 1):
        idx = flags[i]
        if labels[idx - 1] > 0:
            if flags[i + 1] == n:
                runs.append((idx, flags[i + 1]))
                break
            runs.append((idx, flags[i + 1] - 1))
    return np.array(runs) if runs else None


def _runlength_sign(seq: np.ndarray) -> int:
    """``.runlengthSign(seq, leftOrright = -1, mysign = -1)``: run length of
    non-positive values at the right end."""
    positive = np.nonzero(seq[::-1] > 0)[0]
    return int(positive[0]) if len(positive) else 0


def _initial_cuts(heights: np.ndarray, comp: float) -> np.ndarray:
    """``.processIndividualCluster``'s candidate cut points: where the height
    crosses `comp` downward, plus both ends."""
    diff = heights - comp
    shifted = np.concatenate([diff[1:], diff[-1:]])
    cuts = (diff < 0) & (shifted > 0)
    cuts[0] = cuts[-1] = True
    return cuts


def _prune_short_tails(
    cuts: np.ndarray, heights: np.ndarray, comp: float, min_size: int
) -> np.ndarray:
    """Drop interior cut points whose tail run (non-positive height
    differences at the right end of the preceding segment) is shorter than
    ``min_size // 3 + 1``."""
    diff = heights - comp
    index = np.nonzero(cuts)[0]
    min_tail = int(min_size / 3) + 1
    pruned = cuts.copy()
    for i in range(1, len(index) - 1):
        if _runlength_sign(diff[index[i - 1] + 1 : index[i] + 1]) < min_tail:
            pruned[index[i]] = False
    return pruned


def _merge_small_children(
    cuts: np.ndarray, heights: np.ndarray, min_attach: int
) -> None:
    """``.processIndividualCluster``'s attach pass: a candidate cluster lower
    than its left neighbour and smaller than `min_attach` merges into it."""
    if cuts.sum() <= 3:
        return
    cur = 1
    while True:
        index = np.nonzero(cuts)[0]
        if cur > len(index) - 2:
            break
        pre = heights[index[cur - 1] + 1 : index[cur] + 1]
        nxt = heights[index[cur] + 1 : index[cur + 1] + 1]
        if nxt.mean() < pre.mean() and len(nxt) < min_attach:
            cuts[index[cur]] = False
        else:
            cur += 1


def _process_individual_cluster(
    heights: np.ndarray, min_size: int, min_attach: int, use_mean: int = 0
) -> np.ndarray:
    """``.processIndividualCluster``: split one static cluster (its merge
    heights in dendrogram order) into 1-based (start, end) sub-runs."""
    n = len(heights)
    mean, hmax, hmin = heights.mean(), heights.max(), heights.min()
    comp = {0: mean, 1: (mean + hmax) / 2, -1: (mean + hmin) / 2}[use_mean]
    initial = _initial_cuts(heights, comp)
    if initial.sum() == 2:
        if use_mean == -1:
            return np.array([[1, n]])
        return _process_individual_cluster(
            heights, min_size, min_attach, 1 if use_mean == 0 else -1
        )
    cuts = _prune_short_tails(initial, heights, comp, min_size)
    _merge_small_children(cuts, heights, min_attach)
    index = np.nonzero(cuts)[0] + 1
    if len(index) > 2 and index[-1] - index[-2] + 1 < min_size:
        cuts[index[-2] - 1] = False
    index = np.nonzero(cuts)[0] + 1
    index[0] -= 1
    runs = (
        np.column_stack([index[:-1] + 1, index[1:]])
        if len(index) > 2
        else np.array([[1, n]])
    )
    if len(runs) == 1 and use_mean in (0, 1):
        return _process_individual_cluster(
            heights, min_size, min_attach, 1 if use_mean == 0 else -1
        )
    return runs


def _assign_module_number(labels: np.ndarray, min_size: int) -> np.ndarray:
    """``.assignModuleNumber``: renumber positive labels 1.. by decreasing size,
    dropping groups smaller than `min_size` to 0."""
    names, counts = _sorted_label_counts(labels[labels > 0])
    out = np.zeros(len(labels), dtype=int)
    for i, name in enumerate(names[counts >= min_size]):
        out[labels == name] = i + 1
    return out


def cutree_dynamic_tree(tree: HClust, cut_height: float, min_size: int) -> np.ndarray:
    """``cutreeDynamic(dendro, cutHeight, minClusterSize, method = "tree")``,
    i.e. ``cutreeDynamicTree(deepSplit = FALSE)``. Returns module labels in
    observation order: 0 = unassigned, 1.. by decreasing module size."""
    if cut_height > tree.height.max():
        cut_height = 0.99 * tree.height.max()
    n = len(tree.order)
    static = _cut_tree_static(tree, cut_height, min_size)
    # Height at which each singleton joins the tree, in observation order
    join = np.empty(n)
    for col in (0, 1):
        singles = tree.merge[:, col] < 0
        join[-tree.merge[singles, col] - 1] = tree.height[singles]
    order = tree.order - 1
    heights_ord, static_ord = join[order], static[order]
    runs = _locate_cluster(static_ord)
    if runs is None:
        return np.zeros(n, dtype=int)
    assign = np.zeros(n, dtype=int)
    count = 1
    for start, end in runs:
        sub = _process_individual_cluster(
            heights_ord[start - 1 : end], min_size, 2 * min_size
        )
        for s, e in sub + start - 1:
            if e - s + 1 >= min_size:
                assign[s - 1 : e] = count
                count += 1
    labels_ord = _assign_module_number(assign, min_size)
    out = np.empty(n, dtype=int)
    out[order] = labels_ord
    return out


def r_quantile7(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R ``quantile(type = 7)``, NumPy's default "linear" method."""
    return np.quantile(np.asarray(x, float), probs)


def cut_number(x: np.ndarray, n: int) -> np.ndarray:
    """``ggplot2::cut_number(x, n, labels = FALSE, right = FALSE)``: 1-based
    bins with roughly equal counts (type-7 quantile breaks, left-closed,
    the last bin closed on both sides)."""
    breaks = r_quantile7(x, np.linspace(0, 1, n + 1))
    if len(np.unique(breaks)) != len(breaks):
        raise ValueError("cut_number: insufficient data values for distinct breaks")
    bins = np.searchsorted(breaks, x, side="right")
    return np.clip(bins, 1, n)
