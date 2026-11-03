"""Python re-implementation of Scriabin's interaction-program workflow
(BlishLab/scriabin, R; pinned at ``bce695b``).

Scriabin (Wilk et al., Nat Biotech 2023) scores communication between every
pair of cells as the cell-cell interaction matrix (CCIM),
``M[(s, r), l] = sqrt(L[s, l] * R[r, l])`` (`ccim`, upstream
``GenerateCCIM``). Its comparative workflow for multi-sample data is
**interaction-program discovery**:

1. per sample, a signed co-variation network of LR pairs across cell pairs
   (stratified 300-cell CCIMs, averaged), soft-thresholded, turned into a
   signed topological overlap and cut into modules with a dynamic tree cut
   (`interaction_programs`, upstream ``InteractionPrograms``);
2. modules of all samples are merged while their Jaccard overlap exceeds 0.15
   (`merge_programs`, ``FindAllInteractionPrograms``);
3. each merged module is tested per sample against random LR sets of the
   same size and kept if significant in at least one sample
   (`program_significance`, ``InteractionProgramSignificance``);
4. every cell gets a sender score on the module's ligands and a receiver
   score on its receptors (`score_programs`, ``ScoreInteractionPrograms`` ->
   Seurat ``AddModuleScore``, ported as `module_score`).

Comparing against cccRISE: an interaction program is the Scriabin output a
cccRISE component maps onto. Its LR set corresponds to the component's LR
loading ``D``, its per-cell sender and receiver scores to the per-cell
weights ``sc_B`` / ``rc_C`` (unscaled), and its per-sample activity to ``A``.
Per-cell-pair (CCIM edge) concordance is not used, since cccRISE's per-edge
reconstruction does not recover per-cell expression
(`notes/ANALYSIS_RECORD.md` §4.5, E16/E17).

The WGCNA / dynamicTreeCut / hclust routines it calls are ported in
`cccrise.comparators._wgcna`. Every stage is checked against the R source by
`rcheck/compare.py` (`cccrise/tests/test_comparators_rcheck.py`).

`ProgramSettings` exposes the module-discovery parameters for the
parameter sweep. Its defaults are upstream's. Two of them, the soft-power
cap and the tree-cut quantile, are hard-coded upstream (``softPower > 3``
falls back to 1; ``cutHeight = quantile(height, 0.5)``, while the
``tree.cut.quantile`` argument is ignored), so R runs at other values use the
patched ``rcheck/patched_ip.R``.

Differences from upstream, all deliberate:

- **Inputs** are cccRISE's (`cccrise.comparators.inputs`): its normalized,
  uncentred expression instead of SCT ``data``; the fit's LR pairs instead of
  OmniPath; complexes aggregated with ``min`` (upstream drops them).
- **Subsampling a type with fewer cells than its quota.** Upstream draws
  ``max(round(300 * n_type / N), 3)`` cells per type with ``sample_n``, which
  errors when a type has fewer cells than that; here the draw is capped at the
  type's size.
- **Random number streams** are NumPy's, so subsamples, random LR sets and
  ``AddModuleScore`` control genes are not R's draws. Each stochastic step
  accepts its draws as an argument so the R checks can feed both sides the
  same ones.
- **Not reproduced:** NicheNet ligand-activity weighting (off by default and
  not used by interaction programs), the summarized interaction graph,
  binning and perturbed-bin tests; none of them feed interaction programs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import norm

from . import _wgcna

#: InteractionPrograms defaults (upstream signature and body).
ITERATE_THRESHOLD = 300
MIN_CELL = 3
R2_CUTOFF = 0.6
MIN_MODULE_SIZE = 5
MAX_SOFT_POWER = 3
CUT_HEIGHT_QUANTILE = 0.5
#: FindAllInteractionPrograms / InteractionProgramSignificance defaults.
SIM_THRESHOLD = 0.15
N_REPLICATE = 10_000
#: AddModuleScore defaults.
N_BIN = 24
N_CTRL = 100


@dataclass(frozen=True)
class ProgramSettings:
    """``InteractionPrograms``' module-discovery parameters; the defaults are
    upstream's."""

    #: ``pickSoftThreshold`` R^2 cut (``r2_cutoff``).
    r2_cutoff: float = R2_CUTOFF
    #: Highest soft power accepted before falling back to 1; None: no cap.
    max_soft_power: int | None = MAX_SOFT_POWER
    #: A fixed soft power (upstream ``softPower``); None picks it.
    soft_power: int | None = None
    #: ``cutreeDynamic`` ``minClusterSize`` (upstream ``min.size``).
    min_module_size: int = MIN_MODULE_SIZE
    #: ``cutHeight`` as a quantile of the tree's merge heights.
    cut_height_quantile: float = CUT_HEIGHT_QUANTILE
    #: Subsamples to average (upstream ``n.iterate``); None:
    #: ``round(sqrt(N / iterate_threshold))``.
    n_iterate: int | None = None
    #: Samples of at most this many cells use the full CCIM.
    iterate_threshold: int = ITERATE_THRESHOLD
    #: Smallest per-type quota in a subsample (``min.cell``).
    min_cell: int = MIN_CELL


UPSTREAM = ProgramSettings()


def ccim(lig_send: np.ndarray, rec_recv: np.ndarray) -> np.ndarray:
    """``GenerateCCIM`` (unweighted): the (n_send * n_recv, n_lr) matrix
    ``sqrt(L[s, l] * R[r, l])``. Row ``s + n_send * r`` is sender s with
    receiver r, the order upstream builds (sender varying fastest)."""
    prod = lig_send[None, :, :] * rec_recv[:, None, :]  # (recv, send, lr)
    return np.sqrt(prod.reshape(-1, lig_send.shape[1]))


def lr_similarity(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns of `m` with a positive sum, and their signed similarity
    ``0.5 + 0.5 * cor`` (upstream ``corSparse``, Pearson)."""
    keep = m.sum(axis=0) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        cor = np.corrcoef(m[:, keep], rowvar=False)
    return keep, 0.5 + 0.5 * np.atleast_2d(cor)


def stratified_subsample(
    cell_types: np.ndarray,
    rng: np.random.Generator,
    n_target: int = ITERATE_THRESHOLD,
    min_cell: int = MIN_CELL,
) -> np.ndarray:
    """Cell indices for one subsample, proportional to type: per type,
    ``max(round(n_target * n_type / N), min_cell)`` cells without replacement,
    capped at the type's size (upstream errors instead). ``round`` is R's
    (half to even), as NumPy's."""
    types, inverse = np.unique(cell_types, return_inverse=True)
    picks = []
    for t in range(len(types)):
        pool = np.nonzero(inverse == t)[0]
        quota = max(int(np.round(n_target * len(pool) / len(cell_types))), min_cell)
        picks.append(rng.choice(pool, size=min(quota, len(pool)), replace=False))
    return np.concatenate(picks)


def n_iterations(
    n_cells: int,
    n_target: int = ITERATE_THRESHOLD,
    n_iterate: int | None = None,
) -> int:
    """Subsamples upstream averages over: `n_iterate` if given, else
    ``round(sqrt(N / n_target))``; 0 (the full CCIM) when N <= `n_target`."""
    if n_cells <= n_target:
        return 0
    if n_iterate is not None:
        return n_iterate
    return int(np.round(np.sqrt(n_cells / n_target)))


def averaged_similarity(
    lig: np.ndarray,
    rec: np.ndarray,
    lr_names: Sequence[str],
    subsamples: Sequence[np.ndarray],
) -> pd.DataFrame:
    """The LR-pair similarity ``InteractionPrograms`` clusters: one per
    subsample (all ordered pairs of its cells, self-pairs included), averaged
    elementwise over the union of LR pairs, ignoring pairs absent from a
    subsample."""
    names = np.asarray(lr_names)
    mats = []
    for cells in subsamples:
        keep, sim = lr_similarity(ccim(lig[cells], rec[cells]))
        mats.append(pd.DataFrame(sim, index=names[keep], columns=names[keep]))
    union = pd.Index(pd.unique(np.concatenate([m.index.to_numpy() for m in mats])))
    stack = np.stack([m.reindex(index=union, columns=union).to_numpy() for m in mats])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0) if len(mats) > 1 else stack[0]
    return pd.DataFrame(mean, index=union, columns=union)


@dataclass
class SamplePrograms:
    """One sample's ``InteractionPrograms`` output."""

    similarity: pd.DataFrame
    soft_power: int
    modules: dict[str, list[str]]
    #: kTotal / kWithin per LR pair (``intramodularConnectivity``)
    connectivity: pd.DataFrame
    labels: np.ndarray = field(repr=False)


def choose_soft_power(sim: np.ndarray, settings: ProgramSettings = UPSTREAM) -> int:
    """``InteractionPrograms``' soft power: `settings.soft_power` if fixed,
    else ``pickSoftThreshold``'s estimate at `settings.r2_cutoff`, falling
    back to 1 when none passes or it exceeds `settings.max_soft_power`."""
    if settings.soft_power is not None:
        return settings.soft_power
    power, _ = _wgcna.pick_soft_threshold(sim, r2_cut=settings.r2_cutoff)
    cap = settings.max_soft_power
    return 1 if power is None or (cap is not None and power > cap) else power


def modules_from_similarity(
    similarity: pd.DataFrame, settings: ProgramSettings = UPSTREAM
) -> SamplePrograms:
    """``InteractionPrograms``' network steps on a given similarity matrix:
    soft power (`choose_soft_power`), signed TOM, complete-linkage tree,
    ``cutreeDynamic(method = "tree")`` at the `settings`' ``minClusterSize``
    and ``cutHeight`` quantile; the unassigned (grey) module is dropped and
    modules are named ``IP-1..`` by first appearance in LR order."""
    sim = similarity.to_numpy()
    soft_power = choose_soft_power(sim, settings)
    adj = sim**soft_power
    tree = _wgcna.hclust_complete(1.0 - _wgcna.signed_tom(adj))
    labels = _wgcna.cutree_dynamic_tree(
        tree,
        float(np.quantile(tree.height, settings.cut_height_quantile)),
        settings.min_module_size,
    )
    names = similarity.index.to_numpy()
    order = [lab for lab in pd.unique(labels) if lab != 0]
    modules = {f"IP-{i + 1}": list(names[labels == lab]) for i, lab in enumerate(order)}
    con = _wgcna.intramodular_connectivity(adj, labels)
    connectivity = pd.DataFrame(
        con, index=similarity.index, columns=["kTotal", "kWithin"]
    )
    return SamplePrograms(similarity, soft_power, modules, connectivity, labels)


def interaction_programs(
    lig: np.ndarray,
    rec: np.ndarray,
    lr_names: Sequence[str],
    cell_types: np.ndarray,
    rng: np.random.Generator,
    subsamples: Sequence[np.ndarray] | None = None,
    settings: ProgramSettings = UPSTREAM,
) -> SamplePrograms:
    """``InteractionPrograms`` for one sample's cells (`lig`/`rec` from
    `inputs.lr_expression`). `subsamples` overrides the stratified draws."""
    if subsamples is None:
        subsamples = draw_subsamples(cell_types, rng, settings)
    similarity = averaged_similarity(lig, rec, lr_names, subsamples)
    return modules_from_similarity(similarity, settings)


def draw_subsamples(
    cell_types: np.ndarray,
    rng: np.random.Generator,
    settings: ProgramSettings = UPSTREAM,
) -> list[np.ndarray]:
    """The cells of each CCIM ``InteractionPrograms`` averages: every cell once
    when the sample is small enough, else `n_iterations` stratified draws."""
    n_rep = n_iterations(
        len(cell_types), settings.iterate_threshold, settings.n_iterate
    )
    if n_rep == 0:
        return [np.arange(len(cell_types))]
    return [
        stratified_subsample(
            cell_types, rng, settings.iterate_threshold, settings.min_cell
        )
        for _ in range(n_rep)
    ]


def _jaccard_similarity(modules: Mapping[str, list[str]]) -> np.ndarray:
    """``1 - dist.binary(method = 1)^2`` between modules, computed the way
    ade4 does (``sqrt(1 - a / (a + b + c))``, then squared) so ties match."""
    universe = pd.unique(np.concatenate([np.asarray(v) for v in modules.values()]))
    member = np.array([np.isin(universe, v) for v in modules.values()], dtype=float)
    shared = member @ member.T
    sizes = member.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - shared
    return 1.0 - np.sqrt(1.0 - shared / union) ** 2


def _merge_most_similar(
    mods: dict[str, list[str]], sim_threshold: float, label: str
) -> dict[str, list[str]] | None:
    """One merge step of ``FindAllInteractionPrograms``: replace every module
    in a pair achieving the largest Jaccard similarity below 1 by their union
    (named `label`), or None when that similarity is <= `sim_threshold`."""
    d = _jaccard_similarity(mods)
    below = d[d < 1]
    if below.size == 0 or below.max() <= sim_threshold:
        return None
    # rownames(which(d == max, arr.ind = TRUE)): column-major order, repeats kept
    _, rows = np.nonzero((d == below.max()).T)
    names = list(mods)
    to_merge = [names[i] for i in rows]
    merged = list(pd.unique(np.concatenate([mods[n] for n in to_merge])))
    out = {k: v for k, v in mods.items() if k not in to_merge}
    out[label] = merged
    return out


def merge_programs(
    per_sample: Mapping[str, Mapping[str, list[str]]],
    sim_threshold: float = SIM_THRESHOLD,
) -> dict[str, list[str]]:
    """``FindAllInteractionPrograms``' merge: pool every sample's modules
    (named ``sample.IP-k``) and, while the largest Jaccard similarity below 1
    exceeds `sim_threshold`, replace every module in a pair achieving it by
    their union (``merge_k``). Merged modules are renamed ``IP-1..``."""
    mods: dict[str, list[str]] | None = {
        f"{s}.{name}": list(v) for s, m in per_sample.items() for name, v in m.items()
    }
    n_merge = 1
    while mods is not None and len(mods) > 1:
        step = _merge_most_similar(mods, sim_threshold, f"merge_{n_merge}")
        if step is None:
            break
        mods, n_merge = step, n_merge + 1
    assert mods is not None
    return {f"IP-{i + 1}": [str(x) for x in v] for i, v in enumerate(mods.values())}


def wilcox_greater_p(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``wilcox.test(a, b[i], alternative = "greater", exact = FALSE)$p.value``
    for every row ``b[i]`` (normal approximation with continuity and tie
    correction, as R). `a` and the rows of `b` must be finite."""
    a = np.sort(a)
    na, nb = len(a), b.shape[1]
    # W = sum over a of #(b < a) + #(b == a)/2 = sum over b of #(a > b) + #(a == b)/2
    below = np.searchsorted(a, b, side="left")
    upto = np.searchsorted(a, b, side="right")
    w = (na - upto).sum(axis=1) + 0.5 * (upto - below).sum(axis=1)
    combined = np.sort(
        np.concatenate([np.broadcast_to(a, (len(b), na)), b], axis=1), axis=1
    )
    width = combined.shape[1]
    starts = np.ones_like(combined, dtype=bool)
    starts[:, 1:] = combined[:, 1:] != combined[:, :-1]
    flat_starts = np.nonzero(starts.ravel())[0]
    runs = np.diff(np.append(flat_starts, combined.size)).astype(float)
    ties = np.bincount(flat_starts // width, weights=runs**3 - runs, minlength=len(b))
    n = na + nb
    z = w - na * nb / 2
    sigma = np.sqrt((na * nb / 12) * ((n + 1) - ties / (n * (n - 1))))
    return norm.sf((z - 0.5) / sigma)


def _finite_block(sim: pd.DataFrame, members: Sequence[str]) -> np.ndarray:
    """``as.vector(m_cor[rows in members, cols in members])``, finite values."""
    present = sim.index.isin(members)
    block = sim.to_numpy()[np.ix_(present, present)].ravel()
    return block[np.isfinite(block)]


def module_significance(
    sim: pd.DataFrame,
    members: Sequence[str],
    random_sets: np.ndarray,
) -> float:
    """``InteractionProgramSignificance`` for one module in one sample: the
    fraction of random LR sets (rows of `random_sets`, column indices into
    `sim`) for which the module's within-module similarity is not greater
    (one-sided Wilcoxon p > 0.05). 1 where upstream's ``try`` would fail
    (module absent from the sample, or larger than its LR pairs)."""
    a = _finite_block(sim, members)
    if a.size == 0 or len(members) > sim.shape[0]:
        return 1.0
    values = sim.to_numpy()
    b = np.stack([values[np.ix_(r, r)].ravel() for r in random_sets])
    if np.isfinite(b).all():
        p = wilcox_greater_p(a, b)
    else:
        p = np.array(
            [wilcox_greater_p(a, row[np.isfinite(row)][None, :])[0] for row in b]
        )
    return float(np.mean(p > 0.05))


def program_significance(
    similarities: Mapping[str, pd.DataFrame],
    modules: Mapping[str, list[str]],
    rng: np.random.Generator,
    n_replicate: int = N_REPLICATE,
) -> pd.DataFrame:
    """Module x sample table of `module_significance` values, with random LR
    sets of the module's size drawn per (sample, module)."""
    table = pd.DataFrame(index=list(modules), columns=list(similarities), dtype=float)
    for sample, sim in similarities.items():
        for name, members in modules.items():
            if len(members) > sim.shape[0]:
                table.loc[name, sample] = 1.0
                continue
            sets = np.stack(
                [
                    rng.choice(sim.shape[0], len(members), replace=False)
                    for _ in range(n_replicate)
                ]
            )
            table.loc[name, sample] = module_significance(sim, members, sets)
    return table


def kept_programs(
    modules: Mapping[str, list[str]],
    significance: pd.DataFrame,
    min_members: int = 1,
) -> dict[str, list[str]]:
    """Upstream's filters: significant (<= 0.05) in at least one sample, and
    more than `min_members` distinct ligands and receptors."""
    kept = {}
    for name, members in modules.items():
        if not (significance.loc[name] <= 0.05).any():
            continue
        ligands, receptors = program_sides(members)
        if len(ligands) > min_members and len(receptors) > min_members:
            kept[name] = list(members)
    return kept


def expression_bins(
    gene_means: pd.Series, rng: np.random.Generator, n_bin: int = N_BIN
) -> pd.Series:
    """``AddModuleScore``'s gene bins: means sorted increasingly, jittered by
    ``rnorm(n) / 1e30`` and cut into `n_bin` equal-count bins."""
    means = gene_means.sort_values(kind="stable")
    jittered = means.to_numpy() + rng.standard_normal(len(means)) / 1e30
    return pd.Series(_wgcna.cut_number(jittered, n_bin), index=means.index)


def control_genes(
    gene_means: pd.Series,
    features: Sequence[str],
    rng: np.random.Generator,
    n_bin: int = N_BIN,
    n_ctrl: int = N_CTRL,
) -> list[str]:
    """``AddModuleScore``'s control set for one feature list: genes binned by
    mean expression into `n_bin` equal-count bins (``cut_number`` of the means
    plus ``rnorm / 1e30``, which only separates exact ties such as all-zero
    genes), and for each feature `n_ctrl` genes drawn without replacement from
    its bin; the union is returned."""
    bins = expression_bins(gene_means, rng, n_bin)
    ctrl: list[str] = []
    for f in features:
        pool = bins.index[bins.to_numpy() == bins[f]]
        ctrl.extend(rng.choice(pool.to_numpy(), size=n_ctrl, replace=False))
    return list(pd.unique(np.asarray(ctrl)))


def _column_means(X: Any, cols: np.ndarray) -> np.ndarray:
    sub = X[:, cols]
    means = sub.mean(axis=1) if issparse(sub) else np.mean(sub, axis=1)
    return np.asarray(means).ravel()


def module_score(
    X: Any,
    gene_names: pd.Index,
    features: Sequence[str],
    controls: Sequence[str],
) -> np.ndarray:
    """``AddModuleScore`` for one feature list given its control genes: per
    cell, the mean of the features present in `gene_names` minus the mean of
    the controls."""
    pos = pd.Index(gene_names)
    feat = pos.get_indexer([f for f in features if f in pos])
    ctrl = pos.get_indexer(list(controls))
    return _column_means(X, feat) - _column_means(X, ctrl)


def program_sides(members: Sequence[str]) -> tuple[list[str], list[str]]:
    """A program's ligand set and receptor set (``ScoreInteractionPrograms``
    splits each ``ligand=receptor`` label)."""
    split = [m.split("=") for m in members]
    ligands = list(pd.unique(np.array([s[0] for s in split])))
    receptors = list(pd.unique(np.array([s[1] for s in split])))
    return ligands, receptors


def score_programs(
    X: Any,
    gene_names: pd.Index,
    programs: Mapping[str, list[str]],
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``ScoreInteractionPrograms``: (cells x programs) sender scores on each
    program's ligands and receiver scores on its receptors. As upstream, the
    ligand and receptor calls each reseed (``AddModuleScore(seed = 1)``)."""
    means = pd.Series(np.asarray(cast(Any, X).mean(axis=0)).ravel(), index=gene_names)
    scores = []
    for side in (0, 1):
        rng = np.random.default_rng(seed)
        cols = {}
        for name, members in programs.items():
            feats = [f for f in program_sides(members)[side] if f in means.index]
            controls = control_genes(means, feats, rng)
            cols[name] = module_score(X, gene_names, feats, controls)
        scores.append(pd.DataFrame(cols))
    return scores[0], scores[1]
