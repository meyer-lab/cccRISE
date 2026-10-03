"""NicheNet downstream-target validation of the tensor comparators, and a
statistic comparing methods.

`nichenet.component_ligand_scores` scores every ligand on a component's
receiver ranking: the cells with the highest vs lowest receiver loading,
contrasted by differential expression. cccRISE gives every cell its own
receiver loading (`obsm["rc_C"]`). A cell-type pseudobulk comparator (pseudobulk
CPD, Tensor-cell2cell) has one receiver loading per cell type, so every cell is
given its type's row of the comparator's receiver factor
(`broadcast_receiver_loadings`). Within a type the method cannot tell cells
apart, so ties at the high/low cut are broken at random: the groups stay the
same size as cccRISE's (`top_perc` percent of cells each), drawn uniformly
within the boundary type. STACCato is Tucker, with no per-component receiver
loading, and is not scored.

Methods are compared on one scale (`standardized_top_scores`): each top
ligand's percentile among its own component's other scored ligands, so 0.5
means a component's top ligands are no better supported by downstream targets
than the ligands it barely uses. `compare_methods` tests a reference method
against each comparator on those percentiles; `component_permutation_test`
compares the methods' per-component means.
"""

from collections.abc import Sequence

import anndata
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ..anova import adjust_pvalues
from .nichenet import component_ligand_scores


def orient_receivers(receiver: np.ndarray) -> np.ndarray:
    """Flip receiver-factor columns to a positive mean. CP leaves each column's
    sign arbitrary (paired with a flip in another mode), and the NicheNet
    "high" group is the high end of the receiver loading."""
    receiver = np.asarray(receiver, float)
    return receiver * np.where(receiver.mean(axis=0) < 0, -1.0, 1.0)


def broadcast_receiver_loadings(
    cell_types: Sequence[str],
    fit_celltypes: Sequence[str],
    receiver: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(n_cells, rank)`` receiver loadings: each cell gets its type's row of
    `receiver`, plus a uniform jitter smaller than the smallest gap between two
    distinct type values in that column, so cells only reorder within a type."""
    lookup = {str(t): i for i, t in enumerate(fit_celltypes)}
    missing = sorted(set(map(str, cell_types)) - set(lookup))
    if missing:
        raise KeyError(f"cell types not in the comparator's receiver mode: {missing}")
    codes = np.array([lookup[str(t)] for t in cell_types])
    base = np.asarray(receiver, float)[codes]
    out = np.empty_like(base)
    for k in range(base.shape[1]):
        values = np.unique(np.asarray(receiver, float)[:, k])
        gaps = np.diff(values)
        scale = gaps[gaps > 0].min() * 1e-3 if (gaps > 0).any() else 1e-9
        out[:, k] = base[:, k] + rng.uniform(0.0, scale, len(base))
    return out


def comparator_ligand_scores(
    adata: anndata.AnnData,
    celltype_column: str,
    fit_celltypes: Sequence[str],
    receiver: np.ndarray,
    lr_factor: np.ndarray,
    lr_pairs: pd.DataFrame,
    lt_df: pd.DataFrame,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """`nichenet.component_ligand_scores` for one comparator fit, with
    broadcast receivers (`broadcast_receiver_loadings`) and the comparator's
    LR factor. `lr_pairs` ("ligand"/"receptor") follows `lr_factor`'s rows;
    `kwargs` pass through. `adata`'s `obsm["rc_C"]` / `uns["D"]` are
    restored afterwards."""
    rng = np.random.default_rng(random_state)
    loadings = broadcast_receiver_loadings(
        adata.obs[celltype_column].astype(str).to_numpy(),
        fit_celltypes,
        orient_receivers(receiver),
        rng,
    )
    saved = (adata.obsm.get("rc_C"), adata.uns.get("D"))
    adata.obsm["rc_C"] = loadings
    adata.uns["D"] = np.asarray(lr_factor, float)
    try:
        components = list(range(1, loadings.shape[1] + 1))
        return component_ligand_scores(adata, lr_pairs, lt_df, components, **kwargs)
    finally:
        for key, store, value in (
            ("rc_C", adata.obsm, saved[0]),
            ("D", adata.uns, saved[1]),
        ):
            if value is None:
                store.pop(key, None)
            else:
                store[key] = value


# --- comparing methods -----------------------------------------------------------------

KEYS = ["method", "rank_mode", "component"]


def standardized_top_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Each top ligand's percentile among its component's non-top scored
    ligands (the fraction it beats, ties counting one half).

    `scores` stacks `component_ligand_scores` outputs with "method" and
    "rank_mode" columns. Returns one row per (method, rank mode, component,
    top ligand) with its AUROC and "percentile".
    """
    rows = []
    for key, c in scores.dropna(subset=["auroc"]).groupby(KEYS, sort=False):
        rest = c.loc[~c["is_top"], "auroc"].to_numpy()
        if not len(rest):
            continue
        for ligand, auroc in c.loc[c["is_top"], ["ligand", "auroc"]].itertuples(
            index=False
        ):
            pct = (np.sum(rest < auroc) + 0.5 * np.sum(rest == auroc)) / len(rest)
            rows.append(
                {
                    **dict(zip(KEYS, key, strict=True)),
                    "ligand": ligand,
                    "auroc": auroc,
                    "percentile": pct,
                }
            )
    return pd.DataFrame(rows)


def compare_methods(
    standardized: pd.DataFrame, reference: str = "cccRISE"
) -> pd.DataFrame:
    """One-sided Mann-Whitney U of the reference's pooled top-ligand
    percentiles against each other (method, rank mode)'s (H1: the reference's
    are larger), BH-adjusted across comparators.

    Returns one row per comparator fit: sample sizes, medians, the common-
    language effect size ``U / (n_ref n_cmp)``, p and q.
    """
    ref = standardized.loc[standardized["method"] == reference, "percentile"]
    rows = []
    others = standardized[standardized["method"] != reference]
    for (method, mode), c in others.groupby(["method", "rank_mode"], sort=False):
        u, p = mannwhitneyu(ref, c["percentile"], alternative="greater")
        rows.append(
            {
                "reference": reference,
                "method": method,
                "rank_mode": mode,
                "n_reference": len(ref),
                "n_comparator": len(c),
                "median_reference": float(np.median(ref)),
                "median_comparator": float(c["percentile"].median()),
                "cles": float(u / (len(ref) * len(c))),
                "p": float(p),
            }
        )
    out = pd.DataFrame(rows)
    out["q"] = adjust_pvalues(out["p"].to_numpy(), method="fdr_bh")
    return out


def component_means(standardized: pd.DataFrame) -> pd.DataFrame:
    """Per (method, rank mode, component): the mean top-ligand percentile, the
    AUROC form of that component's own top-vs-rest comparison."""
    return (
        standardized.groupby(KEYS, sort=False)["percentile"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "mean_percentile", "size": "n_top"})
        .reset_index()
    )


def component_permutation_test(
    a: np.ndarray, b: np.ndarray, n_perm: int = 10_000, random_state: int = 0
) -> dict[str, float]:
    """Two-sided permutation test of ``mean(a) - mean(b)`` over component-level
    values, relabelling components between the two methods."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    observed = a.mean() - b.mean()
    rng = np.random.default_rng(random_state)
    perms = np.argsort(rng.random((n_perm, len(pooled))), axis=1)
    shuffled = pooled[perms]
    null = shuffled[:, : len(a)].mean(axis=1) - shuffled[:, len(a) :].mean(axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + n_perm)
    return {"difference": float(observed), "p": float(p)}
