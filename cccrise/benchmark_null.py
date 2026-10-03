"""Label-permutation null for the ANOVA benchmark.

No decomposition in the benchmark sees the phenotype, so a null needs no refits:
the condition labels are shuffled and only the ANOVA is recomputed on each
method's fixed condition factor.

The statistic is the maximum eta-squared across a fit's components, per
association. Its permutation p-value is family-wise across components however
many a method has, so a rank-2 fit and a rank-10 fit are judged on the same
footing (Westfall-Young single-step maxT; `PermPAdj` is the per-component
version). Within an association, eta-squared, F and p are monotone in one
another, so the maximum eta-squared is also the minimum ANOVA p.

Permutations are drawn once per association from one seed and shared by every
method, so two methods' null draws are paired: `paired_difference` tests one
method's maximum eta-squared against another's by the difference each shuffle
produces, which carries whatever advantage a method's extra components give it
under the null. They permute only the
conditions an association keeps (its levels and non-missing labels), and move a
patient's conditions together: patients contributing the same number of
conditions are exchanged as blocks, and a block's order is shuffled within.
"""

import hashlib
from collections.abc import Sequence
from typing import Any

import anndata
import numpy as np
import pandas as pd

from .anova import adjust_pvalues, get_sample_to_group, prepare_group_labels

N_PERMUTATIONS = 10_000


def association_config(item: dict[str, Any] | str) -> dict[str, Any]:
    """Normalize one association spec the way `run_multiple_factors_anova` does."""
    if isinstance(item, str):
        return {"name": item, "group_column": item}
    return {
        "name": item.get("name", item["group_column"]),
        "group_column": item["group_column"],
        "levels": item.get("levels"),
        "mapping": item.get("mapping"),
        "control_label": item.get("control_label"),
        "other_label": item.get("other_label"),
    }


def association_labels(
    adata: anndata.AnnData, condition_column: str, association: dict[str, Any] | str
) -> pd.Series:
    """Processed group label per condition, in `np.unique` condition order --
    exactly the labels `run_multiple_factors_anova` tests."""
    config = association_config(association)
    return prepare_group_labels(
        get_sample_to_group(adata, condition_column, config["group_column"]),
        levels=config.get("levels"),
        mapping=config.get("mapping"),
        control_label=config.get("control_label"),
        other_label=config.get("other_label"),
    )


def block_permutations(
    patients: Sequence[Any] | np.ndarray, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Permutations of ``range(len(patients))`` that keep each patient's
    positions together.

    Returns ``(n_permutations, n)`` indices; row ``b`` is a relabelling
    ``labels[perm[b]]``. Blocks of the same size are exchanged with one another
    and each block's internal order is shuffled.
    """
    patients = np.asarray(patients)
    blocks = [np.flatnonzero(patients == p) for p in pd.unique(patients)]
    by_size: dict[int, list[np.ndarray]] = {}
    for block in blocks:
        by_size.setdefault(block.size, []).append(block)

    perms = np.empty((n_permutations, patients.size), dtype=np.int64)
    for b in range(n_permutations):
        for group in by_size.values():
            targets = np.concatenate(group)
            order = rng.permutation(len(group))
            sources = np.concatenate([rng.permutation(group[i]) for i in order])
            perms[b, targets] = sources
    return perms


def eta_squared(
    loadings: np.ndarray, codes: np.ndarray, perms: np.ndarray
) -> np.ndarray:
    """Eta-squared of every column of `loadings` under every permutation.

    Parameters
    ----------
    loadings : np.ndarray
        ``(n, n_components)``, the kept conditions only.
    codes : np.ndarray
        ``(n,)`` integer group codes.
    perms : np.ndarray
        ``(n_permutations, n)`` relabellings, from `block_permutations`.

    Returns
    -------
    np.ndarray
        ``(n_permutations, n_components)``.
    """
    onehot = np.eye(codes.max() + 1)[codes[perms]]  # (B, n, G)
    counts = onehot.sum(axis=1)  # (B, G), constant across B
    sums = np.einsum("bng,nk->bgk", onehot, loadings)
    n = loadings.shape[0]
    ss_between = (sums**2 / counts[:, :, None]).sum(axis=1) - loadings.sum(
        axis=0
    ) ** 2 / n
    ss_total = ((loadings - loadings.mean(axis=0)) ** 2).sum(axis=0)
    return ss_between / ss_total


def association_permutations(
    labels: pd.Series,
    patients: pd.Series,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The kept conditions, their group codes and their shared permutations.

    Returns ``(kept, codes, perms)``: `kept` indexes the condition factor's rows,
    `codes` are the observed group codes over `kept`, and `perms` relabels them.
    """
    kept = np.flatnonzero(labels.notna().to_numpy())
    kept_labels = labels.iloc[kept].astype(str).to_numpy()
    _, codes = np.unique(kept_labels, return_inverse=True)
    perms = block_permutations(patients.iloc[kept].to_numpy(), n_permutations, rng)
    return kept, codes, perms


def permutation_digest(perms: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(perms).tobytes()).hexdigest()[:16]


def maxt_null(
    condition_factor: np.ndarray, kept: np.ndarray, codes: np.ndarray, perms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed eta-squared per component, their maxT-adjusted permutation
    p-values, and the null distribution of the maximum across components.

    Components whose kept loadings are not all finite get NaN, as the ANOVA
    gives them; an association with a group of fewer than two conditions gets
    NaN throughout.
    """
    loadings = np.asarray(condition_factor, dtype=float)[kept]
    n_components = loadings.shape[1]
    counts = np.bincount(codes)
    if counts.size < 2 or counts.min() < 2:
        nan = np.full(n_components, np.nan)
        return nan, nan.copy(), np.full(perms.shape[0], np.nan)

    finite = np.all(np.isfinite(loadings), axis=0)
    observed = np.full(n_components, np.nan)
    identity = np.arange(len(codes))[None, :]
    observed[finite] = eta_squared(loadings[:, finite], codes, identity)[0]
    null_max = eta_squared(loadings[:, finite], codes, perms).max(axis=1)

    exceed = (null_max[None, :] >= observed[finite][:, None]).sum(axis=1)
    p_adj = np.full(n_components, np.nan)
    p_adj[finite] = (1 + exceed) / (1 + perms.shape[0])
    return observed, p_adj, null_max


def summarize_null(
    observed: np.ndarray, p_adj: np.ndarray, null_max: np.ndarray
) -> dict[str, float]:
    """One (fit, association) row: the best component against its null."""
    if not np.any(np.isfinite(observed)):
        return {
            "BestComponent": np.nan,
            "MaxEtaSquared": np.nan,
            "NullMean": np.nan,
            "NullSD": np.nan,
            "NullQ95": np.nan,
            "NullN": 0,
            "PermP": np.nan,
        }
    best = int(np.nanargmax(observed))
    return {
        "BestComponent": best + 1,
        "MaxEtaSquared": float(observed[best]),
        "NullMean": float(null_max.mean()),
        "NullSD": float(null_max.std(ddof=1)),
        "NullQ95": float(np.quantile(null_max, 0.95)),
        "NullN": int(null_max.size),
        "PermP": float(p_adj[best]),
    }


def paired_difference(
    observed_a: float, null_a: np.ndarray, observed_b: float, null_b: np.ndarray
) -> dict[str, float]:
    """Method A's maximum eta-squared against method B's, on shared permutations.

    The null of ``A - B`` is the per-shuffle difference of the two methods'
    maxima, so it is centred wherever the methods' component counts put it
    rather than at zero. `PairedP` is one-sided (A exceeds B by more than
    chance); `PairedPTwoSided` measures distance from the null's centre.
    """
    difference = observed_a - observed_b
    null = np.asarray(null_a) - np.asarray(null_b)
    if not (np.isfinite(difference) and np.all(np.isfinite(null))):
        return {
            "Difference": np.nan,
            "NullDiffMean": np.nan,
            "NullDiffQ025": np.nan,
            "NullDiffQ975": np.nan,
            "PairedP": np.nan,
            "PairedPTwoSided": np.nan,
        }
    centre = float(null.mean())
    n = null.size
    return {
        "Difference": float(difference),
        "NullDiffMean": centre,
        "NullDiffQ025": float(np.quantile(null, 0.025)),
        "NullDiffQ975": float(np.quantile(null, 0.975)),
        "PairedP": float((1 + np.sum(null >= difference)) / (1 + n)),
        "PairedPTwoSided": float(
            (1 + np.sum(np.abs(null - centre) >= abs(difference - centre))) / (1 + n)
        ),
    }


def bh_across_associations(
    summary: pd.DataFrame, by: Sequence[str], column: str = "PermP"
) -> pd.Series:
    """BH over associations within each group of `by`, applied to `column`."""
    return summary.groupby(list(by), sort=False)[column].transform(
        lambda p: adjust_pvalues(p.to_numpy(), method="fdr_bh")
    )
