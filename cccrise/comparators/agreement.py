"""Cross-method agreement of cccRISE with the tensor comparators.

Only the condition and ligand-receptor modes are shared between cccRISE (whose
sender/receiver modes index RISE eigen-states) and the cell-type pseudobulk
comparators, so whole-model agreement is measured there:

- `subspace_agreement`: linear CKA (Kornblith et al. 2019) and the mean
  principal-angle cosine between two fits' column spans, against a
  row-permutation null (one fit's rows shuffled, keeping its column geometry)
  and a Gaussian null of the same shapes.
- `containment_table`: per component, the fraction of a unit-normalized column
  of one fit that lies in the other fit's column span, against the
  row-permutation null; `shared_calls` turns that into "contained" calls.

`cell_type_fractions` describes how a cccRISE component's per-cell loadings
spread over cell types: the fraction of each type's cells that carry it, and
each type's share of its loading -- whether a component is carried by nearly
all cells of a type, as a cell-type mean would see it. Per-cell loadings are
first put on one scale across samples (`sample_scaled`): a sample's RISE
projection columns are unit-norm over its own cells, so raw per-cell loadings
shrink as ``1 / sqrt(n_cells)`` and a single threshold would favour cells of
small samples.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#: Containment at or above this, and above the null, calls a component contained.
CONTAINED = 0.85


def unit_columns(matrix: np.ndarray) -> np.ndarray:
    """Columns scaled to unit L2 norm (zero columns left at zero)."""
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column spans of `a` and
    `b`, descending."""
    qa, _ = np.linalg.qr(np.asarray(a, float))
    qb, _ = np.linalg.qr(np.asarray(b, float))
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between column-unit-normalized, column-centered `x` and `y`."""
    x, y = unit_columns(x), unit_columns(y)
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    den = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.linalg.norm(y.T @ x) ** 2 / den) if den > 0 else np.nan


def per_column_containment(span: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``||P_span c||^2`` for every unit-normalized column c of `columns`."""
    q, _ = np.linalg.qr(np.asarray(span, float))
    return np.sum((q.T @ unit_columns(columns)) ** 2, axis=0)


def joint_congruence(
    fa: dict[str, np.ndarray], fb: dict[str, np.ndarray], modes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Product over `modes` of |cosine| between every component pair, and its
    Hungarian one-to-one matching (rows, cols)."""
    congruence = np.ones((fa[modes[0]].shape[1], fb[modes[0]].shape[1]))
    for mode in modes:
        congruence *= np.abs(unit_columns(fa[mode]).T @ unit_columns(fb[mode]))
    rows, cols = linear_sum_assignment(-congruence)
    return congruence, rows, cols


# --- subspaces --------------------------------------------------------------------------


def _subspace_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {
        "linear_cka": linear_cka(a, b),
        "mean_principal_cosine": float(principal_cosines(a, b).mean()),
    }


def subspace_agreement(
    a: np.ndarray, b: np.ndarray, n_null: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Linear CKA and mean principal cosine of spans `a`, `b` (same rows), with
    row-permutation (`a`'s rows shuffled) and Gaussian nulls.

    One row per statistic: observed value, each null's mean and 95th
    percentile, and the null-normalized value ``(obs - m) / (1 - m)`` against
    the null with the larger 95th percentile.
    """
    observed = _subspace_stats(a, b)
    perm = pd.DataFrame(
        [_subspace_stats(a[rng.permutation(len(a))], b) for _ in range(n_null)]
    )
    gauss = pd.DataFrame(
        [
            _subspace_stats(rng.standard_normal(a.shape), rng.standard_normal(b.shape))
            for _ in range(n_null)
        ]
    )
    rows = []
    for stat, value in observed.items():
        q_perm, q_gauss = perm[stat].quantile(0.95), gauss[stat].quantile(0.95)
        ref = perm[stat] if q_perm >= q_gauss else gauss[stat]
        rows.append(
            {
                "statistic": stat,
                "observed": value,
                "perm_mean": perm[stat].mean(),
                "perm_q95": q_perm,
                "gauss_mean": gauss[stat].mean(),
                "gauss_q95": q_gauss,
                "above_null": bool(value > max(q_perm, q_gauss)),
                "normalized": (value - ref.mean()) / (1 - ref.mean()),
            }
        )
    return pd.DataFrame(rows)


# --- per-component containment --------------------------------------------------------------


def containment_table(
    span: np.ndarray, columns: np.ndarray, n_null: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Per column of `columns`: its containment in span(`span`) and the 95th
    percentile of that containment with `span`'s rows permuted."""
    observed = per_column_containment(span, columns)
    null = np.array(
        [
            per_column_containment(span[rng.permutation(len(span))], columns)
            for _ in range(n_null)
        ]
    )
    q95 = np.quantile(null, 0.95, axis=0)
    return pd.DataFrame(
        {
            "component": np.arange(1, columns.shape[1] + 1),
            "containment": observed,
            "null_q95": q95,
            "contained": (observed >= CONTAINED) & (observed > q95),
        }
    )


def shared_calls(containment: pd.DataFrame, modes: Sequence[str]) -> pd.DataFrame:
    """A component is contained in a fit when it is contained on every one of
    `modes`. `containment` has one row per (direction, fit, mode, component)
    with a boolean "contained" column."""
    sub = containment[containment["mode"].isin(modes)]
    keys = ["direction", "method", "rank_mode", "component"]
    calls = sub.groupby(keys)["contained"].agg(
        lambda c: bool(c.all()) and len(c) == len(modes)
    )
    return calls.rename("contained_all_modes").reset_index()


# --- cell-type fractions ---------------------------------------------------------------------


def cell_type_fractions(
    loadings: np.ndarray, cell_types: Sequence[str], tau_fracs: Sequence[float]
) -> pd.DataFrame:
    """Per (component, cell type): how many cells, the type's share of the
    component's total positive loading, and for each ``tau_frac`` the fraction
    of the type's cells whose loading exceeds ``tau_frac`` x the component's
    99th percentile across all cells."""
    loadings = np.asarray(loadings, float)
    types = np.asarray(cell_types).astype(str)
    positive = np.clip(loadings, 0, None)
    total = positive.sum(axis=0)
    p99 = np.percentile(loadings, 99, axis=0)
    rows = []
    for t in np.unique(types):
        cells = types == t
        for k in range(loadings.shape[1]):
            row = {
                "component": k + 1,
                "celltype": t,
                "n_cells": int(cells.sum()),
                "loading_share": float(positive[cells, k].sum() / total[k])
                if total[k] > 0
                else np.nan,
            }
            for tau in tau_fracs:
                row[f"active_frac_{tau:g}"] = float(
                    np.mean(loadings[cells, k] > tau * p99[k])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def cell_type_overlap(
    sender: np.ndarray,
    receiver: np.ndarray,
    cell_types: Sequence[str],
    tau_frac: float,
) -> pd.DataFrame:
    """Per (component, cell type): the fractions of the type's cells active as
    sender, as receiver, as both, and as either (sender, receiver or both),
    and the Jaccard overlap of the active sender and receiver cells (1: the
    same cells play both roles; 0: distinct cells). Active is as in
    `cell_type_fractions`, per side."""
    types = np.asarray(cell_types).astype(str)
    active = {}
    for side, loadings in (("sender", sender), ("receiver", receiver)):
        loadings = np.asarray(loadings, float)
        active[side] = loadings > tau_frac * np.percentile(loadings, 99, axis=0)
    both = active["sender"] & active["receiver"]
    either = active["sender"] | active["receiver"]
    rows = []
    for t in np.unique(types):
        cells = types == t
        for k in range(both.shape[1]):
            n_either = int(either[cells, k].sum())
            rows.append(
                {
                    "component": k + 1,
                    "celltype": t,
                    "n_cells": int(cells.sum()),
                    "sender_frac": float(active["sender"][cells, k].mean()),
                    "receiver_frac": float(active["receiver"][cells, k].mean()),
                    "both_frac": float(both[cells, k].mean()),
                    "either_frac": float(either[cells, k].mean()),
                    "jaccard": both[cells, k].sum() / n_either if n_either else np.nan,
                }
            )
    return pd.DataFrame(rows)


def sample_scaled(loadings: np.ndarray, samples: Sequence[str]) -> np.ndarray:
    """Per-cell loadings times ``sqrt(n_cells)`` of the cell's sample, which
    puts cells of differently sized samples on one scale."""
    samples = pd.Series(np.asarray(samples).astype(str))
    n = samples.map(samples.value_counts()).to_numpy(float)
    return np.asarray(loadings, float) * np.sqrt(n)[:, None]


def type_eta_squared(loadings: np.ndarray, cell_types: Sequence[str]) -> np.ndarray:
    """Per component: the fraction of per-cell loading variance explained by
    cell type (one-way ANOVA eta-squared)."""
    loadings = np.asarray(loadings, float)
    types = np.asarray(cell_types).astype(str)
    grand = loadings.mean(axis=0)
    between = sum(
        (types == t).sum() * (loadings[types == t].mean(axis=0) - grand) ** 2
        for t in np.unique(types)
    )
    total = ((loadings - grand) ** 2).sum(axis=0)
    return np.where(total > 0, between / np.where(total > 0, total, 1), np.nan)


def celltype_profiles(
    loadings: np.ndarray, cell_types: Sequence[str], order: Sequence[str]
) -> np.ndarray:
    """``(n_types, rank)`` mean |loading| per cell type, rows in `order`: a
    cccRISE sender/receiver mode expressed on the comparators' cell types."""
    loadings = np.abs(np.asarray(loadings, float))
    types = np.asarray(cell_types).astype(str)
    return np.vstack([loadings[types == str(t)].mean(axis=0) for t in order])
