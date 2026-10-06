"""Clinical association of cccRISE components within single cell types.

cccRISE's condition factor summarises each component over all of a sample's
cells. A signal confined to one cell type -- especially a rare one -- can be
diluted out of it. Here each *unit* is one component, one side (sender
`sc_B` / receiver `rc_C` per-cell loadings, sample-scaled) and one cell type:
its per-sample mean loading over that type's cells. A sample enters a unit
when it has at least `min_cells` cells of the type, and a unit is tested when
at least `min_samples` samples enter and every group keeps two.

Each unit is tested against an association by one-way ANOVA eta-squared.
The null permutes the association's labels over all its conditions with the
benchmark's patient-block permutations (`benchmark_null`) and then restricts
to the unit's samples, so every unit sees the same shuffles; the maxT
adjustment takes the maximum eta-squared across all tested units per shuffle.
"""

import numpy as np
import pandas as pd

MIN_CELLS = 3
MIN_SAMPLES = 20


def per_sample_means(
    loadings: np.ndarray,
    samples: np.ndarray,
    types: np.ndarray,
    conditions: np.ndarray,
) -> pd.DataFrame:
    """Long table (type, condition, component, mean, n_cells) of each
    type's per-sample mean loading, conditions in `conditions` order."""
    k = loadings.shape[1]
    frame = pd.DataFrame(loadings, columns=np.arange(1, k + 1))
    frame["type"], frame["condition"] = types, samples
    grouped = frame.groupby(["type", "condition"])
    means = grouped.mean().stack().rename("mean").reset_index()
    means = means.rename(columns={"level_2": "component"})
    sizes = grouped.size().rename("n_cells").reset_index()
    out = means.merge(sizes, on=["type", "condition"])
    return out[out["condition"].isin(conditions)]


def eta_squared_subset(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Eta-squared of `x` under each row of `codes` (``(B, n)`` group codes);
    group sizes may differ between rows, empty groups contribute nothing."""
    onehot = np.eye(codes.max() + 1)[codes]  # (B, n, G)
    counts = onehot.sum(axis=1)
    sums = np.einsum("bng,n->bg", onehot, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        between = np.where(counts > 0, sums**2 / counts, 0.0).sum(axis=1)
    between -= x.sum() ** 2 / len(x)
    total = ((x - x.mean()) ** 2).sum()
    return between / total if total > 0 else np.full(len(codes), np.nan)


def _testable(
    values: np.ndarray, codes: np.ndarray, min_samples: int
) -> np.ndarray | None:
    """The unit's available positions, or None if it cannot be tested."""
    avail = np.isfinite(values)
    counts = np.bincount(codes[avail], minlength=codes.max() + 1)
    if avail.sum() < min_samples or counts.size < 2 or (counts < 2).any():
        return None
    return avail


def unit_maxt(
    units: dict[tuple, np.ndarray],
    kept: np.ndarray,
    codes: np.ndarray,
    perms: np.ndarray,
    min_samples: int = MIN_SAMPLES,
) -> pd.DataFrame:
    """Per testable unit: n samples, observed eta-squared, its own permutation
    p and its maxT-adjusted p across all testable units.

    `units` maps a key tuple to per-condition values (NaN where the sample did
    not enter); `kept`, `codes`, `perms` are one association's plan
    (`benchmark_null.association_permutations`)."""
    permuted = codes[perms]  # (B, n_kept)
    keys, observed, nulls, sizes = [], [], [], []
    for key, values in units.items():
        x = np.asarray(values, float)[kept]
        avail = _testable(x, codes, min_samples)
        if avail is None:
            continue
        keys.append(key)
        sizes.append(int(avail.sum()))
        observed.append(eta_squared_subset(x[avail], codes[avail][None, :])[0])
        nulls.append(eta_squared_subset(x[avail], permuted[:, avail]))
    if not keys:
        return pd.DataFrame()
    null = np.column_stack(nulls)  # (B, units)
    obs = np.asarray(observed)
    null_max = np.nanmax(null, axis=1)
    n_perm = perms.shape[0]
    return pd.DataFrame(
        {
            "key": keys,
            "n_samples": sizes,
            "eta_squared": obs,
            "perm_p": (1 + (null >= obs[None, :]).sum(axis=0)) / (1 + n_perm),
            "maxt_p": (1 + (null_max[:, None] >= obs[None, :]).sum(axis=0))
            / (1 + n_perm),
            "n_units": len(keys),
        }
    )
