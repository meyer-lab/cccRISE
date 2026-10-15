from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import tensorly as tl
from scipy.optimize import linear_sum_assignment
from tensorly.base import unfold
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import mode_dot  # ty: ignore[unresolved-import]

Factors = list[np.ndarray]


def calculate_r2x(weights: np.ndarray, factors: Factors, tensor: np.ndarray) -> float:
    """Calculate reconstruction R2X for a CP decomposition.

    Parameters
    ----------
    weights : np.ndarray
        Component weights from the CP decomposition.
    factors : list[np.ndarray]
        List of factor matrices from the CP decomposition.
    tensor : np.ndarray
        Original data tensor against which reconstruction is evaluated.

    Returns
    -------
    float
        Fraction of variance explained (R2X) by the CP decomposition.
    """
    reconstruction = cp_to_tensor((weights, factors))
    norm_tensor_sq = np.sum(tensor**2)
    if norm_tensor_sq == 0:
        return 0.0
    norm_residual_sq = np.sum((tensor - reconstruction) ** 2)
    return float(1.0 - norm_residual_sq / norm_tensor_sq)


def r2x_without_reconstruction(tensor, weights, factors) -> float:
    """R2X of a CP model against `tensor` on the active tensorly backend, from
    ``||T||^2 - 2 <T, T_hat> + ||T_hat||^2`` without forming the
    reconstruction (`calculate_r2x`'s value, to rounding), so it fits beside
    a tensor that fills the GPU or memory."""
    t_norm2 = sum(float(tl.sum(s * s)) for s in tensor)
    if t_norm2 == 0:
        return 0.0
    core = tl.to_numpy(tl.tenalg.multi_mode_dot(tensor, [f.T for f in factors]))
    w = tl.to_numpy(weights)
    diagonal = np.array([core[(r,) * 4] for r in range(len(w))])
    gram = np.ones((len(w), len(w)))
    for f in factors:
        f = tl.to_numpy(f)
        gram *= f.T @ f
    residual = t_norm2 - 2 * float(w @ diagonal) + float(w @ gram @ w)
    return float(1.0 - residual / t_norm2)


#: A pair whose cosines, multiplied over every mode, reach minus this is a
#: canceling pair: its two rank-1 terms are near-negatives of each other.
CANCELING_PRODUCT = 0.8

#: The share of each mode's energy a CANDELINC basis keeps when compression
#: must not change the fit (`candelinc_compress`'s `energy`). On ALAD and IBD
#: it reproduces the uncompressed CP fit from the same start (FMS >= 0.9998,
#: R2X within 3e-5); 99-99.5% does not at high rank.
CP_COMPRESSION_ENERGY = 0.999


def cpd_component_collinearity(
    factors_by_rank: dict[int, tuple[np.ndarray, Factors]],
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Screen CPD factor models across ranks for collinear and canceling pairs.

    A pair collinear (|cos| > `threshold`) in two or more modes is not
    identified: its two terms sum to the shared vectors times a rank-2
    matrix over the other modes, and any factorization of that matrix fits
    equally well. ``MaxModesCollinear`` counts the most such modes over pairs,
    whichever modes they are; collinearity in one mode (a shared sender,
    receiver or LR pair) leaves the pair identified.

    A pair is canceling (degenerate) when its cosines multiply over every
    mode to at most ``-CANCELING_PRODUCT``: the two rank-1 terms nearly cancel
    in the sum. Only this product is invariant to CP's per-mode sign
    convention -- one mode's sign can be moved to any other -- so a negative
    cosine in one mode alone is not a cancellation.

    Returns one row per rank: ``MaxModesCollinear``, ``WorstPairI`` /
    ``WorstPairJ`` (the pair with that count, 1-based, -1 when rank < 2),
    ``MinCosineProduct`` (over pairs) and ``WorstPairCanceling`` (whether any
    pair is canceling; NaN when rank < 2).
    """
    rows = []
    for rank in sorted(factors_by_rank):
        _weights, factors = factors_by_rank[rank]
        if rank < 2:
            rows.append(_no_collinearity_row(rank))
            continue

        best_count, best_pair, min_product = _worst_collinear_pair(factors, threshold)
        rows.append(
            {
                "Rank": rank,
                "MaxModesCollinear": best_count,
                "WorstPairCanceling": bool(min_product <= -CANCELING_PRODUCT),
                "WorstPairI": best_pair[0],
                "WorstPairJ": best_pair[1],
                "MinCosineProduct": min_product,
            }
        )

    return pd.DataFrame(rows)


def _no_collinearity_row(rank: int) -> dict:
    return {
        "Rank": rank,
        "MaxModesCollinear": 0,
        "WorstPairCanceling": np.nan,
        "WorstPairI": -1,
        "WorstPairJ": -1,
        "MinCosineProduct": np.nan,
    }


def _pair_cosines(factors: Factors) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per mode, the cosine of every component pair, ``(modes, pairs)``, and
    the pairs' 0-based indices ``i < j``."""
    rank = factors[0].shape[1]
    i, j = np.triu_indices(rank, k=1)
    cos = np.stack([_unit(f).T @ _unit(f) for f in factors])  # (modes, rank, rank)
    return cos[:, i, j], i, j


def _unit(f: np.ndarray) -> np.ndarray:
    return f / np.linalg.norm(f, axis=0, keepdims=True)


def _worst_collinear_pair(
    factors: Factors, threshold: float
) -> tuple[int, tuple[int, int], float]:
    """The pair collinear in the most modes (first on ties), and the smallest
    product of a pair's cosines over every mode."""
    cos, i, j = _pair_cosines(factors)
    counts = (np.abs(cos) > threshold).sum(axis=0)
    worst = int(np.argmax(counts))
    min_product = float(np.prod(cos, axis=0).min())
    return int(counts[worst]), (int(i[worst]) + 1, int(j[worst]) + 1), min_product


def cp_core_consistency(
    tensor: np.ndarray, weights: np.ndarray, factors: Factors
) -> float:
    """Core consistency diagnostic (CORCONDIA), generalized to 4 modes.

    Projects `tensor` onto each factor matrix's pseudoinverse to get the
    best-fit core in the CP factors' own basis, then compares that core to the
    super-diagonal core a valid quadrilinear model implies. Near 100 means the
    rank-R model is appropriate and essentially unique. It drops sharply once
    rank exceeds the number of components a valid CP model supports, because the extra
    components start trading off against one another and the projection picks
    up large off-diagonal entries.

    Returned in percent; can go negative for a badly-fitting rank.
    """
    rank = factors[0].shape[1]
    scaled_factors = [factors[0] * weights[np.newaxis, :], *factors[1:]]
    pinvs = [np.linalg.pinv(f) for f in scaled_factors]

    core = tensor
    for mode, pinv in enumerate(pinvs):
        core = mode_dot(core, pinv, mode)

    ideal = np.zeros((rank,) * 4)
    diag_idx = np.arange(rank)
    ideal[diag_idx, diag_idx, diag_idx, diag_idx] = 1.0

    return float(100.0 * (1.0 - np.sum((core - ideal) ** 2) / np.sum(ideal**2)))


def cp_core_consistency_sweep(
    tensor: np.ndarray, factors_by_rank: dict[int, tuple[np.ndarray, Factors]]
) -> pd.DataFrame:
    """Core consistency across every rank in a set of CP fits.

    Columns: 'Rank', 'CoreConsistency'.
    """
    rows = [
        {
            "Rank": rank,
            "CoreConsistency": cp_core_consistency(tensor, weights, factors),
        }
        for rank, (weights, factors) in sorted(factors_by_rank.items())
    ]
    return pd.DataFrame(rows)


def candelinc_compress(
    tensor: np.ndarray,
    max_rank: int,
    oversample: int = 5,
    energy: float | None = None,
) -> tuple[np.ndarray, Factors, float]:
    """Compress a tensor to a CANDELINC core sized for ranks up to `max_rank`.

    CANDELINC (Carroll, Pruzansky & Kruskal 1980) fits CP under the constraint
    that each factor matrix lies in the column space of a fixed basis. Taking
    those bases from a truncated HOSVD turns CP-ALS on the full tensor into
    CP-ALS on a core small enough to sweep many ranks over, then maps the
    solution back with ``A_n = U_n F_n``.

    Both rank-selection metrics are invariant to this, as long as each basis
    keeps at least `max_rank` columns. The bases are orthonormal, so
    within-mode cosines between decompressed components equal those between
    core components (`cpd_component_collinearity`), and
    ``pinv(U_n F_n) = pinv(F_n) U_n^T`` makes the projected core identical
    whether taken from the original tensor or the core
    (`cp_core_consistency`).

    The bases constrain the fit, though: a basis of ``max_rank +
    oversample`` columns discards enough of a sample or LR mode to change the
    optimum CP-ALS reaches. With `energy`, each basis also keeps the leading
    directions holding that fraction of the mode's energy (measured on
    `tensor`), which at 0.999 reproduces the uncompressed fit
    (`CP_COMPRESSION_ENERGY`).

    Raises
    ------
    ValueError
        If `max_rank` exceeds the smallest mode of `tensor`, which would leave
        the core unable to represent a rank-`max_rank` model.
    """
    if max_rank > min(tensor.shape):
        raise ValueError(
            f"max_rank={max_rank} exceeds the smallest mode of the tensor "
            f"(shape {tuple(tensor.shape)}). Compression cannot support a rank "
            f"the tensor itself cannot."
        )

    keep = [min(dim, max_rank + oversample) for dim in tensor.shape]
    if energy is not None:
        keep = [max(k, _energy_size(tensor, m, energy)) for m, k in enumerate(keep)]
    # Compress the most-compressible modes first: every later Gram matrix is
    # then formed from an already smaller tensor.
    order = sorted(range(tensor.ndim), key=lambda m: keep[m] / tensor.shape[m])

    norm_sq = float(np.sum(tensor**2))
    core = tensor
    bases: Factors = [np.empty(0)] * tensor.ndim
    for mode in order:
        if keep[mode] == core.shape[mode]:
            bases[mode] = np.eye(core.shape[mode])
            continue
        unfolded = unfold(core, mode)
        gram = unfolded @ unfolded.T
        # `eigh` returns ascending eigenvalues, so the leading singular
        # directions are the trailing columns.
        _eigvals, eigvecs = np.linalg.eigh(gram)
        bases[mode] = eigvecs[:, -keep[mode] :]
        core = mode_dot(core, bases[mode].T, mode)

    lost_sq = max(0.0, norm_sq - float(np.sum(core**2)))
    return core, bases, lost_sq


def _energy_size(tensor: np.ndarray, mode: int, energy: float) -> int:
    """How many leading directions of `mode` hold `energy` of its energy."""
    unfolded = unfold(tensor, mode)
    eigvals = np.linalg.eigvalsh(unfolded @ unfolded.T)[::-1]
    share = np.cumsum(eigvals) / eigvals.sum()
    return min(tensor.shape[mode], int(np.searchsorted(share, energy)) + 1)


def candelinc_r2x(core_r2x: float, core: np.ndarray, lost_sq: float) -> float:
    """Restate a core-space R2X as R2X against the original tensor.

    A CP model fit inside a CANDELINC core reports error against the core,
    which omits everything the projection discarded. Against the original
    tensor the error is ``lost_sq + ||core - CP(core)||^2``.
    """
    core_sq = float(np.sum(core**2))
    norm_sq = core_sq + lost_sq
    if norm_sq == 0:
        return 0.0
    return float(1.0 - (lost_sq + (1.0 - core_r2x) * core_sq) / norm_sq)


def kneedle_elbow(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, S: float = 1.0
) -> int:
    """Elbow of a convex, decreasing curve by offline Kneedle.

    Port of what Tensor-cell2cell calls (`kneed.KneeLocator(curve="convex",
    direction="decreasing", S=1.0).elbow`) without the interpolation step,
    which is a no-op on integer ranks. Returns the first local maximum of the
    difference curve that the curve then drops below its threshold from, or
    the difference curve's argmax if none does.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < 3:
        return int(xs[np.argmin(ys)])

    def _normalize(v: np.ndarray) -> np.ndarray:
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 0 else np.zeros_like(v)

    x_norm = _normalize(xs)
    y_norm = _normalize(ys)
    difference = (y_norm.max() - y_norm) - x_norm

    maxima = [
        i
        for i in range(1, xs.size - 1)
        if difference[i] >= difference[i - 1] and difference[i] >= difference[i + 1]
    ]
    step = S * float(np.abs(np.diff(x_norm)).mean())
    for i in maxima:
        threshold = difference[i] - step
        later_maxima = [j for j in maxima if j > i]
        stop = later_maxima[0] if later_maxima else xs.size
        if np.any(difference[i + 1 : stop] < threshold):
            return int(xs[i])
    return int(xs[int(np.argmax(difference))])


# --- The multi-start rank rule (RANK_SELECTION_RULE.md) --------------------------------
#
# Per rank, CP is fit from the production starts. Each fit is screened for
# canceling, non-identified and borderline pairs; the rank is kept when
# nearly every start passes, the passing starts' core consistency is above
# the published invalid boundary, and the equally good passing starts agree.

#: Production starts per rank (`ccc_rise.cp_start_seeds`).
RULE_N_STARTS = 10
#: |cos| above this in two or more modes leaves a pair not identified.
NOT_IDENTIFIED_COS = 0.9
#: |cos| above this in two or more modes is borderline: the pair passes only
#: if both its components reproduce in the best passing start.
BORDERLINE_COS = 0.8
#: Share of starts that must pass the screen.
RULE_PASS_SHARE = 0.9
#: Passing starts within this R2X of the best are equally good fits.
RULE_R2X_WINDOW = 0.005
#: Matched congruence (FMS) at which two fits, or a component, reproduce.
RULE_AGREE_FMS = 0.9

#: One start's fit: weights, factors and R2X.
Start = tuple[np.ndarray, Factors, float]


def screen_pairs(factors: Factors) -> tuple[bool, bool, list[tuple[int, int]]]:
    """Whether a fit has a canceling pair, a non-identified pair, and its
    borderline pairs (0-based), every mode treated alike."""
    if factors[0].shape[1] < 2:
        return False, False, []
    cos, i, j = _pair_cosines(factors)
    canceling = bool((np.prod(cos, axis=0) <= -CANCELING_PRODUCT).any())
    n_strict = (np.abs(cos) > NOT_IDENTIFIED_COS).sum(axis=0)
    n_loose = (np.abs(cos) > BORDERLINE_COS).sum(axis=0)
    borderline = [
        (int(a), int(b))
        for a, b, strict, loose in zip(i, j, n_strict, n_loose, strict=True)
        if strict < 2 <= loose
    ]
    return canceling, bool((n_strict >= 2).any()), borderline


def matched_congruence(reference: Factors, factors: Factors) -> np.ndarray:
    """For each component of `factors`, the absolute product over modes of its
    cosines with its optimal (Hungarian) match in `reference`. The mean is the
    two fits' FMS without weights, as `validation.stability.per_component_fms`."""
    congruence = np.ones((reference[0].shape[1], factors[0].shape[1]))
    for ref, f in zip(reference, factors, strict=True):
        congruence = congruence * (_unit(ref).T @ _unit(f))
    congruence = np.abs(congruence)
    rows, cols = linear_sum_assignment(-congruence)
    out = np.zeros(factors[0].shape[1])
    out[cols] = congruence[rows, cols]
    return out


def screen_starts(starts: Sequence[Start]) -> pd.DataFrame:
    """Per start: R2X, the screen, its FMS to the best start clear of canceling
    and non-identified pairs, and whether it passes (borderline pairs
    included, by reproducing in that best start)."""
    rows = [
        dict(
            zip(
                ("Canceling", "NotIdentified", "Borderline"),
                screen_pairs(f),
                strict=True,
            )
        )
        | {"R2X": r2x}
        for _w, f, r2x in starts
    ]
    table = pd.DataFrame(rows)
    clear = ~table["Canceling"] & ~table["NotIdentified"]
    table["FMS"], table["Passes"] = np.nan, False
    if not clear.any():
        return table
    reference = starts[int(table.loc[clear, "R2X"].idxmax())][1]
    for k, (_w, f, _r2x) in enumerate(starts):
        per = matched_congruence(reference, f)
        table.loc[k, "FMS"] = per.mean()
        reproduced = all(
            min(per[a], per[b]) >= RULE_AGREE_FMS for a, b in table.at[k, "Borderline"]
        )
        table.loc[k, "Passes"] = bool(clear[k]) and reproduced
    return table


def rank_rule_row(tensor: np.ndarray, starts: Sequence[Start]) -> tuple[dict, int]:
    """One rank under the rule: the share of starts passing, the passing
    starts' median core consistency, the median FMS to the best passing start
    of the other passing starts within `RULE_R2X_WINDOW` of it, whether the
    rank passes, and the index of its production fit (the best passing start,
    or the best start when none passes)."""
    table = screen_starts(starts)
    passing = table[table["Passes"]]
    row = {"PassShare": float(table["Passes"].mean()), "CoreConsistency": np.nan,
           "AgreementFMS": np.nan, "NAgreeing": 0, "BestPassingR2X": np.nan}  # fmt: skip
    if passing.empty:
        return row | {"Passes": False}, int(table["R2X"].idxmax())
    best = int(passing["R2X"].idxmax())
    near = passing[
        (passing["R2X"] >= passing["R2X"].max() - RULE_R2X_WINDOW)
        & (passing.index != best)
    ]
    row |= {
        "CoreConsistency": float(
            np.median(
                [cp_core_consistency(tensor, *starts[k][:2]) for k in passing.index]
            )
        ),
        "AgreementFMS": float(near["FMS"].median()) if len(near) else np.nan,
        "NAgreeing": len(near),
        "BestPassingR2X": float(passing["R2X"].max()),
    }
    row["Passes"] = bool(
        row["PassShare"] >= RULE_PASS_SHARE and row["CoreConsistency"] > 0
        and row["AgreementFMS"] >= RULE_AGREE_FMS
    )  # fmt: skip
    return row, best


def select_rank_by_rule(
    tensor: np.ndarray,
    starts_by_rank: Iterable[tuple[int, Sequence[Start]]],
    stop_at_failure: bool = False,
) -> tuple[int, pd.DataFrame, tuple[np.ndarray, Factors]]:
    """The largest rank passing the rule, the per-rank table and that rank's
    production fit (weights, factors). Ranks are consumed in order; with
    `stop_at_failure` the sweep ends at the first failing rank, which selects
    the same rank whenever the passing ranks are contiguous from 1. With no
    passing rank, the first rank is returned."""
    rows, chosen = [], {}
    for rank, starts in starts_by_rank:
        row, k = rank_rule_row(tensor, starts)
        rows.append({"Rank": rank} | row)
        chosen[rank] = starts[k][:2]
        if stop_at_failure and not row["Passes"]:
            break
    table = pd.DataFrame(rows)
    passing = table.loc[table["Passes"], "Rank"]
    rank = int(passing.max()) if len(passing) else int(table["Rank"].iloc[0])
    return rank, table, chosen[rank]
