"""How RISE eigen-states and cccRISE components spread across fine cell types.

Every cell has a score on each RISE eigen-state (its projection) and on each
cccRISE component (`sc_B` / `rc_C`). A cell-type pseudobulk method instead
gives every cell of a type one value, so a program it can represent is all or
none of each (fine) type. Per score this module asks which cells are active
(score above `TAU` x the score's 99th percentile, as in the cell-fraction
panel) and how they sit across fine types:

- **fine-aligned**: active cells come from fine types that are almost wholly
  active (>= `ALIGNED`), so a finer annotation would capture the program;
- **cross-cutting**: active cells come mostly from fine types that are only
  partly active (between `PARTIAL` and `ALIGNED`), in at least two fine types,
  so no annotation at either granularity represents it;
- otherwise **partial in one fine type** or **mixed**.

A fraction is only a sub-population if it does not depend on where the cut
falls: `fraction_checks` reports, for the fine type holding most active
cells, its active fraction at `SENSITIVITY_TAUS` and the bimodality
coefficient of its scores. A fraction that moves with the cut through a
unimodal distribution is a graded program, not a subset of cells.

Scores are sample-scaled (`comparators.agreement.sample_scaled`) and oriented
so that their largest-magnitude values are positive (`orient`). A
random-direction null (`random_direction_null`) gives the same summaries for
directions in RISE space unrelated to any program.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

TAU = 0.5
ALIGNED = 0.8
PARTIAL = 0.2
#: A fine type counts towards "cross-cutting" if it holds this share of the active cells.
MIN_SHARE = 0.05
SENSITIVITY_TAUS = (0.25, 0.75)
#: Bimodality coefficient of a uniform distribution; larger suggests two modes.
BIMODAL = 5 / 9


def orient(scores: np.ndarray) -> np.ndarray:
    """Flip each column so its top 1% of |values| has a positive mean."""
    scores = np.asarray(scores, float)
    cut = np.percentile(np.abs(scores), 99, axis=0)
    big = np.abs(scores) >= cut
    signs = np.array(
        [np.sign(scores[big[:, k], k].mean()) or 1.0 for k in range(scores.shape[1])]
    )
    return scores * signs


def active(score: np.ndarray, tau: float = TAU) -> np.ndarray:
    return score > tau * np.percentile(score, 99)


def fine_profile(
    is_active: np.ndarray, fine: np.ndarray, coarse: np.ndarray
) -> pd.DataFrame:
    """Per fine type: its coarse type, cell count, active fraction and share
    of all active cells."""
    frame = pd.DataFrame({"fine": fine, "coarse": coarse, "active": is_active})
    out = frame.groupby(["coarse", "fine"])["active"].agg(["size", "mean", "sum"])
    out = out.rename(
        columns={"size": "n_cells", "mean": "active_frac", "sum": "n_active"}
    )
    total = max(int(is_active.sum()), 1)
    out["active_share"] = out["n_active"] / total
    return out.reset_index()


def _call(aligned_share: float, partial_share: float, n_partial: int) -> str:
    if aligned_share >= ALIGNED:
        return "fine-aligned"
    if partial_share >= 0.5:
        return "cross-cutting" if n_partial >= 2 else "partial in one fine type"
    return "mixed"


def classify(profile: pd.DataFrame) -> dict:
    """The fine-type summary of one score's active cells (see module doc)."""
    frac, share = profile["active_frac"], profile["active_share"]
    aligned_share = float(share[frac >= ALIGNED].sum())
    partial = (frac >= PARTIAL) & (frac < ALIGNED)
    partial_share = float(share[partial].sum())
    n_partial = int((partial & (share >= MIN_SHARE)).sum())
    top = profile.loc[share.idxmax()]
    return {
        "n_active": int(profile["n_active"].sum()),
        "aligned_share": aligned_share,
        "partial_share": partial_share,
        "n_partial_fine": n_partial,
        "top_fine": str(top["fine"]),
        "top_fine_active_frac": float(top["active_frac"]),
        "top_fine_share": float(top["active_share"]),
        "call": _call(aligned_share, partial_share, n_partial),
    }


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient (sample skewness and excess kurtosis)."""
    n = len(x)
    g, k = stats.skew(x), stats.kurtosis(x)
    return float((g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def fraction_checks(score: np.ndarray, fine: np.ndarray, top_fine: str) -> dict:
    """The top fine type's active fraction at each `SENSITIVITY_TAUS` and the
    bimodality coefficient of its scores."""
    cut = np.percentile(score, 99)
    x = score[fine == top_fine]
    return {
        **{
            f"top_fine_active_frac_tau{t:g}": float(np.mean(x > t * cut))
            for t in SENSITIVITY_TAUS
        },
        "top_fine_bimodality": bimodality_coefficient(x),
    }


def home_types(profile: pd.DataFrame, min_cells: int = 50) -> list[str]:
    """Coarse types holding the score's active cells: those at least
    `PARTIAL` active (>= `min_cells` cells), else the most active one."""
    by_type = profile.groupby("coarse")[["n_cells", "n_active"]].sum()
    by_type = by_type[by_type["n_cells"] >= min_cells]
    frac = by_type["n_active"] / by_type["n_cells"]
    home = list(frac[frac >= PARTIAL].index)
    return home or [str(frac.idxmax())]


def histogram_table(
    score: np.ndarray, fine: np.ndarray, threshold: float, n_bins: int = 60
) -> pd.DataFrame:
    """Cell counts per (bin, fine type) of `score`, with the active cut."""
    fine = np.asarray(fine).astype(str)
    edges = np.linspace(score.min(), score.max(), n_bins + 1)
    rows = []
    for f in np.unique(fine):
        counts, _ = np.histogram(score[fine == f], bins=edges)
        rows += [
            {"bin_left": lo, "bin_right": hi, "fine": f, "count": int(c)}
            for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True)
        ]
    return pd.DataFrame(rows).assign(threshold=threshold)


def random_direction_null(
    basis: np.ndarray,
    fine: np.ndarray,
    coarse: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """`classify` for `n_draws` random unit directions in the span of `basis`
    (cells x states, e.g. sample-scaled RISE projections), oriented."""
    v = rng.normal(size=(basis.shape[1], n_draws))
    scores = orient(basis @ (v / np.linalg.norm(v, axis=0)))
    return pd.DataFrame(
        [
            classify(fine_profile(active(scores[:, j]), fine, coarse))
            for j in range(n_draws)
        ]
    )


def score_units(
    scores: dict[str, np.ndarray],
    coarse: Sequence[str],
    fine: Sequence[str],
    n_bins: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per named score: its `classify` summary, its per-fine-type profile,
    and its histogram over the cells of its `home_types`."""
    coarse = np.asarray(coarse).astype(str)
    fine = np.asarray(fine).astype(str)
    summaries, profiles, hists = [], [], []
    for unit, score in scores.items():
        profile = fine_profile(active(score), fine, coarse)
        home = home_types(profile)
        cells = np.isin(coarse, home)
        threshold = TAU * float(np.percentile(score, 99))
        summary = classify(profile)
        summaries.append(
            {
                "unit": unit,
                "home_types": ", ".join(home),
                **summary,
                **fraction_checks(score, fine, summary["top_fine"]),
            }
        )
        profiles.append(profile.assign(unit=unit))
        hists.append(
            histogram_table(score[cells], fine[cells], threshold, n_bins).assign(
                unit=unit
            )
        )
    return (
        pd.DataFrame(summaries),
        pd.concat(profiles, ignore_index=True),
        pd.concat(hists, ignore_index=True),
    )
