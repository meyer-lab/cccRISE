"""Figure 4's derived inputs: the scores tables of both labellings, each fit's
LR columns, and what each LR column is (`CLASSES`)."""

from functools import cache

import numpy as np
import pandas as pd
import xarray as xr

from ...comparators import scores
from ...factors import load_factors_netcdf
from ...runners.datasets import (
    COMPARATOR_FITS_DIR,
    COMPARATOR_SCORES_DIR,
    DATASETS,
    comparator_filename,
    scores_filename,
)
from .. import common, style
from ._common import COMPARATORS, DATASET, HINT, load

LABELLINGS = ("coarse", "fine")
KEYS = {"PB": "pb", "TC2C": "tc2c", "STAC": "stac"}
#: Shorthand for a method's components: C (cccRISE), P, T, S.
SHORT = {"cccRISE": "C", "PB": "P", "TC2C": "T", "STAC": "S"}
#: What an LR column is, in stacking order, and its color. A column is
#: associated with an outcome (FDR < `common.SIG_ALPHA`) first of all, then
#: contained in cccRISE's LR span (comparators only), then homophilic
#: (homophilic pairs carry at least `HOMOPHILIC` of its squared loading),
#: else other (`_column_class`).
CLASSES = {
    "cccRISE contained": style.COLORS["emphasis"],
    "homophilic": "0.55",
    "associated": style.COLORS["up"],
    "other": "0.78",
}
HOMOPHILIC = 0.8
#: A column is single-pair, whatever its class, when its top pair carries at
#: least this share of its squared loading.
SINGLE_PAIR = 0.8
#: Below this |cosine| a column carries none of a reference's content, so its
#: Δρ is not shown.
NO_MATCH = 0.2


def short(method: str, column) -> str:
    return f"{SHORT[method]}{int(column)}"


def both(what: str) -> dict[str, pd.DataFrame]:
    """One scores table, keyed by labelling."""
    return {lab: load(what, lab) for lab in LABELLINGS}


def pair_covariation() -> pd.DataFrame:
    """The runner's cross-labelling ``pair_covariation`` table."""
    return common.load_runner_output(
        scores_filename("pair_covariation", DATASET, "both"),
        COMPARATOR_SCORES_DIR,
        HINT,
    )


def beyond(t: pd.DataFrame, suffix: str = "") -> pd.Series:
    """Whether each component's ρ̄ is above the stricter of its two nulls'
    95th percentiles (`suffix` ``"_disjoint"`` for gene-disjoint pairs)."""
    nulls = t[[f"random_q95{suffix}", f"expression_matched_q95{suffix}"]].max(axis=1)
    return t[f"Observed{suffix}"] > nulls


@cache
def fit_components(method: str, labelling: str) -> scores.Components:
    path = COMPARATOR_FITS_DIR / comparator_filename(
        KEYS[method], "fit", DATASET, labelling
    )
    with xr.open_dataset(path, engine="h5netcdf") as fit:
        return scores.comparator_components(method, fit.load())


@cache
def lr_columns() -> dict[tuple, pd.Series]:
    """Every fit's LR columns on cccRISE's LR pairs, keyed (method,
    labelling, column); cccRISE's once, as coarse."""
    factors = load_factors_netcdf(DATASETS[DATASET].factors_path())
    index = factors["lr_pair"].values.astype(str)
    columns = {
        ("cccRISE", "coarse", k + 1): pd.Series(
            factors["cp_D"].values[:, k], index=index
        )
        for k in range(factors.sizes["cp_component"])
    }
    for lab in LABELLINGS:
        for m in COMPARATORS:
            lr = fit_components(m, lab).modes["lr"]
            columns |= {(m, lab, int(c)): lr[c].loc[index] for c in lr.columns}
    return columns


def abs_cos(u, v) -> float:
    u, v = np.asarray(u, float), np.asarray(v, float)
    return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def _column_class(row) -> str:
    if row["BestFDR"] < common.SIG_ALPHA:
        return "associated"
    if row["Contained"]:
        return "cccRISE contained"
    return "homophilic" if row["HomophilicShare"] >= HOMOPHILIC else "other"


def classified_labelling(labelling: str) -> pd.DataFrame:
    """Every component of one labelling's fits (cccRISE's too): its class
    and whether it is single-pair."""
    best = (
        load("associations", labelling).groupby(["Method", "Component"])["PAdj"].min()
    )
    breadth = load("breadth", labelling)
    top = breadth[breadth["Mode"] == "lr"].set_index(["Method", "Component"])[
        "TopShare"
    ]
    cont = load("containment", labelling)
    cont = cont[(cont["In"] == "cccRISE") & (cont["Mode"] == "lr")]
    contained = cont.set_index(["Of", "Component"])["Contained"]
    t = load("homophily", labelling)
    t = t[t["Method"].isin(("cccRISE", *COMPARATORS))]
    keys = list(zip(t["Method"], t["Component"], strict=True))
    t = t.assign(
        Labelling=labelling,
        BestFDR=best.reindex(keys).to_numpy(),
        TopShare=top.reindex(keys).to_numpy(),
        Contained=contained.reindex(keys).fillna(False).astype(bool).to_numpy(),
    )
    t = t.assign(
        Class=t.apply(_column_class, axis=1), SinglePair=t["TopShare"] >= SINGLE_PAIR
    )
    return t.assign(IsHomophilic=t["Class"] == "homophilic")


def classified() -> pd.DataFrame:
    """`classified_labelling` of both labellings, cccRISE's once (coarse)."""
    t = pd.concat([classified_labelling(lab) for lab in LABELLINGS], ignore_index=True)
    return t[(t["Method"] != "cccRISE") | (t["Labelling"] == "coarse")]


def capture_margin(reference: tuple, column: tuple, cos: float) -> str:
    """Δρ of `reference`'s top pairs in `column`'s context over the stricter
    null's 95th percentile (``pair_covariation``); "—" below `NO_MATCH`."""
    t = pair_covariation()
    keys = ("RefMethod", "RefLabelling", "RefColumn", "Method", "Labelling", "Column")
    match = t[
        np.logical_and.reduce(
            [t[k] == v for k, v in zip(keys, (*reference, *column), strict=True)]
        )
    ]
    if cos < NO_MATCH or match.empty:
        return "—"
    r = match.iloc[0]
    return f"{r['Observed'] - max(r['random_q95'], r['expression_matched_q95']):+.2f}"
