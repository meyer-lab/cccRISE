"""Figure 4's derived inputs: the scores tables of both labellings, each fit's
LR columns, and what each LR column is (`classified`)."""

from functools import cache

import numpy as np
import pandas as pd

from .. import common
from ._common import COMPARATORS, DATASET, load

LABELLINGS = ("coarse", "fine")
#: Shorthand for a method's components: C (cccRISE), P, T, S.
SHORT = {"cccRISE": "C", "PB": "P", "TC2C": "T", "STAC": "S"}
#: A column is homophilic when homophilic pairs carry at least this share of
#: its squared loading.
HOMOPHILIC = 0.8
#: A column is single-pair, whatever its class, when its top pair carries at
#: least this share of its squared loading.
SINGLE_PAIR = 0.8
#: Below this |cosine| a column carries none of a reference's content, so its
#: Δρ is not shown.
NO_MATCH = 0.2


def short(method: str, column) -> str:
    return f"{SHORT[method]}{int(column)}"


def source_data(name: str, t: pd.DataFrame, columns: dict, description: str) -> None:
    """`common.source_data` of `t`'s components, each named as the figure
    names it (`short`) with its method and labelling, and `t`'s `columns`
    renamed to their values."""
    named = t.assign(
        Name=[short(m, c) for m, c in zip(t["Method"], t["Component"], strict=True)]
    )
    keys = {"Name": "Component", "Method": "Method", "Labelling": "Labelling"}
    common.source_data(
        name, named[list(keys | columns)].rename(columns=keys | columns), description
    )


def both(what: str, dataset: str = DATASET) -> dict[str, pd.DataFrame]:
    """One scores table, keyed by labelling."""
    return {lab: load(what, lab, dataset) for lab in LABELLINGS}


def beyond(t: pd.DataFrame, suffix: str = "") -> pd.Series:
    """Whether each component's ρ̄ is above the stricter of its two nulls'
    95th percentiles (`suffix` ``"_disjoint"`` for gene-disjoint pairs)."""
    nulls = t[[f"random_q95{suffix}", f"expression_matched_q95{suffix}"]].max(axis=1)
    return t[f"Observed{suffix}"] > nulls


def fit_lr(method: str, labelling: str) -> pd.DataFrame:
    """`method`'s LR columns (one per component) on every LR pair its fit
    has, from the runner's ``lr_loadings``."""
    t = load("lr_loadings", labelling)
    return t[t["Method"] == method].pivot(
        index="LRPair", columns="Component", values="Loading"
    )


@cache
def lr_columns() -> dict[tuple, pd.Series]:
    """Every fit's LR columns on cccRISE's LR pairs, keyed (method,
    labelling, column); cccRISE's once, as coarse."""
    reference = fit_lr("cccRISE", "coarse")
    index = reference.index
    columns = {("cccRISE", "coarse", int(c)): reference[c] for c in reference}
    for lab in LABELLINGS:
        for m in COMPARATORS:
            lr = fit_lr(m, lab)
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


def capture_margin(reference: tuple, column: tuple, cos: float) -> float:
    """Δρ of `reference`'s top pairs in `column`'s context over the stricter
    null's 95th percentile (``pair_covariation``); NaN below `NO_MATCH`."""
    t = load("pair_covariation")
    keys = ("RefMethod", "RefLabelling", "RefColumn", "Method", "Labelling", "Column")
    match = t[
        np.logical_and.reduce(
            [t[k] == v for k, v in zip(keys, (*reference, *column), strict=True)]
        )
    ]
    if cos < NO_MATCH or match.empty:
        return np.nan
    r = match.iloc[0]
    return r["Observed"] - max(r["random_q95"], r["expression_matched_q95"])
