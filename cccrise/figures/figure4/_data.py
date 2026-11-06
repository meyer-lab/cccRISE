"""Figure 4's shared names, and the scores tables of both labellings."""

import pandas as pd

from .. import common
from ._common import DATASET, load

LABELLINGS = ("coarse", "fine")
#: Shorthand for a method's components: C (cccRISE), P, T, S.
SHORT = {"cccRISE": "C", "PB": "P", "TC2C": "T", "STAC": "S"}
#: A column is homophilic when homophilic pairs carry at least this share of
#: its squared loading.
HOMOPHILIC = 0.8
#: A column is single-pair, whatever its class, when its top pair carries at
#: least this share of its squared loading.
SINGLE_PAIR = 0.8


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
