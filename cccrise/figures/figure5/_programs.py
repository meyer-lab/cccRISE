"""Each comparator fit's LR columns, and each cccRISE component's best single
column in every fit, from a cohort's scores tables."""

from functools import cache

import numpy as np
import pandas as pd

from ... import covariation
from ._common import COMPARATORS, load
from ._data import LABELLINGS

#: Every comparator fit, in the order panels draw them.
FITS = [(m, lab) for m in COMPARATORS for lab in LABELLINGS]


def lr_matrix(dataset: str, method: str, labelling: str = "coarse") -> pd.DataFrame:
    """`method`'s LR columns (pairs by columns, unscored pairs 0)."""
    t = load("lr_loadings", labelling, dataset)
    return (
        t[t["Method"] == method]
        .pivot(index="LRPair", columns="Component", values="Loading")
        .fillna(0)
    )


def _unit(columns: pd.DataFrame) -> pd.DataFrame:
    return columns / np.linalg.norm(columns.to_numpy(), axis=0, keepdims=True)


@cache
def carrying_pairs(dataset: str) -> dict[int, list[str]]:
    """Each program's carrying pairs (`covariation.carrying_pairs`)."""
    lr = lr_matrix(dataset, "cccRISE")
    return {
        int(c): lr.index[covariation.carrying_pairs(lr[c].to_numpy())].tolist()
        for c in lr.columns
    }


@cache
def best_columns(dataset: str) -> pd.DataFrame:
    """Each program's largest cos^2 with one LR column of each fit."""
    programs = _unit(lr_matrix(dataset, "cccRISE"))
    rows = []
    for method, lab in FITS:
        columns = _unit(
            lr_matrix(dataset, method, lab).reindex(programs.index).fillna(0)
        )
        cos2 = (programs.to_numpy().T @ columns.to_numpy()) ** 2
        rows += [
            {"Program": int(c), "Method": method, "Labelling": lab, "Cos2": best}
            for c, best in zip(programs.columns, cos2.max(axis=1), strict=True)
        ]
    return pd.DataFrame(rows)
