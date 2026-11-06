"""What becomes of each cccRISE program in the comparators' fits, and what each
comparator LR column is on its own terms, from the scores tables of a cohort."""

from functools import cache

import numpy as np
import pandas as pd

from ... import covariation
from .. import style
from ._common import COMPARATORS, DATASET_LABELS, METHOD_COLORS, load
from ._data import HOMOPHILIC, LABELLINGS

#: Every comparator fit, in the order panels draw them.
FITS = [(m, lab) for m in COMPARATORS for lab in LABELLINGS]
#: A column is part of a program when the program's carrying pairs hold at
#: least this share of its squared loading.
PART_SHARE = 0.5
#: A column has no clear axis at this many effective LR pairs or more.
BROAD_PAIRS = 20
#: A column is a single pair under this many effective LR pairs.
SINGLE_PAIRS = 1.5
#: A STAC column is tested against a program when it holds at least this
#: share of the program's squared loading.
TESTED_SHARE = 0.1
#: What a comparator column is, in stacking order, and its color; the first
#: rule that fits (`_column_class`).
COLUMN_CLASSES = {
    "Same as a cccRISE program": METHOD_COLORS["cccRISE"],
    "Part of a cccRISE program": "#9fbbd6",
    "Homophilic adhesion": "0.45",
    "Broad": "0.8",
    "Single pair": "#f0c9a0",
    "Mixed pairs": style.CATEGORICAL[3],
}


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


def _column_class(row) -> str:
    if row["Contained"]:
        return "Same as a cccRISE program"
    if row["OnProgram"] >= PART_SHARE:
        return "Part of a cccRISE program"
    if row["EffectiveLR"] >= BROAD_PAIRS:
        return "Broad"
    if row["HomophilicShare"] >= HOMOPHILIC:
        return "Homophilic adhesion"
    return "Single pair" if row["EffectiveLR"] < SINGLE_PAIRS else "Mixed pairs"


def _on_programs(dataset: str, method: str, labelling: str) -> pd.DataFrame:
    """Each column's largest share of squared loading on one program's
    carrying pairs, and that program."""
    squared = lr_matrix(dataset, method, labelling) ** 2
    shares = pd.DataFrame(
        {
            program: (squared.reindex(pairs).fillna(0) / squared.sum()).sum()
            for program, pairs in carrying_pairs(dataset).items()
        }
    )
    return pd.DataFrame(
        {
            "Component": shares.index,
            "OnProgram": shares.max(axis=1).to_numpy(),
            "Program": shares.idxmax(axis=1).to_numpy(),
        }
    )


@cache
def column_census(dataset: str) -> pd.DataFrame:
    """Every comparator column: its containment, breadth, homophilic share,
    overlap with its closest program, and class (`COLUMN_CLASSES`)."""
    frames = []
    for lab in LABELLINGS:
        contained = load("containment", lab, dataset)
        contained = contained[
            (contained["In"] == "cccRISE") & (contained["Mode"] == "lr")
        ]
        contained = contained.rename(columns={"Of": "Method"})
        keys = ["Method", "Component"]
        t = (
            contained[[*keys, "Containment", "Contained"]]
            .merge(load("covariation", lab, dataset)[[*keys, "EffectiveLR"]], on=keys)
            .merge(load("homophily", lab, dataset), on=keys)
        )
        for method in COMPARATORS:
            rows = t[t["Method"] == method].merge(
                _on_programs(dataset, method, lab), on="Component"
            )
            frames.append(rows.assign(Labelling=lab, Dataset=DATASET_LABELS[dataset]))
    census = pd.concat(frames, ignore_index=True)
    return census.assign(Class=census.apply(_column_class, axis=1))


def program_homes(dataset: str, program: int, labelling: str) -> pd.DataFrame:
    """Each carrying pair of `program`: its share of the program's squared
    loading and the STAC column holding most of the pair (none if it has none)."""
    lr = lr_matrix(dataset, "cccRISE")[program] ** 2
    squared = lr_matrix(dataset, "STAC", labelling) ** 2
    pairs = carrying_pairs(dataset)[program]
    held = squared.reindex(pairs).fillna(0)
    return pd.DataFrame(
        {
            "LRPair": pairs,
            "Share": (lr / lr.sum()).loc[pairs].to_numpy(),
            "Column": held.idxmax(axis=1).where(held.sum(axis=1) > 0).to_numpy(),
        }
    )


def tested_columns(dataset: str, programs) -> list[tuple[int, str, int]]:
    """(program, labelling, column), largest share first, of every STAC column
    holding at least `TESTED_SHARE` of a program, and with pairs to test."""
    out = []
    for program in programs:
        for lab in LABELLINGS:
            covaried = load("covariation", lab, dataset)
            covaried = covaried[covaried["Method"] == "STAC"].set_index("Component")
            homes = program_homes(dataset, program, lab)
            shares = homes.groupby("Column")["Share"].sum().sort_values(ascending=False)
            out += [
                (program, lab, int(column))
                for column, share in shares.items()
                if share >= TESTED_SHARE
                and np.isfinite(covaried.loc[column, "Observed"])
            ]
    return out
