"""Shared loading for the comparator-programs figure, from
``uv run cccrise-program-breadth`` on the coarse and the fine fits."""

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common

DATASET = "alad"
GRANULARITIES = {"coarse": "", "fine": "_fine"}
SHORT = {"Pseudobulk CPD": "PB CPD", "Tensor-cell2cell": "TC2C"}
METHODS = ("Pseudobulk CPD", "Tensor-cell2cell", "STACCato")


def load() -> pd.DataFrame:
    """program_breadth at both granularities; cccRISE (which uses no cell
    types for its fit) is kept once, from the coarse table."""
    frames = []
    for granularity, suffix in GRANULARITIES.items():
        hint = f"    python -m cccrise.runners.run_program_breadth --dataset {DATASET}"
        table = common.load_runner_output(
            f"program_breadth_{DATASET}{suffix}.parquet", VALIDATION_DIR, hint=hint
        ).assign(granularity=granularity)
        if granularity == "fine":
            table = table[table["method"] != "cccRISE"]
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def fits(table: pd.DataFrame) -> list[tuple[str, str]]:
    """(method, granularity) in display order, cccRISE first."""
    return [("cccRISE", "coarse")] + [
        (m, g)
        for m in METHODS
        for g in GRANULARITIES
        if ((table["method"] == m) & (table["granularity"] == g)).any()
    ]


def fit_label(method: str, granularity: str, table: pd.DataFrame) -> str:
    rank = table.loc[
        (table["method"] == method) & (table["granularity"] == granularity), "rank"
    ].iloc[0]
    if method == "cccRISE":
        return f"cccRISE\n({rank})"
    return f"{SHORT.get(method, method)} {granularity}\n({rank})"
