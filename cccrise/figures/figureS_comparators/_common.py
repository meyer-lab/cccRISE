"""Shared loading for the comparator-agreement figure."""

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common

DATASET = "alad"
HINT = f"    uv run cccrise-comparator-agreement --dataset {DATASET}"
MODE_LABELS = {"condition": "Condition mode", "lr": "LR mode"}


def load(name: str) -> pd.DataFrame:
    """One `cccrise-comparator-agreement` table: subspace, containment,
    matching, cell_fractions or celltype_modes."""
    return common.load_runner_output(
        f"comparator_{name}_{DATASET}.parquet", VALIDATION_DIR, hint=HINT
    )


def fit_order(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(method, rank mode, rank) of every comparator fit, in a fixed order."""
    order = {"Pseudobulk CPD": 0, "Tensor-cell2cell": 1, "STACCato": 2}
    fits = table[["method", "rank_mode", "rank"]].drop_duplicates()
    return sorted(
        fits.itertuples(index=False, name=None),
        key=lambda f: (order.get(f[0], 9), f[1] != "own"),
    )


def fit_label(fit: tuple[str, str, str]) -> str:
    method, mode, rank = fit
    short = {"Pseudobulk CPD": "PB CPD", "Tensor-cell2cell": "TC2C"}.get(method, method)
    return f"{short} {mode} ({rank})"
