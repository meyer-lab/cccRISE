"""Shared loading for the granularity figure: `cccrise-comparator-agreement`
tables for the comparators fit on coarse ("") and fine ("_fine") cell types."""

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common

DATASET = "alad"
GRANULARITIES = {"coarse": "", "fine": "_fine"}
METHODS = ("Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
SHORT = {"Pseudobulk CPD": "PB CPD", "Tensor-cell2cell": "TC2C", "STACCato": "STACCato"}


def load(name: str) -> pd.DataFrame:
    """One agreement table at both granularities, own-rank fits only, with a
    'granularity' column."""
    frames = []
    for granularity, suffix in GRANULARITIES.items():
        hint = f"    uv run cccrise-comparator-agreement --dataset {DATASET}" + (
            " --celltype-column celltype --suffix _fine" if suffix else ""
        )
        table = common.load_runner_output(
            f"comparator_{name}_{DATASET}{suffix}.parquet", VALIDATION_DIR, hint=hint
        )
        if "rank_mode" in table:
            table = table[table["rank_mode"] == "own"]
        frames.append(table.assign(granularity=granularity))
    return pd.concat(frames, ignore_index=True)
