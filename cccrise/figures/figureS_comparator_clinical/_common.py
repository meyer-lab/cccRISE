"""Shared loading for the comparator clinical-association figure: the
benchmark ANOVA (`cccrise-benchmark`) on the coarse and the fine annotation."""

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common

DATASET = "alad"
#: Annotation granularity -> the benchmark runner's output suffix.
SUFFIXES = {"coarse": "", "fine": "_fine"}
HINTS = {
    "coarse": f"    uv run cccrise-benchmark --dataset {DATASET}",
    "fine": f"    uv run cccrise-benchmark --dataset {DATASET} --rank-mode own "
    "--celltype-column celltype --suffix _fine",
}
METHODS = ("cccRISE", "Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
SHORT = {"Pseudobulk CPD": "PB CPD", "Tensor-cell2cell": "TC2C"}


def load(table: str, granularity: str) -> pd.DataFrame | None:
    """`benchmark_<table>` ("anova" or "anova_null") at `granularity`, own-rank
    comparators and cccRISE only; None if the run has not been made."""
    name = f"benchmark_{table}_{DATASET}{SUFFIXES[granularity]}.parquet"
    if not (VALIDATION_DIR / name).exists():
        return None
    df = common.load_runner_output(name, VALIDATION_DIR, hint=HINTS[granularity])
    return df[df["RankMode"].isin(["published", "own"])].reset_index(drop=True)


def fit_label(method: str, rank: str) -> str:
    return f"{SHORT.get(method, method)} ({rank})"
