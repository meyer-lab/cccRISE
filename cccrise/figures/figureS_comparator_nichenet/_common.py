"""Shared loading for the comparator NicheNet figure."""

import pandas as pd

from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common

DATASET = "alad"

#: (method, rank mode) in display order; cccRISE first, as the reference.
FITS = [
    ("cccRISE", "published"),
    ("Pseudobulk CPD", "own"),
    ("Pseudobulk CPD", "shared"),
    ("Tensor-cell2cell", "own"),
    ("Tensor-cell2cell", "shared"),
]


def all_scores() -> pd.DataFrame:
    """cccRISE's NicheNet scores (`cccrise-nichenet`) stacked with every
    comparator fit's (`cccrise-nichenet --comparators`)."""
    spec = DATASETS[DATASET]
    cccrise = common.load_runner_output(
        spec.nichenet_scores_filename(),
        VALIDATION_DIR,
        hint=f"    uv run cccrise-nichenet --dataset {DATASET}",
    ).assign(method="cccRISE", rank_mode="published", rank="")
    comparators = common.load_runner_output(
        spec.nichenet_comparator_scores_filename(),
        VALIDATION_DIR,
        hint=f"    uv run cccrise-nichenet --dataset {DATASET} --comparators",
    )
    return pd.concat([cccrise, comparators], ignore_index=True)


def fit_label(method: str, rank_mode: str, scores: pd.DataFrame) -> str:
    if method == "cccRISE":
        return "cccRISE"
    rank = scores.loc[
        (scores["method"] == method) & (scores["rank_mode"] == rank_mode), "rank"
    ].iloc[0]
    return f"{method} ({rank_mode}, rank {rank})"
