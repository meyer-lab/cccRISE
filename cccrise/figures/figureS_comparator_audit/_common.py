"""The audit of the comparator programs cccRISE does not contain
(``run_comparator_audit``)."""

from functools import cache

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common, style

DATASET = "alad"
SHORT = {"Pseudobulk CPD": "PB", "Tensor-cell2cell": "TC2C", "STACCato": "STAC"}
CLASSES = {
    "one sample": style.COLORS["muted"],
    "marker or homophilic pair": style.COLORS["reference"],
    "abundance readout": style.COLORS["sender"],
    "candidate biology": style.COLORS["emphasis"],
}
OUTCOME = "6-Month Outcome (4-Way)"
RISE_FITS = {"canonical": "RISE 17", "alad_rise34": "RISE 34", "alad_rise51": "RISE 51"}


@cache
def audit() -> pd.DataFrame:
    table = common.load_runner_output(
        f"comparator_audit_{DATASET}.parquet",
        VALIDATION_DIR,
        hint=f"    uv run python -m cccrise.runners.run_comparator_audit --dataset {DATASET}",
    )
    return table.assign(fit=[f"{SHORT[m]} {a}" for m, a in zip(table["method"], table["annotation"], strict=True)])


def pair_label(row) -> str:
    return row.top_pair.replace("^", "–")
