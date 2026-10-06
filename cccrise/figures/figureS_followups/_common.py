"""Shared loading for the follow-up figure: the comparators refit on
tempered pseudobulk tensors (F1; ``cccrise-benchmark --tensor-transform``)
and cccRISE's components tested within single cell types (F2)."""

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common

DATASET = "alad"
#: Tensor variant -> runner output suffix.
VARIANTS = {"none": "", "winsor": "_winsor", "blocknorm": "_blocknorm"}
VARIANT_LABELS = {"none": "raw", "winsor": "winsorised", "blocknorm": "block-normed"}
METHODS = ("Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
SHORT = {"Pseudobulk CPD": "PB CPD", "Tensor-cell2cell": "TC2C", "STACCato": "STACCato"}


def load(stem: str) -> pd.DataFrame:
    """`<stem>_alad<suffix>.parquet` for every tensor variant, stacked with a
    'variant' column."""
    frames = [
        common.load_runner_output(
            f"{stem}_{DATASET}{suffix}.parquet", VALIDATION_DIR
        ).assign(variant=variant)
        for variant, suffix in VARIANTS.items()
    ]
    return pd.concat(frames, ignore_index=True)
