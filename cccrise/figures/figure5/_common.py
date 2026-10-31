"""The dataset Figure 5 and its supplements draw -- the IBDverse terminal
ileum cohort (the registry's `ibd` entry: RISE 60 / CP 6, mesenchymal lineage
excluded, at most 2,000 cells per biopsy) -- and the `cccrise-cohort` tables
that summarize its biopsies and cell types."""

from functools import cache

import pandas as pd

from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common

DATASET = "ibd"
SPEC = DATASETS[DATASET]

#: `inflammation_status` levels, in disease order, and how figures name them
GROUP_LABELS = {
    "Healthy": "Healthy",
    "CD_uninflamed": "CD uninflamed",
    "CD_inflamed": "CD inflamed",
}


def _load(table: str) -> pd.DataFrame:
    return common.load_runner_output(
        f"{table}_{SPEC.output_key}.parquet",
        VALIDATION_DIR,
        hint=f"    uv run cccrise-cohort --dataset {DATASET}",
    )


@cache
def samples() -> pd.DataFrame:
    """One row per biopsy: its sample-level metadata, cell count, and raw and
    read-depth-corrected CP loadings (`run_cohort.sample_table`)."""
    return _load("cohort_samples")


@cache
def celltype_counts() -> pd.DataFrame:
    """Cells per biopsy x lineage x fine cell type."""
    return _load("cohort_celltype_counts")
