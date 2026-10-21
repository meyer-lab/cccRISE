"""Loader shared by this figure's panels: the ridge comparison from
``uv run cccrise-cpdrank`` -- full-data CP fits of the canonical tensor with and
without the masked-imputation ridge, on the imputation backend (the GPU), and
the canonical CPU core fit compared against both at the selected rank."""

import pandas as pd

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from ...runners.provenance import read_metadata
from .. import common, style

DATASET = "alad"
#: Cross markers for the canonical CPU core fit, drawn over the GPU curves
CORE_MARKER_SIZE = 4 * style.POINT_SIZE
CORE_MARKER = {"s": CORE_MARKER_SIZE, "zorder": 3}


def load() -> tuple[pd.DataFrame, pd.Series, float]:
    """Per-component rows, the selected rank's row (with the 'Core*' columns),
    and the ridge strength."""
    df = common.load_cpd_rank(DATASET)
    sidecar = read_metadata(RANK_SELECTION_DIR / DATASETS[DATASET].cpd_rank_filename())
    rank = common.selected_ranks(DATASET)[1]
    return df, df[df["Rank"] == rank].iloc[0], sidecar["imputation_options"]["l2_reg"]
