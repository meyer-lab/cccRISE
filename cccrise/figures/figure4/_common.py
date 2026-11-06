"""Shared inputs of Figure 4 and its supplements: the comparator scores
(``uv run cccrise-comparator-scores``), on ALAD unless a panel names IBD, and
how each method and dataset is named and colored."""

import pandas as pd

from ...runners.datasets import COMPARATOR_SCORES_DIR, scores_filename
from .. import common, style

DATASET = "alad"
HINT = "    uv run cccrise-comparator-scores --dataset <dataset>"
#: The cohorts a panel may draw together, in order.
DATASET_LABELS = {"alad": "ALAD", "ibd": "IBD"}
DATASET_COLORS = {"alad": style.CATEGORICAL[6], "ibd": style.CATEGORICAL[5]}
#: Comparators of the main figure, in column order.
COMPARATORS = ("PB", "TC2C", "STAC")
METHOD_LABELS = {
    "cccRISE": "CCC-RISE",
    "PB": "PB",
    "TC2C": "TC2C",
    "TC2C inner": "TC2C (published)",
    "STAC": "STAC",
}
METHOD_COLORS = {
    "cccRISE": style.COLORS["emphasis"],
    "PB": style.CATEGORICAL[1],
    "TC2C": style.CATEGORICAL[2],
    "TC2C inner": style.CATEGORICAL[4],
    "STAC": style.CATEGORICAL[3],
}


def load(
    what: str, labelling: str | None = None, dataset: str = DATASET
) -> pd.DataFrame:
    """One scores table (`run_comparator_scores`): every labelling's rows,
    or only `labelling`'s, without the ``Labelling`` column."""
    t = common.load_runner_output(
        scores_filename(what, dataset), COMPARATOR_SCORES_DIR, HINT
    )
    if labelling is None:
        return t
    rows = t[t["Labelling"] == labelling]
    return rows.drop(columns="Labelling").reset_index(drop=True)
