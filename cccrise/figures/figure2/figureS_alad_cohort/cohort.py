"""Placeholder for a schematic of the ALAD cohort. The schematic is drawn by
hand, but its patient and sample counts per ALAD status come from the data
and are saved as its source data."""

from typing import cast

import pandas as pd

from ....runners.datasets import DATASETS
from ... import common

DATASET = "alad"
SCHEMATIC = True  # an illustration, but its sample sizes are source data
AXES = ("full",)


def plot(axs):
    (ax,) = axs
    spec = DATASETS[DATASET]
    obs = cast(pd.DataFrame, spec.load().obs)
    samples = obs.drop_duplicates(spec.condition_column)
    counts = samples.groupby(spec.group_column, observed=True).agg(
        patients=(spec.patient_column, "nunique"),
        samples=(spec.condition_column, "nunique"),
    )
    common.source_data(
        "cohort_sample_sizes",
        counts.rename_axis("ALAD status"),
        "Number of lung transplant recipients (patients) and BAL samples per "
        "ALAD status. A patient sampled at more than one status counts once "
        "under each.",
    )
    common.placeholder(
        ax,
        "ALAD cohort: 45 lung transplant recipients, 47 BAL samples\n"
        "(17 control, 30 ALAD: 17 declined, 13 recovered)",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
