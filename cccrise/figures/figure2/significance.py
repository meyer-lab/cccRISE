"""Permutation significance of communication and of differential signaling,
from ``uv run cccrise-significance``."""

import pandas as pd

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...runners.provenance import read_metadata
from .. import common, style

DATASET = "alad"


def plot(fig):
    spec = DATASETS[DATASET]
    lr_table = common.load_runner_output(
        spec.significance_lr_filename(), VALIDATION_DIR
    )
    differential = pd.read_parquet(
        VALIDATION_DIR / spec.significance_differential_filename()
    )
    meta = read_metadata(VALIDATION_DIR / spec.significance_lr_filename())

    axs = fig.subplots(1, 3)
    common.plot_significance_calibration(
        lr_table,
        axs[0],
        cohort_rates={
            "p < 0.05": meta["cohort_fraction_significant_p05"],
            "p < 0.01": meta["cohort_fraction_significant_p01"],
        },
    )
    common.plot_observed_vs_null(lr_table, axs[1])
    common.plot_differential_volcano(
        differential,
        axs[2],
        contrast=f"{meta['other_label']} - {meta['control_label']}",
    )
    for ax, title in zip(
        axs,
        ("Null calibration", "Observed vs. null", "Differential communication"),
        strict=True,
    ):
        ax.set_title(title)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
