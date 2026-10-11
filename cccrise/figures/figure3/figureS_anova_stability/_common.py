"""The jackknifed CP component--outcome ANOVA this figure plots, from
``uv run cccrise-stability``.

Every jackknife replicate leaves one patient out, refits RISE and CP, matches
its components back onto the full-cohort reference by the CPD factor match
score, and records their ANOVA statistics, so every component's effect size
comes with a leave-one-patient-out distribution. Components keep the
reference's numbering.
"""

from functools import cache

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ....runners.datasets import DATASETS, VALIDATION_DIR
from ....runners.provenance import read_metadata
from ....validation.stability import JACKKNIFE_AXIS, STAGE_CPD
from ... import common, style

DATASET = "alad"


@cache
def metadata(dataset: str = DATASET) -> dict:
    return read_metadata(VALIDATION_DIR / DATASETS[dataset].stability_filename())


@cache
def replicates(dataset: str = DATASET) -> pd.DataFrame:
    """Jackknife CPD rows, one per (replicate, component)."""
    df = common.load_runner_output(
        DATASETS[dataset].stability_filename(), VALIDATION_DIR
    )
    return df[(df["Axis"] == JACKKNIFE_AXIS) & (df["Stage"] == STAGE_CPD)]


@cache
def reference(dataset: str = DATASET) -> pd.DataFrame:
    """The full-cohort ANOVA per component, from the sidecar."""
    meta = metadata(dataset)
    n = len(meta["reference_eta_squared"])
    return pd.DataFrame(
        {
            "Component": np.arange(1, n + 1),
            "EtaSquared": meta["reference_eta_squared"],
            "P": meta["reference_p"],
        }
    )


def alpha(dataset: str = DATASET) -> float:
    return float(metadata(dataset)["alpha"])


def outcome(dataset: str = DATASET) -> str:
    return metadata(dataset)["group_column"]


def components(dataset: str = DATASET) -> list[int]:
    return reference(dataset)["Component"].astype(int).tolist()


def reference_markers(ax, x, y, dataset: str = DATASET) -> None:
    """Full-cohort values as diamonds, filled where p < alpha."""
    significant = reference(dataset)["P"].to_numpy() < alpha(dataset)
    for filled in (True, False):
        keep = significant == filled
        ax.scatter(
            np.asarray(x)[keep],
            np.asarray(y)[keep],
            marker="D",
            s=style.POINT_SIZE,
            facecolor="black" if filled else "white",
            edgecolor="black",
            linewidth=style.LINE_WIDTH_THIN,
            zorder=5,
        )


def reference_legend_handles(dataset: str = DATASET) -> list:
    return [
        Line2D([], [], marker="D", color="black", lw=0, label=f"Full cohort, p < {alpha(dataset):g}"),
        Line2D(
            [], [], marker="D", color="black", markerfacecolor="white", lw=0,
            label="Full cohort, n.s.",
        ),
    ]  # fmt: skip
