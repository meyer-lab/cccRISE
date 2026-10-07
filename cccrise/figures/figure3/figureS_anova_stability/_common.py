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
FILENAME = DATASETS[DATASET].stability_filename()


@cache
def metadata() -> dict:
    return read_metadata(VALIDATION_DIR / FILENAME)


@cache
def replicates() -> pd.DataFrame:
    """Jackknife CPD rows, one per (replicate, component)."""
    df = common.load_runner_output(FILENAME, VALIDATION_DIR)
    return df[(df["Axis"] == JACKKNIFE_AXIS) & (df["Stage"] == STAGE_CPD)]


@cache
def reference() -> pd.DataFrame:
    """The full-cohort ANOVA per component, from the sidecar."""
    meta = metadata()
    n = len(meta["reference_eta_squared"])
    return pd.DataFrame(
        {
            "Component": np.arange(1, n + 1),
            "EtaSquared": meta["reference_eta_squared"],
            "P": meta["reference_p"],
        }
    )


def alpha() -> float:
    return float(metadata()["alpha"])


def outcome() -> str:
    return metadata()["group_column"]


def components() -> list[int]:
    return reference()["Component"].astype(int).tolist()


def reference_markers(ax, x, y) -> None:
    """Full-cohort values as diamonds, filled where p < alpha."""
    significant = reference()["P"].to_numpy() < alpha()
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


def reference_legend_handles() -> list:
    return [
        Line2D([], [], marker="D", color="black", lw=0, label=f"Full cohort, p < {alpha():g}"),
        Line2D(
            [], [], marker="D", color="black", markerfacecolor="white", lw=0,
            label="Full cohort, n.s.",
        ),
    ]  # fmt: skip
