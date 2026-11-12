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
    """The full-cohort ANOVA per component, from the sidecar, with p
    Benjamini-Hochberg corrected across components (`PAdj`)."""
    meta = metadata()
    n = len(meta["reference_eta_squared"])
    return pd.DataFrame(
        {
            "Component": np.arange(1, n + 1),
            "EtaSquared": meta["reference_eta_squared"],
            "P": meta["reference_p"],
            "PAdj": meta["reference_p_adj"],
        }
    )


#: Columns of `replicates` exported as source data
REPLICATE_COLUMNS = [
    "Replicate",
    "Seed",
    "LeftOut",
    "Component",
    "MatchedComponent",
    "FMS",
    "F",
    "EtaSquared",
    "P",
    "PAdj",
]


def save_source_data() -> None:
    """Every jackknife replicate's ANOVA per component, and the full-cohort
    reference, as this figure's panels draw them."""
    common.source_data(
        "anova_jackknife_replicates",
        replicates()[REPLICATE_COLUMNS],
        "One row per (leave-one-patient-out replicate, CPD component): the "
        "left-out patient (LeftOut), the refit component matched to the "
        "full-cohort component (MatchedComponent, by factor match score FMS), "
        f"and its one-way ANOVA against {outcome()}. P is raw; PAdj is "
        "Benjamini-Hochberg corrected across the replicate's components.",
    )
    common.source_data(
        "anova_full_cohort",
        reference(),
        f"One row per CPD component: its full-cohort one-way ANOVA against "
        f"{outcome()} (the diamonds). P is raw; PAdj is Benjamini-Hochberg "
        f"corrected across components; filled where PAdj < {alpha():g}.",
    )


def alpha() -> float:
    return float(metadata()["alpha"])


def outcome() -> str:
    return metadata()["group_column"]


def components() -> list[int]:
    return reference()["Component"].astype(int).tolist()


def reference_markers(ax, x, y) -> None:
    """Full-cohort values as diamonds, filled where q < alpha."""
    significant = reference()["PAdj"].to_numpy() < alpha()
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
        Line2D([], [], marker="D", color="black", lw=0, label=f"Full cohort, q < {alpha():g}"),
        Line2D(
            [], [], marker="D", color="black", markerfacecolor="white", lw=0,
            label="Full cohort, n.s.",
        ),
    ]  # fmt: skip
