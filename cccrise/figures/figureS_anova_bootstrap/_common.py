"""The resampled CP component--outcome ANOVA shared by this figure's panels,
from ``uv run cccrise-stability``.

Every replicate of every stability axis refits CP (most also refit RISE),
matches its components back onto the full-cohort reference by the CPD factor
match score, and records their ANOVA statistics, so every component's effect
size comes with a distribution per axis. Components keep the reference's
numbering.
"""

from functools import cache

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...runners.provenance import read_metadata
from ...validation.stability import STAGE_CPD, summarize_anova_stability
from .. import common, style

DATASET = "alad"
FILENAME = DATASETS[DATASET].stability_filename()


@cache
def metadata() -> dict:
    return read_metadata(VALIDATION_DIR / FILENAME)


@cache
def replicates() -> pd.DataFrame:
    """CPD rows of every axis, one per (axis, replicate, component)."""
    df = common.load_runner_output(FILENAME, VALIDATION_DIR)
    return df[df["Stage"] == STAGE_CPD]


def axes() -> list[str]:
    """Axes present, in `style.STABILITY_AXIS_COLORS` order."""
    present = set(replicates()["Axis"])
    return [axis for axis in style.STABILITY_AXIS_COLORS if axis in present]


@cache
def reference() -> pd.DataFrame:
    """The full-cohort ANOVA and top LR pair per component, from the sidecar."""
    meta = metadata()
    n = len(meta["reference_eta_squared"])
    return pd.DataFrame(
        {
            "Component": np.arange(1, n + 1),
            "EtaSquared": meta["reference_eta_squared"],
            "F": meta["reference_f"],
            "P": meta["reference_p"],
            "GroupMeanDiff": meta["reference_group_mean_diff"],
            "TopLRPair": meta["reference_top_lr_pair"],
        }
    )


@cache
def summary() -> pd.DataFrame:
    """Per (axis, component): the effect-size interval (jackknife: standard
    error), significance / sign stability, and the reference statistics."""
    meta = metadata()
    return pd.concat(
        [
            summarize_anova_stability(
                replicates(), reference(), axis, meta["ci_level"], alpha()
            ).assign(Axis=axis)
            for axis in axes()
        ],
        ignore_index=True,
    )


def alpha() -> float:
    return float(metadata()["alpha"])


def outcome() -> str:
    return metadata()["group_column"]


def components() -> list[int]:
    return reference()["Component"].astype(int).tolist()


def dodge(n: int, width: float = 0.8) -> np.ndarray:
    """Offsets of `n` side-by-side marks within one category of `width`."""
    return (np.arange(n) - (n - 1) / 2) * width / n


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


def legend(fig) -> None:
    """One legend of the resampling-axis colors, above `fig`."""
    handles = [Patch(color=style.STABILITY_AXIS_COLORS[a], label=a) for a in axes()]
    fig.legend(
        handles=handles,
        title="Resampled",
        loc="outside upper center",
        ncols=len(handles),
    )
