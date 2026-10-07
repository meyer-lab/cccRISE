"""The per-component stability box plot shared by this figure's RISE and CP
panels, and the instability rule its histogram panel selects components by,
from ``uv run cccrise-stability``."""

from functools import cache

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import kurtosis, skew

from ....runners.datasets import DATASETS, VALIDATION_DIR
from ... import common, style

DATASET = "alad"
COMPONENT_NAMES = {"RISE": "RISE", "CPD": "CP"}  # stage -> axis label
#: A replicate below this FMS did not re-identify the component.
UNSTABLE_FMS = 0.5
#: Sarle's bimodality coefficient above this suggests two modes (the value of
#: a uniform distribution). The coefficient is scale-free, so it also needs
#: the FMS to span `BIMODAL_MIN_RANGE`: a tight cluster at 0.999 with a tail
#: to 0.99 otherwise reads as bimodal.
BIMODAL_COEFFICIENT = 5 / 9
BIMODAL_MIN_RANGE = 0.25


@cache
def stability() -> pd.DataFrame:
    df = common.load_runner_output(
        DATASETS[DATASET].stability_filename(), VALIDATION_DIR
    )
    return df[df["Axis"].isin(style.STABILITY_AXIS_COLORS)]


def axes_present(df: pd.DataFrame) -> list[str]:
    """Stability axes in `df`, in `style.STABILITY_AXIS_COLORS` order."""
    return [a for a in style.STABILITY_AXIS_COLORS if a in set(df["Axis"])]


def bimodality_coefficient(fms: np.ndarray) -> float:
    """Sarle's bimodality coefficient, with sample-size corrected skewness and
    excess kurtosis; NaN for fewer than four replicates or a constant FMS."""
    n = len(fms)
    if n < 4 or np.ptp(fms) == 0:
        return np.nan
    g, k = skew(fms, bias=False), kurtosis(fms, bias=False)
    return (g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def unstable_components(df: pd.DataFrame) -> list[tuple[str, int]]:
    """(stage, component) pairs, RISE first, with at least one replicate below
    `UNSTABLE_FMS`, or a bimodality coefficient above `BIMODAL_COEFFICIENT`
    over an FMS range of at least `BIMODAL_MIN_RANGE`, on any axis."""
    flagged = df.groupby(["Stage", "Axis", "Component"])["FMS"].agg(
        lambda fms: (
            fms.min() < UNSTABLE_FMS
            or (
                np.ptp(fms) >= BIMODAL_MIN_RANGE
                and bimodality_coefficient(fms.to_numpy()) > BIMODAL_COEFFICIENT
            )
        )
    )
    unstable = flagged[flagged].reset_index()[["Stage", "Component"]].drop_duplicates()
    return [
        (stage, int(component))
        for stage in COMPONENT_NAMES
        for component in sorted(unstable.loc[unstable["Stage"] == stage, "Component"])
    ]


def plot_stage(fig, stage: str) -> None:
    """Per-component FMS over replicates for `stage` ('RISE' or 'CPD'), one
    box per stability axis. The grey seed-only box is that
    component's reproducibility ceiling: the gap below it is what perturbing
    the data costs beyond reseeding."""
    ax = fig.subplots()
    rows = stability()[stability()["Stage"] == stage]
    axes = axes_present(rows)
    sns.boxplot(
        data=rows,
        x="Component",
        y="FMS",
        hue="Axis",
        hue_order=axes,
        palette=style.STABILITY_AXIS_COLORS,
        fliersize=1,
        linewidth=style.LINE_WIDTH_THIN,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{COMPONENT_NAMES[stage]} component")
    ax.set_ylabel("Factor match score")


def legend(fig) -> None:
    """One legend of the stability-axis colors, above `fig`."""
    handles = [
        Patch(color=color, label=style.STABILITY_AXIS_LABELS[axis])
        for axis, color in style.STABILITY_AXIS_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        title="Refit",
        loc="outside upper center",
        ncols=len(handles),
    )
