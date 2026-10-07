"""The per-component stability box plot shared by this figure's RISE and CP
panels, and the bimodality rule its histogram panel selects components by,
from ``uv run cccrise-stability``."""

from functools import cache

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from ....runners.datasets import DATASETS, VALIDATION_DIR
from ... import common, style

DATASET = "alad"
COMPONENT_NAMES = {"RISE": "RISE", "CPD": "CP"}  # stage -> axis label
#: A component's replicate FMS is bimodal when its sorted values have a gap of
#: at least `BIMODAL_GAP` with at least `BIMODAL_MINOR_FRACTION` of the
#: replicates on the smaller side -- a second population rather than one or
#: two replicates that failed to re-identify the component.
BIMODAL_GAP = 0.25
BIMODAL_MINOR_FRACTION = 0.1


@cache
def stability() -> pd.DataFrame:
    df = common.load_runner_output(
        DATASETS[DATASET].stability_filename(), VALIDATION_DIR
    )
    return df[df["Axis"].isin(style.STABILITY_AXIS_COLORS)]


def axes_present(df: pd.DataFrame) -> list[str]:
    """Stability axes in `df`, in `style.STABILITY_AXIS_COLORS` order."""
    return [a for a in style.STABILITY_AXIS_COLORS if a in set(df["Axis"])]


def is_bimodal(fms: np.ndarray) -> bool:
    """Whether `fms` splits into two populations at its largest gap; see
    `BIMODAL_GAP` and `BIMODAL_MINOR_FRACTION`."""
    fms = np.sort(np.asarray(fms, dtype=float))
    if fms.size < 2:
        return False
    split = int(np.argmax(np.diff(fms)))
    minor = min(split + 1, fms.size - split - 1)
    return bool(
        fms[split + 1] - fms[split] >= BIMODAL_GAP
        and minor >= BIMODAL_MINOR_FRACTION * fms.size
    )


def bimodal_components(df: pd.DataFrame) -> list[tuple[str, int]]:
    """(stage, component) pairs, RISE first, whose replicate FMS is bimodal
    (`is_bimodal`) on any axis."""
    flagged = df.groupby(["Stage", "Axis", "Component"])["FMS"].agg(
        lambda fms: is_bimodal(fms.to_numpy())
    )
    bimodal = flagged[flagged].reset_index()[["Stage", "Component"]].drop_duplicates()
    return [
        (stage, int(component))
        for stage in COMPONENT_NAMES
        for component in sorted(bimodal.loc[bimodal["Stage"] == stage, "Component"])
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
