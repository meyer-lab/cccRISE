"""The per-component stability box plot shared by this figure's RISE and CP
panels, from ``uv run cccrise-stability``."""

import seaborn as sns
from matplotlib.patches import Patch

from ....runners.datasets import DATASETS, VALIDATION_DIR
from ... import common, style

DATASET = "alad"
COMPONENT_NAMES = {"RISE": "RISE", "CPD": "CP"}  # stage -> axis label


def plot_stage(fig, stage: str) -> None:
    """Per-component FMS over replicates for `stage` ('RISE' or 'CPD'), one
    box per stability axis. The grey seed-only box is that
    component's reproducibility ceiling: the gap below it is what perturbing
    the data costs beyond reseeding."""
    ax = fig.subplots()
    stability = common.load_runner_output(
        DATASETS[DATASET].stability_filename(), VALIDATION_DIR
    )
    rows = stability[stability["Stage"] == stage]
    axes = [a for a in style.STABILITY_AXIS_COLORS if a in set(rows["Axis"])]
    sns.boxplot(
        data=rows[rows["Axis"].isin(axes)],
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
