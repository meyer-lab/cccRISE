"""Aggregate RISE and CP stability per stability axis (the Methods'
stability metric: a replicate's mean FMS over components), as the median with
a 2.5-97.5 percentile band over replicates, from ``uv run cccrise-stability``."""

import numpy as np
from matplotlib.lines import Line2D

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...validation.stability import metric_interval
from .. import common, style

DATASET = "alad"
STAGE_ORDER = ["RISE", "CPD"]  # pipeline order
STAGE_LABELS = {"RISE": "RISE", "CPD": "CP"}
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    stability = common.load_runner_output(
        DATASETS[DATASET].stability_filename(), VALIDATION_DIR
    )
    stats = metric_interval(
        stability[stability["Axis"].isin(style.STABILITY_AXIS_COLORS)],
        "FMS",
        by=("Stage", "Axis"),
        lower=2.5,
        upper=97.5,
    )
    stats = stats.sort_values(
        "Stage", key=lambda s: s.map(STAGE_ORDER.index), kind="stable"
    )
    # Points grouped by stage, one per stability axis, a blank slot between.
    stages = list(dict.fromkeys(stats["Stage"]))
    within = stats.groupby("Stage").cumcount().to_numpy()
    sizes = stats.groupby("Stage").size()
    offset = np.cumsum([0] + [sizes[s] + 1 for s in stages[:-1]])
    x = offset[[stages.index(s) for s in stats["Stage"]]] + within
    ax.errorbar(
        x,
        stats["Median"],
        yerr=[stats["Median"] - stats["P2.5"], stats["P97.5"] - stats["Median"]],
        fmt="none",
        ecolor=style.COLORS["ink"],
        capsize=2,
        linewidth=style.LINE_WIDTH_THIN,
    )
    ax.scatter(
        x,
        stats["Median"],
        c=[style.STABILITY_AXIS_COLORS[a] for a in stats["Axis"]],
        s=style.POINT_SIZE_FOREGROUND,
        zorder=3,
        clip_on=False,
    )
    ax.set_xticks(
        offset + (sizes[stages].to_numpy() - 1) / 2, [STAGE_LABELS[s] for s in stages]
    )
    ax.tick_params(axis="x", length=0)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                color=color,
                label=style.STABILITY_AXIS_LABELS[axis],
            )
            for axis, color in style.STABILITY_AXIS_COLORS.items()
        ],
        loc="lower center",
        ncols=2,
        fontsize=style.FONT_SIZE_DENSE,
        columnspacing=0.8,
    )
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Aggregate FMS")


def makeFigure():
    return common.standalone_fixed(__name__)
