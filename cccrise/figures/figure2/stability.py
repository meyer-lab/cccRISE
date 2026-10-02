"""Aggregate RISE and CPD stability per resampling axis (the Methods'
stability metric: a replicate's mean FMS over components), as the median with
a 2.5-97.5 percentile band over replicates, from ``uv run cccrise-stability``."""

import numpy as np

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...validation.component_stability import metric_interval
from .. import common, style

DATASET = "alad"


def plot(fig):
    ax = fig.subplots()
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
    x = np.arange(len(stats))
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
    )
    ax.set_xticks(
        x,
        [f"{s}: {a}" for s, a in zip(stats["Stage"], stats["Axis"], strict=True)],
        rotation=30,
        ha="right",
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Aggregate FMS")
    ax.set_title("Component stability")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
