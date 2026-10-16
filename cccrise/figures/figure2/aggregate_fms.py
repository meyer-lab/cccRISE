"""Aggregate RISE and CP stability per stability axis (the Methods'
stability metric: a replicate's mean FMS over components), as the median with
a 2.5-97.5 percentile band over replicates, each replicate's aggregate FMS
jittered beneath, from ``uv run cccrise-stability``."""

import numpy as np
from matplotlib.lines import Line2D

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...validation.stability import metric_interval
from .. import common, style

DATASET = "alad"
STAGE_ORDER = ["RISE", "CPD"]  # pipeline order
STAGE_LABELS = {"RISE": "RISE", "CPD": "CP"}
JITTER = 0.25
REPLICATE_ALPHA = 0.5
Y_MIN = 0.45  # below every replicate, with room for the legend
Y_TICK_MIN = 0.55
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    stability = common.load_runner_output(
        DATASETS[DATASET].stability_filename(), VALIDATION_DIR
    )
    stability = stability[stability["Axis"].isin(style.STABILITY_AXIS_COLORS)]
    stats = metric_interval(
        stability,
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
    _plot_replicates(ax, stability, dict(zip(zip(stats["Stage"], stats["Axis"]), x)))
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
    # Two columns, filled column-wise: colors on the top row, sizes below
    colors = [
        Line2D([], [], ls="", marker="o", color=color, label=label)
        for color, label in (
            (style.STABILITY_AXIS_COLORS[a], style.STABILITY_AXIS_LABELS[a])
            for a in style.STABILITY_AXIS_COLORS
        )
    ]
    sizes = [
        ax.scatter([], [], s=size, color=style.COLORS["ink"], label=label)
        for size, label in (
            (style.POINT_SIZE_FOREGROUND, "Aggregate"),
            (style.POINT_SIZE_BACKGROUND, "Replicate"),
        )
    ]
    ax.legend(
        handles=[h for pair in zip(colors, sizes) for h in pair],
        loc="lower center",
        ncols=2,
        fontsize=style.FONT_SIZE_DENSE,
        columnspacing=0.8,
    )
    ax.set_ylim(Y_MIN, 1.02)
    ax.set_yticks(np.arange(Y_TICK_MIN, 1.01, 0.2))
    ax.set_ylabel("Aggregate FMS")


def _plot_replicates(ax, stability, positions: dict) -> None:
    """Each replicate's aggregate FMS (its mean over components), jittered
    about its (stage, axis) position."""
    per_replicate = stability.groupby(["Stage", "Axis", "Replicate"])["FMS"].mean()
    rng = np.random.default_rng(0)
    for (stage, axis), fms in per_replicate.groupby(["Stage", "Axis"]):
        ax.scatter(
            positions[stage, axis] + rng.uniform(-JITTER, JITTER, len(fms)),
            fms,
            c=style.STABILITY_AXIS_COLORS[axis],
            s=style.POINT_SIZE_BACKGROUND,
            alpha=REPLICATE_ALPHA,
            linewidths=0,
            zorder=2,
        )


def makeFigure():
    return common.standalone_fixed(__name__)
