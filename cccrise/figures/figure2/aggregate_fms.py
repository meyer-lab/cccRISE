"""Aggregate RISE and CPD stability (a replicate's mean FMS over components)
under leave-one-patient-out jackknifing and seed refits, ALAD cohort:
the mean ± SD over replicates, each replicate jittered about it."""

import numpy as np
from matplotlib.lines import Line2D

from .. import common, style
from .figureS_alad_cohort import _stability as S

AXES = ("square",)
STAGES = ("RISE", "CPD")  # pipeline order
JITTER = 0.25  # half-width, in slots
REPLICATE_ALPHA = 0.5
ERRORBAR_CAPSIZE = 2  # pt


def plot(axs):
    (ax,) = axs
    aggregate = S.aggregate(S.stability())
    common.source_data(
        "aggregate_fms_by_replicate",
        aggregate.rename(columns={"Stage": "Model", "FMS": "Aggregate FMS"}),
        "Each stability replicate's aggregate FMS (mean factor match score to "
        "the full-cohort fit over components), per model (RISE or CPD) and "
        "stability axis (jackknife: leave one patient out; seed_only: refit "
        "from another seed).",
    )
    # One slot per (stage, axis), a blank slot between stages
    slots = len(S.AGGREGATE_AXES) + 1
    rng = np.random.default_rng(common.JITTER_SEED)
    for i, stage in enumerate(STAGES):
        for j, axis in enumerate(S.AGGREGATE_AXES):
            fms = aggregate.query("Stage == @stage and Axis == @axis")["FMS"]
            _mean_sd(ax, i * slots + j, fms, style.STABILITY_AXIS_COLORS[axis], rng)
    ax.set_xticks(np.arange(len(STAGES)) * slots + (slots - 2) / 2, STAGES)
    ax.set_xlim(-0.75, len(STAGES) * slots - 1.25)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                color=style.STABILITY_AXIS_COLORS[a],
                label=style.STABILITY_AXIS_LABELS[a],
            )
            for a in S.AGGREGATE_AXES
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncols=2,
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Aggregate FMS")


def _mean_sd(ax, x, fms, color, rng) -> None:
    """The mean of `fms` at `x` with ± SD error bars, each value jittered
    behind it."""
    ax.scatter(
        x + rng.uniform(-JITTER, JITTER, len(fms)),
        fms,
        color=color,
        s=style.POINT_SIZE_BACKGROUND,
        alpha=REPLICATE_ALPHA,
        linewidths=0,
        zorder=2,
    )
    mean, sd = fms.mean(), fms.std()
    ax.errorbar(
        x,
        mean,
        yerr=sd,
        fmt="none",
        ecolor=style.COLORS["ink"],
        elinewidth=style.LINE_WIDTH_THIN,
        capsize=ERRORBAR_CAPSIZE,
        zorder=3,
    )
    ax.scatter(
        x,
        mean,
        color=color,
        s=style.POINT_SIZE_FOREGROUND,
        edgecolors=style.COLORS["ink"],
        linewidths=style.LINE_WIDTH_THIN,
        zorder=4,
        clip_on=False,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
