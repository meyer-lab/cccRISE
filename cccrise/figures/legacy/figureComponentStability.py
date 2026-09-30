"""
Figure for per-component RISE and CPD stability at fixed ranks.

Plots the resampling replicates produced by ``uv run cccrise-stability``; run
that first. Panels a and b are the primary reading -- the manuscript's
conclusions rest on a handful of named components, not on a mean -- with the
aggregate FMS the Methods define in panel c and the match-ambiguity diagnostic
in panel d.

    uv run cccrise-stability --dataset alad --rise-rank 17 --cp-rank 5 \
        --n-replicates 100
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ...runners.datasets import VALIDATION_DIR
from ...validation.component_stability import (
    SEED_AXIS,
    STAGE_CPD,
    STAGE_RISE,
    metric_interval,
    summarize_component_stability,
)
from ..common import getSetup, load_runner_output, subplotLabel

# Change this to switch dataset; must match the runner's --dataset. The ranks
# the run was conditioned on are recorded in the sweep's sidecar.
SELECTED_DATASET = "alad"
STABILITY_FILE = f"component_stability_{SELECTED_DATASET}.parquet"

#: The axes that perturb the data. `SEED_AXIS` is deliberately kept out: it is
#: the baseline the others are read against rather than one of them, and
#: `summarize_component_stability` emits no rows for it, so the panels that
#: read the summary iterate this dict and get the right set for free.
AXIS_COLORS = {
    "cells": "tab:blue",
    "conditions": "tab:orange",
    "lr_pairs": "tab:green",
}
#: Grey, and drawn first in each group, so the ceiling reads as a reference
#: rather than as a fourth result.
SEED_COLOR = "0.45"
PLOT_AXES = {SEED_AXIS: SEED_COLOR, **AXIS_COLORS}
BAND = (2.5, 97.5)


def plot_per_component_fms(
    df: pd.DataFrame,
    ax: plt.Axes,
    stage: str,
    title: str | None = None,
) -> None:
    """Per-component FMS distribution across replicates, one box per axis.

    A component whose box sits high and tight was recovered the same way in
    every replicate; one whose box is low or long was not. The grey box is the
    seed-refit ceiling for that same component -- identical data, a different
    starting point -- so a coloured box reaching it means the perturbation
    cost that component nothing beyond what reseeding costs it anyway, and the
    gap below it is the part attributable to the data. Plotting the ceiling as
    a per-component box rather than as one horizontal line is what the seed
    axis makes possible: it leaves the roster alone, so its components carry
    the reference's own numbering.

    Axes are selected by whitelist, so rows from a run predating the seed axis
    (which carried a permuted chance floor on a different scale) are dropped
    rather than drawn as an unlabelled fourth hue.

    Parameters
    ----------
    df : pd.DataFrame
        Long stability table from `cccrise-stability`.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    stage : str
        'RISE' or 'CPD'; see `validation.STAGE_RISE` / `STAGE_CPD`.
    title : str | None
        Optional title for the subplot.
    """
    stage_rows = df[df["Stage"] == stage]
    axes_present = [a for a in PLOT_AXES if a in set(stage_rows["Axis"])]

    sns.boxplot(
        data=stage_rows[stage_rows["Axis"].isin(axes_present)],
        x="Component",
        y="FMS",
        hue="Axis",
        hue_order=axes_present,
        palette={a: PLOT_AXES[a] for a in axes_present},
        fliersize=1,
        linewidth=0.5,
        ax=ax,
    )

    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{stage} component (reference numbering)")
    ax.set_ylabel("Factor match score")
    ax.legend(frameon=True, fontsize=5)
    if title:
        ax.set_title(title)


def plot_aggregate_fms(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str | None = None,
) -> None:
    """Aggregate FMS per stage and axis, as a median with a percentile band.

    This is the number the Methods define. It is the mean of the per-component
    scores within a replicate, so panels a/b and this panel cannot disagree
    about what a replicate contributed. The seed axis is included: how close
    each resampling axis sits to the reproducibility ceiling is exactly what
    this panel is for.
    """
    stats = metric_interval(
        df[df["Axis"].isin(PLOT_AXES)],
        "FMS",
        by=("Stage", "Axis"),
        lower=BAND[0],
        upper=BAND[1],
    )
    labels = [
        f"{stage}\n{axis}"
        for stage, axis in zip(stats["Stage"], stats["Axis"], strict=True)
    ]
    positions = range(len(stats))
    ax.errorbar(
        positions,
        stats["Median"],
        yerr=[
            stats["Median"] - stats[f"P{BAND[0]:g}"],
            stats[f"P{BAND[1]:g}"] - stats["Median"],
        ],
        fmt="o",
        capsize=3,
        color="black",
        markersize=4,
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(f"Aggregate FMS (median, {BAND[0]}-{BAND[1]}%)")
    if title:
        ax.set_title(title)


def plot_match_ambiguity(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str | None = None,
) -> None:
    """Median FMS against the median runner-up score, per component.

    Separates two failures a single FMS conflates. A point well above the
    diagonal is a component that was re-identified unambiguously, however far
    it moved; a point near the diagonal is one the replicate could not tell
    apart from another component, so its FMS says more about CP
    identifiability than about the data.
    """
    summary = summarize_component_stability(df)

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    for stage, marker in ((STAGE_RISE, "o"), (STAGE_CPD, "s")):
        for axis, color in AXIS_COLORS.items():
            rows = summary[(summary["Stage"] == stage) & (summary["Axis"] == axis)]
            if rows.empty:
                continue
            ax.scatter(
                rows["MedianRunnerUpFMS"],
                rows["MedianFMS"],
                marker=marker,
                color=color,
                s=14,
                alpha=0.8,
                label=f"{stage} / {axis}",
            )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Median runner-up FMS")
    ax.set_ylabel("Median matched FMS")
    ax.legend(frameon=True, fontsize=5)
    if title:
        ax.set_title(title)


def makeFigure():
    ax, f = getSetup((16, 3.6), (1, 4))
    subplotLabel(ax)

    df = load_runner_output(STABILITY_FILE, VALIDATION_DIR)
    label = SELECTED_DATASET.upper()

    plot_per_component_fms(
        df, ax[0], STAGE_RISE, title=f"RISE component stability ({label})"
    )
    plot_per_component_fms(
        df, ax[1], STAGE_CPD, title=f"CPD component stability ({label})"
    )
    plot_aggregate_fms(df, ax[2], title=f"Aggregate FMS ({label})")
    plot_match_ambiguity(df, ax[3], title=f"Match ambiguity ({label})")

    print(summarize_component_stability(df).to_string(index=False))
    return f
