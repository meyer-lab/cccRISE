"""
Figure for per-component RISE and CPD stability at fixed ranks.

Plots the resampling replicates produced by ``uv run cccrise-stability``; run
that first. Panels a and b are the primary reading -- the manuscript's
conclusions rest on a handful of named components, not on a mean -- with the
aggregate FMS the Methods define in panel c and the match-ambiguity diagnostic
in panel d.

    uv run cccrise-stability --dataset alad --rise-rank 25 --cp-rank 4 \
        --n-replicates 20 --axes cells,conditions,lr_pairs,permuted
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..stability.component_stability import (
    PERMUTED_AXIS,
    STAGE_CPD,
    STAGE_RISE,
    metric_interval,
    summarize_component_stability,
)
from .common import getSetup, load_runner_csv, subplotLabel

# Change this to switch dataset; must match the runner's --dataset. The ranks
# the run was conditioned on are recorded in the sweep's sidecar.
SELECTED_DATASET = "alad"
STABILITY_CSV = f"component_stability_{SELECTED_DATASET}.csv"

AXIS_COLORS = {
    "cells": "tab:blue",
    "conditions": "tab:orange",
    "lr_pairs": "tab:green",
}
BAND = (2.5, 97.5)


def plot_per_component_fms(
    df: pd.DataFrame,
    ax: plt.Axes,
    stage: str,
    title: str | None = None,
) -> None:
    """Per-component FMS distribution across replicates, one box per axis.

    A component whose box sits high and tight was recovered the same way in
    every replicate; one whose box is low or long was not. The permuted axis,
    when present, is drawn as a horizontal chance floor rather than as a box,
    and one line is drawn per set of compared modes: an FMS is a product over
    however many modes were comparable, so a floor is only valid against rows
    that compared the same ones.

    Parameters
    ----------
    df : pd.DataFrame
        Long stability table from `cccrise-stability`.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    stage : str
        'RISE' or 'CPD'; see `stability.STAGE_RISE` / `STAGE_CPD`.
    title : str | None
        Optional title for the subplot.
    """
    stage_rows = df[df["Stage"] == stage]
    observed = stage_rows[stage_rows["Axis"] != PERMUTED_AXIS]
    axes_present = [a for a in AXIS_COLORS if a in set(observed["Axis"])]

    sns.boxplot(
        data=observed,
        x="Component",
        y="FMS",
        hue="Axis",
        hue_order=axes_present,
        palette={a: AXIS_COLORS[a] for a in axes_present},
        fliersize=1,
        linewidth=0.5,
        ax=ax,
    )

    floor = stage_rows[stage_rows["Axis"] == PERMUTED_AXIS]
    if not floor.empty:
        ax.axhline(
            floor["FMS"].quantile(0.95),
            color="grey",
            linestyle="--",
            linewidth=0.8,
            label="chance floor P95",
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
    about what a replicate contributed.
    """
    stats = metric_interval(
        df[df["Axis"] != PERMUTED_AXIS],
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

    df = load_runner_csv(STABILITY_CSV)
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
