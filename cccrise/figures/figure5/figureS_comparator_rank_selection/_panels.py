"""Each annotation-pooled comparator's own rank selection on one labelling,
ALAD and IBD drawn together, from ``uv run cccrise-comparators`` and the
scores runner's ``fit_ranks``: pseudobulk CPD by
cccRISE's seeded-start sweep (Figure 2's quantities: fit R2X across starts,
the LR pair mode's collinearity per start, the distinct minima),
Tensor-cell2cell by the Kneedle elbow of its error, and STACCato by each mode's eigenvalue shares above 1%. Each
cohort's chosen rank is marked in its color."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.typing import LegendLocType

from ....comparators import methods
from ....rank_selection import distinct_minima
from ....runners.datasets import COMPARATOR_RANKS_DIR, comparator_filename
from ... import common, style
from .._common import DATASET_COLORS, DATASET_LABELS
from .._common import load as load_scores

HINT = "    uv run cccrise-comparators --dataset <dataset>"
METHOD_NAMES = {"pb": "PB", "tc2c": "TC2C"}
STAC_MODES = {
    "sample": "Sample",
    "lr": "LR pair",
    "sender": "Sender",
    "receiver": "Receiver",
}
#: Leading eigenvalue shares drawn per STACCato mode.
STAC_COMPONENTS = 30
#: A dash of a line two cohorts share (in line widths)
SHARED_DASH = 3
#: The STACCato panels' y floor: room under the curves for the modes' key
STAC_FLOOR = 6e-5
#: Each cohort's line in the STACCato panels, where color is the mode.
STAC_LINES = {"alad": "-", "ibd": "--"}


def load(method: str, what: str, dataset: str, labelling: str) -> pd.DataFrame:
    return common.load_runner_output(
        comparator_filename(method, what, dataset, labelling),
        COMPARATOR_RANKS_DIR,
        HINT,
    )


def fit_rank(method: str, dataset: str, labelling: str) -> int:
    """The rank `method`'s fit on `labelling` was written at."""
    ranks = load_scores("fit_ranks", labelling, dataset)
    return int(ranks.loc[ranks["Method"] == METHOD_NAMES[method], "Rank"].iloc[0])


def minima(method: str, sweep: str, dataset: str, labelling: str) -> pd.DataFrame:
    """`rank_selection.distinct_minima` of `method`'s seeded starts (its
    `sweep` table for each start's R2X)."""
    starts = load(method, sweep, dataset, labelling)
    return distinct_minima(
        load(method, "start_fms", dataset, labelling),
        starts.set_index(["Rank", "Seed"])["FitR2X"],
    )


def rank_ticks(ax: plt.Axes) -> None:
    """Thin a `common.rank_axis` axis's ticks: every rank up to 10, every
    second up to 17, then every fourth (odd ranks crowd a square axis past
    17)."""
    max_rank = round(ax.get_xlim()[1] - 0.6)
    step = 1 if max_rank <= 10 else 2 if max_rank <= 17 else 4
    ax.set_xticks(np.arange(1, max_rank + 1, step))


def both_cohorts(table: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    """`table` of each cohort, with its name as ``Dataset``."""
    return pd.concat(
        [table(d).assign(Dataset=name) for d, name in DATASET_LABELS.items()],
        ignore_index=True,
    )


def band(ax: plt.Axes, by_rank, center: pd.Series, dataset: str) -> None:
    """`center` by rank in `dataset`'s color, banded by `by_rank`'s range."""
    color = DATASET_COLORS[dataset]
    ax.fill_between(
        center.index, by_rank.min(), by_rank.max(), color=color, alpha=0.25, lw=0
    )
    ax.plot(center.index, center, color=color)


def mark_chosen(ax: plt.Axes, method: str, dataset: str, labelling: str) -> None:
    """`dataset`'s chosen rank, dashed in its color; where the cohorts chose
    the same rank, their dashes alternate along one line so both show."""
    ranks = {d: fit_rank(method, d, labelling) for d in DATASET_LABELS}
    shared = [d for d, r in ranks.items() if r == ranks[dataset]]
    ls = "--"
    if len(shared) > 1:  # dash, gap, the other cohort's dash, gap, ...
        period = 2 * SHARED_DASH * len(shared)
        phase = period - 2 * SHARED_DASH * shared.index(dataset)
        ls = (phase, (SHARED_DASH, period - SHARED_DASH))
    ax.axvline(ranks[dataset], color=DATASET_COLORS[dataset], ls=ls, zorder=1)


def circle_kneedle(ax: plt.Axes, rank: int, value: float) -> None:
    ax.plot(
        rank,
        value,
        marker="o",
        markersize=3 * style.MARKER_SIZE,
        fillstyle="none",
        markeredgecolor=common.KNEEDLE_EDGE,
        markeredgewidth=style.LINE_WIDTH,
        zorder=4,
    )


def finish_ranks(ax: plt.Axes, top: int) -> None:
    common.rank_axis(ax, np.arange(1, top + 1))
    rank_ticks(ax)


def _key(ax: plt.Axes, handles: list, loc: LegendLocType, ncol: int = 1) -> Legend:
    return ax.legend(
        handles=handles,
        loc=loc,
        ncol=ncol,
        fontsize=style.FONT_SIZE_DENSE,
        handlelength=1.2,
        columnspacing=0.6,
        borderaxespad=0.3,
    )


def cohort_key(ax: plt.Axes, loc: LegendLocType) -> None:
    """`mark_chosen`'s dashes, heading the cohorts' colors, on a half-opaque key
    inside `ax` at `loc`."""
    chosen = Line2D(
        [], [], color=style.COLORS["reference"], ls="--", label="Chosen\nrank"
    )
    key = _key(
        ax,
        [chosen]
        + [
            Line2D([], [], color=DATASET_COLORS[d], label=name)
            for d, name in DATASET_LABELS.items()
        ],
        loc,
    )
    # Half-opaque over the chosen ranks' dashes
    key.set_frame_on(True)
    key.get_frame().set(facecolor="white", edgecolor="none", alpha=0.5)


def kneedle_key(ax: plt.Axes, loc: LegendLocType) -> None:
    """The Kneedle elbow's circle, inside `ax` at `loc`."""
    _key(
        ax,
        [
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                fillstyle="none",
                markeredgecolor=common.KNEEDLE_EDGE,
                label="Kneedle",
            )
        ],
        loc,
    )


def pb_fit(axs, labelling: str) -> None:
    """Fit R2X by rank: the median over starts, banded by their range."""
    ax = axs[0]
    sweeps = both_cohorts(lambda d: load("pb", "rank", d, labelling))
    common.source_data(
        "pb_fit_r2x_by_rank_and_start",
        sweeps[["Dataset", "Rank", "Start", "Seed", "FitR2X"]],
        "Fit R2X of every seeded SVD start at every pseudobulk CPD rank; plotted "
        "as the median over starts, banded by their range.",
    )
    top = 0
    for dataset, name in DATASET_LABELS.items():
        by_rank = sweeps[sweeps["Dataset"] == name].groupby("Rank")["FitR2X"]
        median = by_rank.median()
        band(ax, by_rank, median, dataset)
        mark_chosen(ax, "pb", dataset, labelling)
        top = max(top, int(median.index.max()))
    finish_ranks(ax, top)
    ax.set_ylabel("Fit $R^2X$")
    if labelling == "coarse":
        cohort_key(ax, "lower right")


def tc2c_elbow(axs, labelling: str) -> None:
    """Normalized error by rank: the mean over random starts, banded by
    their range, with its Kneedle elbow circled."""
    ax = axs[0]
    elbows = both_cohorts(lambda d: load("tc2c", "elbow", d, labelling))
    common.source_data(
        "tc2c_error_by_rank_and_run",
        elbows,
        "Normalized error (observed entries) of every Tensor-cell2cell run at "
        "every rank; plotted as the mean over runs, banded by their range.",
    )
    top = 0
    for dataset, name in DATASET_LABELS.items():
        elbow = elbows[elbows["Dataset"] == name]
        by_rank = elbow.groupby("Rank")["Error"]
        mean = by_rank.mean()
        band(ax, by_rank, mean, dataset)
        rank = methods.tensor_cell2cell_rank(elbow)
        if rank is not None:
            circle_kneedle(ax, rank, mean[rank])
        mark_chosen(ax, "tc2c", dataset, labelling)
        top = max(top, int(mean.index.max()))
    finish_ranks(ax, top)
    ax.set_ylabel("Normalized error")
    if labelling == "coarse":
        kneedle_key(ax, "upper right")


def _per_start(ax: plt.Axes, t: pd.DataFrame, column: str, method: str, labelling):
    """`column` above rank 1: the median over starts, banded by their range,
    each cohort's chosen rank marked."""
    top = 0
    for dataset, name in DATASET_LABELS.items():
        rows = t[(t["Dataset"] == name) & (t["Rank"] > 1)]
        by_rank = rows.groupby("Rank")[column]
        band(ax, by_rank, by_rank.median(), dataset)
        mark_chosen(ax, method, dataset, labelling)
        top = max(top, int(rows["Rank"].max()))
    finish_ranks(ax, top)
    ax.set_ylim(0, 1.05)


def pb_collinearity(axs, labelling: str) -> None:
    """The LR pair mode's largest |cosine| between two components per start,
    as Figure 2C."""
    sweeps = both_cohorts(lambda d: load("pb", "rank", d, labelling))
    common.source_data(
        "pb_lr_max_cosine_by_rank_and_start",
        sweeps[["Dataset", "Rank", "Start", "Seed", "MaxCosD"]],
        "Per pseudobulk CPD rank and seeded SVD start, the largest absolute "
        "cosine between two components' ligand-receptor pair factors; plotted "
        "as the median over starts, banded by their range.",
    )
    _per_start(axs[0], sweeps, "MaxCosD", "pb", labelling)
    axs[0].set_ylabel("LR pair max |cosine|")


def _minima(axs, method: str, sweep: str, labelling: str) -> None:
    """The share of starts reaching each rank's most common minimum, each
    cohort's chosen rank marked."""
    ax = axs[0]
    t = both_cohorts(lambda d: minima(method, sweep, d, labelling))
    common.source_data(
        f"{method}_minima_by_rank",
        t,
        "Distinct minima per rank (starts joined at FMS >= 0.95): the starts "
        "reaching each, its best R2X, its FMS to the minimum most starts reach, "
        "its order by starts (1 = most; ties: higher R2X), and the seeds of "
        "the starts reaching it; plotted as the share of starts at order 1.",
    )
    top = 0
    for dataset, name in DATASET_LABELS.items():
        rows = t[t["Dataset"] == name]
        share = (
            rows[rows["Order"] == 1].groupby("Rank")["Starts"].max()
            / rows.groupby("Rank")["Starts"].sum()
        )
        color = DATASET_COLORS[dataset]
        ax.plot(
            share.index,
            share,
            color=color,
            marker="o",
            markersize=style.MARKER_SIZE / 2,
        )
        mark_chosen(ax, method, dataset, labelling)
        top = max(top, int(share.index.max()))
    finish_ranks(ax, top)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Starts at most\ncommon minimum")


def pb_minima(axs, labelling: str) -> None:
    _minima(axs, "pb", "rank", labelling)


def stac_shares(axs, labelling: str) -> None:
    """STACCato's eigenvalue shares per mode, colored by mode, each cohort's
    line in `STAC_LINES`, keyed with its kept core (components above the 1%
    line); the fine panel keys the modes."""
    ax = axs[0]
    shares = both_cohorts(lambda d: load("stac", "shares", d, labelling))
    common.source_data(
        "staccato_eigenvalue_shares",
        shares,
        "Each mode's eigenvalue shares of its unfolding's Gram matrix, largest "
        "first; STACCato keeps those above 1% per mode.",
    )
    cores = []
    for i, (dataset, name) in enumerate(DATASET_LABELS.items()):
        rows = shares[shares["Dataset"] == name].drop(columns="Dataset")
        core = methods.staccato_core_shape(rows)
        for color, (mode, label) in zip(
            style.CATEGORICAL, STAC_MODES.items(), strict=False
        ):
            drawn = rows[
                (rows["Mode"] == mode) & (rows["Component"] <= STAC_COMPONENTS)
            ]
            ax.plot(
                drawn["Component"],
                drawn["Share"],
                color=color,
                ls=STAC_LINES[dataset],
                label=label if i == 0 else None,
            )
        kept = "×".join(str(core[m]) for m in STAC_MODES)
        cores.append(
            Line2D([], [], color="0.3", ls=STAC_LINES[dataset], label=f"{name} {kept}")
        )
    cutoff = methods.STACCATO_VAREXP / 100
    common.reference_line(ax, cutoff)
    ax.text(
        STAC_COMPONENTS,
        cutoff,
        f"{methods.STACCATO_VAREXP:g}%",
        ha="right",
        va="bottom",
        fontsize=style.FONT_SIZE_DENSE,
        color=style.COLORS["reference"],
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=STAC_FLOOR)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue share")
    modes = ax.get_legend_handles_labels()[0] if labelling == "fine" else []
    _key(ax, cores, "upper right")
    if modes:
        ax.add_artist(ax.get_legend())
        _key(ax, modes, "lower left", ncol=2)  # two rows clear the curves
