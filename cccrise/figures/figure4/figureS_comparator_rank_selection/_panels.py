"""Each annotation-pooled comparator's own rank selection on one labelling,
ALAD and IBD drawn together, from ``uv run cccrise-comparators`` and the
scores runner's ``fit_ranks`` and ``start_congruence``: pseudobulk CPD by
cccRISE's seeded-start sweep (Figure 2's quantities: fit R2X across starts,
the LR pair mode's collinearity per start, the distinct minima),
Tensor-cell2cell by the Kneedle elbow of its error (with how nearly two of a
start's components duplicate each other, per start, and the minima of its
random starts), and STACCato by each mode's eigenvalue shares above 1%. Each
cohort's chosen rank is marked in its color."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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


def title(ax: plt.Axes, name: str, labelling: str) -> None:
    ax.set_title(f"{name} {labelling}", loc="left")


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


def mark_chosen(ax: plt.Axes, rank: int, dataset: str) -> None:
    ax.axvline(rank, color=DATASET_COLORS[dataset], ls="--", zorder=1)


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


def key(ax: plt.Axes) -> None:
    """The figure's key, inside `ax`."""
    handles = [
        *[
            Line2D([], [], color=DATASET_COLORS[d], label=name)
            for d, name in DATASET_LABELS.items()
        ],
        Line2D([], [], color=style.COLORS["reference"], ls="--", label="Chosen rank"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            fillstyle="none",
            markeredgecolor=common.KNEEDLE_EDGE,
            label="Kneedle",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=style.FONT_SIZE_DENSE,
        handlelength=1.2,
    )


def pb_fit(axs, labelling: str) -> None:
    """Fit R2X by rank: the median over starts, banded by their range, with
    its Kneedle elbow circled."""
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
        elbow = common.kneedle_elbow(median.index, 1 - median.to_numpy())
        circle_kneedle(ax, elbow, median[elbow])
        mark_chosen(ax, fit_rank("pb", dataset, labelling), dataset)
        top = max(top, int(median.index.max()))
    finish_ranks(ax, top)
    ax.set_ylabel("Fit $R^2X$")
    title(ax, "PB", labelling)


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
        mark_chosen(ax, fit_rank("tc2c", dataset, labelling), dataset)
        top = max(top, int(mean.index.max()))
    finish_ranks(ax, top)
    ax.set_ylabel("Normalized error")
    title(ax, "TC2C", labelling)


def _per_start(ax: plt.Axes, t: pd.DataFrame, column: str, method: str, labelling):
    """`column` above rank 1: the median over starts, banded by their range,
    each cohort's chosen rank marked."""
    top = 0
    for dataset, name in DATASET_LABELS.items():
        rows = t[(t["Dataset"] == name) & (t["Rank"] > 1)]
        by_rank = rows.groupby("Rank")[column]
        band(ax, by_rank, by_rank.median(), dataset)
        mark_chosen(ax, fit_rank(method, dataset, labelling), dataset)
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


def tc2c_congruence(axs, labelling: str) -> None:
    """Per rank, the largest congruence product between two of a start's
    components (`rank_selection.max_congruence`). Non-negative factors
    overlap by construction, so this asks whether two components duplicate
    each other in every mode, not whether they cancel, as pseudobulk CPD's
    collinearity does."""

    def congruence(dataset: str) -> pd.DataFrame:
        t = load_scores("start_congruence", labelling, dataset)
        return t[t["Method"] == METHOD_NAMES["tc2c"]].drop(columns="Method")

    t = both_cohorts(congruence)
    common.source_data(
        "tc2c_max_congruence_by_rank_and_start",
        t,
        "Per Tensor-cell2cell rank and random start, the largest congruence "
        "product between two components: the product over the four modes of "
        "their absolute cosine (1 = two components identical up to scale); "
        "plotted as the median over starts, banded by their range.",
    )
    _per_start(axs[0], t, "MaxCongruence", "tc2c", labelling)
    axs[0].set_ylabel("Max congruence product")


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
        mark_chosen(ax, fit_rank(method, dataset, labelling), dataset)
        top = max(top, int(share.index.max()))
    finish_ranks(ax, top)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Starts at most\ncommon minimum")


def pb_minima(axs, labelling: str) -> None:
    _minima(axs, "pb", "rank", labelling)
    if labelling == "coarse":
        key(axs[0])


def tc2c_minima(axs, labelling: str) -> None:
    _minima(axs, "tc2c", "elbow", labelling)


def stac_shares(axs, labelling: str) -> None:
    """STACCato's eigenvalue shares per mode, colored by mode, each cohort's
    line in `STAC_LINES`, its kept core noted (components above the 1%
    line)."""
    ax = axs[0]
    shares = both_cohorts(lambda d: load("stac", "shares", d, labelling))
    common.source_data(
        "staccato_eigenvalue_shares",
        shares,
        "Each mode's eigenvalue shares of its unfolding's Gram matrix, largest "
        "first; STACCato keeps those above 1% per mode.",
    )
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
        line = "solid" if STAC_LINES[dataset] == "-" else "dashed"
        ax.text(
            0.98,
            0.98 - 0.09 * i,
            f"{name} ({line}): core {kept}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=style.FONT_SIZE_DENSE,
        )
    common.reference_line(ax, methods.STACCATO_VAREXP / 100)
    ax.set_yscale("log")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue share")
    title(ax, "STAC", labelling)
    if labelling == "fine":
        ax.legend(loc="lower left", fontsize=style.FONT_SIZE_DENSE, handlelength=1.2)
