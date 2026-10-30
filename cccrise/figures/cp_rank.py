"""CP rank-selection panels drawn from ``uv run cccrise-cpdrank``'s seeded
starts (ALAD Figure 2, IBD Figure 5's supplement): fit R2X across starts, each
mode's collinearity, and the distinct minima the starts reach."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

from ..rank_selection import distinct_minima
from ..runners.datasets import DATASETS, RANK_SELECTION_DIR
from . import common, style

MODES = {"A": "Condition", "B": "Sender", "C": "Receiver", "D": "LR pair"}
#: Color slots for distinct minima: slot 0 is f_max, the minimum most starts
#: reach. Sized for the most either dataset needs (ALAD rank 17: eight
#: minima, shifted one slot by a tie), so a slot is one color in every figure.
MINIMA_SLOTS = 9
FMAX = r"$f_{\max}$"
FMAX_EDGE = "#e34948"
KNEEDLE_EDGE = FMAX_EDGE
TOP_STRIP = 0.28  # fraction of the minima panel's height the starts strip takes


def load_minima(dataset: str) -> pd.DataFrame:
    """`rank_selection.distinct_minima` of `dataset`'s starts, with each
    minimum's color 'Slot' (`Order` - 1, shifted one where there is no
    f_max)."""
    start_fms = common.load_runner_output(
        DATASETS[dataset].cpd_start_fms_filename(), RANK_SELECTION_DIR
    )
    r2x = common.load_cpd_rank(dataset).set_index(["Rank", "Seed"])["FitR2X"]
    minima = distinct_minima(start_fms, r2x)
    return minima.assign(Slot=minima["Order"] - 1 + minima["Tied"].astype(int))


def rank_axis(ax: plt.Axes, ranks) -> None:
    ranks = np.asarray(ranks)
    ax.set_xticks(ranks[ranks % 2 == 1] if ranks.max() > 10 else ranks)
    ax.set_xlim(0.4, ranks.max() + 0.6)
    ax.set_xlabel("CP rank")


def kneedle_elbow(x, y) -> int:
    """Elbow of a convex, decreasing curve by offline Kneedle (Satopaa et al.
    2011, sensitivity 1, as `kneed.KneeLocator(curve="convex",
    direction="decreasing")`)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xn, yn = ((v - v.min()) / (v.max() - v.min()) for v in (x, y))
    diff = (yn.max() - yn) - xn
    maxima = [
        i
        for i in range(1, x.size - 1)
        if diff[i] >= diff[i - 1] and diff[i] >= diff[i + 1]
    ]
    step = float(np.abs(np.diff(xn)).mean())
    for i, m in enumerate(maxima):
        stop = maxima[i + 1] if i + 1 < len(maxima) else x.size
        if np.any(diff[m + 1 : stop] < diff[m] - step):
            return int(x[m])
    return int(x[int(np.argmax(diff))])


def plot_r2x_starts(ax: plt.Axes, dataset: str, rank: int | None = None) -> None:
    """Fit R2X against CP rank: the median over starts, banded by their
    range, with the Kneedle elbow of the median curve circled."""
    sweep = common.load_cpd_rank(dataset)
    common.source_data(
        "cp_fit_r2x_by_rank_and_start",
        sweep[["Rank", "Start", "Seed", "FitR2X"]],
        "Fit R2X of every seeded SVD start at every CP rank (start 0 is the "
        "production fit); plotted as the median over starts, banded by their "
        "range.",
    )
    by_rank = sweep.groupby("Rank")["FitR2X"]
    median = by_rank.median()
    color = style.COLORS["fit"]
    ax.fill_between(
        median.index, by_rank.min(), by_rank.max(), color=color, alpha=0.2, lw=0
    )
    ax.plot(median.index, median, color=color, label="Median start")
    elbow = kneedle_elbow(median.index, 1 - median.to_numpy())
    ax.plot(
        elbow,
        median[elbow],
        marker="o",
        markersize=3 * style.MARKER_SIZE,
        fillstyle="none",
        markeredgecolor=KNEEDLE_EDGE,
        markeredgewidth=style.LINE_WIDTH,
        zorder=4,
    )
    ax.annotate(
        "Kneedle",
        (elbow, median[elbow]),
        xytext=(6, -10),
        textcoords="offset points",
        fontsize=style.FONT_SIZE_DENSE,
    )
    common.mark_rank(ax, rank)
    rank_axis(ax, median.index)
    ax.set_ylabel("Fit $R^2X$")


def plot_mode_collinearity(
    ax: plt.Axes, dataset: str, rank: int | None = None, part: int | None = None
) -> None:
    """Per mode, the largest |cosine| between two components: the median
    over starts, banded by their range, with `rank` marked. `part` is
    `common.source_data`'s."""
    sweep = common.load_cpd_rank(dataset)
    columns = [f"MaxCos{m}" for m in MODES]
    common.source_data(
        "cp_max_cosine_by_rank",
        sweep[["Rank", "Start", "Seed", *columns]],
        "Per CP rank and seeded SVD start, each mode's largest absolute cosine "
        "between two components (MaxCosA condition, B sender, C receiver, "
        "D ligand-receptor pair).",
        part=part,
    )
    sweep = sweep[sweep["Rank"] > 1]
    for color, column, label in zip(
        style.CATEGORICAL, columns, MODES.values(), strict=False
    ):
        by_rank = sweep.groupby("Rank")[column]
        median = by_rank.median()
        ax.fill_between(
            median.index, by_rank.min(), by_rank.max(), color=color, alpha=0.15, lw=0
        )
        ax.plot(
            median.index,
            median,
            color=color,
            label=label,
            marker="o",
            markersize=style.MARKER_SIZE,
        )
    common.mark_rank(ax, rank)
    rank_axis(ax, np.arange(1, sweep["Rank"].max() + 1))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Max |cosine|")
    ax.legend(loc="lower right", fontsize=style.FONT_SIZE_DENSE)


def minima_palette() -> list[str]:
    """f_max blue, then an orange -> purple ramp for the other minima."""
    ramp = plt.get_cmap("plasma")(np.linspace(0.78, 0.08, MINIMA_SLOTS - 1))
    return [style.CATEGORICAL[0], *(to_hex(c) for c in ramp)]


def plot_minima(ax: plt.Axes, dataset: str) -> None:
    """`ax` split into two strips sharing the rank axis: the starts reaching
    each distinct minimum (top) and its FMS to f_max (bottom)."""
    table = load_minima(dataset)
    common.source_data(
        "cp_minima_by_rank",
        table,
        "Distinct minima per CP rank (starts joined at FMS >= 0.95): the "
        "starts reaching each, its best R2X, its FMS to the minimum most "
        "starts reach (f_max; where two minima tie for most, 'Tied', to the "
        "one with the higher R2X), its order by starts, and its color slot.",
    )
    palette = minima_palette()
    colors = [palette[s] for s in table["Slot"]]
    ax.axis("off")
    top = ax.inset_axes((0, 1 - TOP_STRIP, 1, TOP_STRIP))
    bottom = ax.inset_axes((0, 0, 1, 1 - TOP_STRIP), sharex=top)
    for _, g in table.groupby("Rank"):
        top.bar(
            g["Rank"],
            g["Starts"],
            bottom=g["Starts"].cumsum() - g["Starts"],
            width=0.72,
            color=[colors[i] for i in g.index],
            edgecolor="white",
            lw=0.6,
        )
    last = table.iloc[::-1]  # f_max drawn on top
    bottom.scatter(
        last["Rank"],
        last["FMS"],
        c=[colors[i] for i in last.index],
        s=style.POINT_SIZE,
        lw=0.4,
        edgecolor="white",
        zorder=3,
    )
    n_starts = int(table.groupby("Rank")["Starts"].sum().max())
    top.set_ylim(0, 1.06 * n_starts)
    top.set_yticks([n_starts // 2, n_starts])
    top.set_ylabel("Starts")
    top.tick_params(labelbottom=False, bottom=False)
    rank_axis(bottom, table["Rank"].unique())
    bottom.set_ylim(min(0.6, table["FMS"].min() - 0.04), 1.03)
    bottom.set_ylabel(f"FMS to {FMAX}")
    _minima_key(bottom, palette)


def _minima_key(ax: plt.Axes, palette: list[str]) -> None:
    """One swatch per color slot, f_max at the bottom in a red box, labeled
    "minima" along its left."""
    key = ax.inset_axes((0.1, 0.05, 0.05, 0.62))
    for i, color in enumerate(palette):
        key.add_patch(
            Rectangle((0, i), 1, 1, facecolor=color, edgecolor="white", lw=0.4)
        )
    key.add_patch(
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=FMAX_EDGE, lw=0.8, zorder=3)
    )
    key.set_xlim(0, 1)
    key.set_ylim(0, len(palette))
    key.axis("off")
    text: dict[str, Any] = {
        "fontsize": style.FONT_SIZE_DENSE,
        "color": style.COLORS["ink"],
    }
    key.text(1.5, 0.5, FMAX, ha="left", va="center", **text)
    key.text(
        -0.4, len(palette) / 2, "minima", rotation=90, ha="right", va="center", **text
    )
