"""Which methods call each LR pair significant (`significance`): one row per
pair significant in cccRISE or any comparator (each comparator's coarse and
fine fits pooled), one column per method, a dot where the method calls it
significant (>= `significance.SIGNIFICANT` x its column's top pair), shaded
by that score. Flagged C3 pairs (`significance.C3_FLAG` to `SIGNIFICANT` in
C3) get an open dot in the cccRISE column. Rows are grouped by which methods
call them: all four, cccRISE and some comparators, cccRISE only, comparators
only (most comparators first). Pair names are coloured as in the scatters."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .. import common, style
from . import significance
from ._common import METHODS, SHORT

COLUMNS = ("cccRISE", *(SHORT[m] for m in METHODS))
GAP = 0.7  # rows of space between groups
CMAP, NORM = style.CMAPS["sequential"], Normalize(0, 1)


def rows() -> pd.DataFrame:
    """Pairs significant anywhere, with their scores per method (`COLUMNS`),
    group, colour and row position `y`."""
    table = significance.combined()
    table = table[table["region"].notna()].rename(columns={"x": "cccRISE"})
    calls = table[list(COLUMNS)].ge(significance.SIGNIFICANT)
    n_comparators = calls[list(COLUMNS[1:])].sum(axis=1)
    table = table.assign(
        group=np.select(
            [calls.all(axis=1), table["sx"] & (n_comparators > 0), table["sx"]],
            [0, 1, 2],
            default=3,
        ),
        n_comparators=n_comparators,
        pattern=calls.astype(int).astype(str).agg("".join, axis=1),
        color=significance._colors(table, sns.color_palette("Dark2", 5)),
    )
    table = table.sort_values(
        ["group", "n_comparators", "pattern", "component_x", "cccRISE"],
        ascending=[True, False, False, True, False],
    )
    return table.assign(y=np.arange(len(table)) + GAP * table["group"].to_numpy())


def _dots(ax, table):
    for j, column in enumerate(COLUMNS):
        score = table[column].fillna(0).to_numpy()
        called = score >= significance.SIGNIFICANT
        ax.scatter(
            np.full(called.sum(), j), table["y"][called], s=style.POINT_SIZE_FOREGROUND,
            c=CMAP(NORM(score[called])), edgecolor=style.COLORS["ink"], lw=0.4,
        )  # fmt: skip
        if column == "cccRISE":
            flagged = table["c3_flag"].to_numpy()
            ax.scatter(
                np.full(flagged.sum(), j), table["y"][flagged],
                s=style.POINT_SIZE_FOREGROUND, facecolor="none",
                edgecolor=sns.color_palette("Dark2", 5)[2], lw=0.8,
            )  # fmt: skip


def plot(fig):
    table = rows()
    ax = fig.subplots()
    for y in table["y"]:
        ax.axhline(y, color="0.93", lw=0.5, zorder=0)
    _dots(ax, table)
    ax.set_yticks(table["y"], [p.replace("^", "–") for p in table.index])
    for tick, color in zip(ax.get_yticklabels(), table["color"], strict=True):
        tick.set_color(color)
    ax.tick_params(axis="y", labelsize=style.FONT_SIZE_DENSE - 1, length=0)
    ax.set_xticks(range(len(COLUMNS)), COLUMNS, rotation=40, ha="left")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(COLUMNS) - 0.4)
    ax.set_ylim(table["y"].max() + 0.8, -0.8)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.colorbar(
        ScalarMappable(NORM, CMAP), ax=ax, label="|loading| / column max",
        location="bottom", shrink=0.8, pad=0.02, ticks=[0, 0.4, 1],
    )  # fmt: skip
    ring = Line2D(
        [], [], marker="o", ls="", markersize=4, markerfacecolor="none",
        markeredgecolor=sns.color_palette("Dark2", 5)[2], label="C3, 0.2–0.4",
    )  # fmt: skip
    fig.legend(
        handles=[ring], loc="outside lower center", frameon=False,
        fontsize=style.FONT_SIZE_DENSE, handletextpad=0.2,
    )  # fmt: skip


def makeFigure():
    return common.standalone(plot, (55 * style.MM, 110 * style.MM))
