"""Which methods call each LR pair significant (`significance`): one row per
pair significant in cccRISE or any comparator (each comparator's coarse and
fine fits pooled), one column per method, a dot where the method calls it
significant (>= `significance.SIGNIFICANT` x its column's top pair). Dot
colour: that score (|loading| / column max); dot size: the pair's best rank
in any of the method's columns (1 to `significance.TOP_K`, the scatters'
filter). Rows are grouped by which methods call them: all four, cccRISE and
some comparators, cccRISE only, comparators only (most comparators first).
Pair names: the cccRISE component's colour when cccRISE calls them, else
black."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D

from .. import common, style
from . import significance
from ._common import METHODS, SHORT, load

COLUMNS = ("cccRISE", *(SHORT[m] for m in METHODS))
GAP = 0.7  # rows of space between groups
# The sequential map without its near-black end.
CMAP = ListedColormap(style.CMAPS["sequential"](np.linspace(0, 0.72, 256)))
NORM = Normalize(0, 1)
RANK_SIZES = (1, 2, 5, 10)  # ranks shown in the size legend


def _size(rank):
    """Marker area, in proportion to 1 / rank (ranks capped at `TOP_K`), so
    the top few ranks stay apart."""
    return 50 / np.clip(rank, 1, significance.TOP_K)


def _ranks(pairs) -> pd.DataFrame:
    """Each pair's best rank in any column of each method (comparators: either
    fit; cccRISE: its coarse copy, the same fit)."""
    p = load("profiles")
    p = p[
        p["lr_pair"].isin(pairs)
        & ~((p["method"] == "cccRISE") & (p["annotation"] != "coarse"))
    ]
    best = p.groupby(["method", "lr_pair"])["lr_rank"].min().unstack(0)
    return best.rename(columns=SHORT).reindex(index=pairs, columns=list(COLUMNS))


def rows() -> pd.DataFrame:
    """Pairs significant anywhere, with their scores per method (`COLUMNS`),
    best ranks (`rank_<method>`), group, colour and row position `y`."""
    table = significance.combined()
    table = table[table["region"].notna()].rename(columns={"x": "cccRISE"})
    calls = table[list(COLUMNS)].ge(significance.SIGNIFICANT)
    n_comparators = calls[list(COLUMNS[1:])].sum(axis=1)
    palette = sns.color_palette("Dark2", 5)
    table = table.assign(
        group=np.select(
            [calls.all(axis=1), table["sx"] & (n_comparators > 0), table["sx"]],
            [0, 1, 2],
            default=3,
        ),
        n_comparators=n_comparators,
        pattern=calls.astype(int).astype(str).agg("".join, axis=1),
        color=[
            palette[c - 1] if sx else "black"
            for c, sx in zip(table["component_x"], table["sx"], strict=True)
        ],
    ).join(_ranks(table.index).add_prefix("rank_"))
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
            np.full(called.sum(), j), table["y"][called],
            s=_size(table[f"rank_{column}"].to_numpy()[called]),
            c=CMAP(NORM(score[called])), edgecolor=style.COLORS["ink"], lw=0.4,
            zorder=2,
        )  # fmt: skip


def _legends(fig):
    fig.colorbar(
        ScalarMappable(NORM, CMAP), ax=fig.axes, label="|loading| / column max",
        location="bottom", shrink=0.8, pad=0.02, ticks=[0, 0.4, 1],
    )  # fmt: skip
    handles = [
        Line2D(
            [], [], marker="o", ls="", markersize=np.sqrt(_size(r)), color="0.6",
            markeredgecolor=style.COLORS["ink"], markeredgewidth=0.4,
            label=f"{r}",
        )
        for r in RANK_SIZES
    ]  # fmt: skip
    fig.legend(
        handles=handles, title="best rank", loc="outside lower center",
        ncol=len(handles), frameon=False, fontsize=style.FONT_SIZE_DENSE,
        title_fontsize=style.FONT_SIZE_DENSE, handletextpad=0.1,
        columnspacing=0.8,
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
    _legends(fig)


def makeFigure():
    return common.standalone(plot, (55 * style.MM, 115 * style.MM))
