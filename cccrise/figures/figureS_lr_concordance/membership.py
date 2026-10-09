"""Which methods call each LR pair significant (`significance`): one row per
pair significant in cccRISE or any comparator (each comparator's coarse and
fine fits pooled; >= `significance.SIGNIFICANT` x its column's top pair),
one column per method, a dot where the pair is in the top
`significance.TOP_K` of any of the method's columns (the scatters' filter)
with at least `MIN_SHARE` of that column's squared loading. Dot size: that
best rank; colour: the pair's |loading| in that column (columns are
unit-norm, so 1 is a column made of the one pair). Rows are
grouped by which methods rank them in the top 10: all four, cccRISE and
some comparators, cccRISE only, comparators only (most comparators first).
Pair names: the cccRISE component's colour when cccRISE ranks them in its
top 10, else black."""

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
MIN_SHARE = 0.01  # a top-10 pair below this share of its column gets no dot


def _size(rank):
    """Marker area, in proportion to 1 / rank (ranks capped at `TOP_K`), so
    the top few ranks stay apart."""
    return 50 / np.clip(rank, 1, significance.TOP_K)


def _ranks(pairs) -> pd.DataFrame:
    """Per pair and method, its best rank in any column (comparators: either
    fit; cccRISE: its coarse copy, the same fit) as ``rank_<method>``, and in
    that column its |loading| (columns are unit-norm: sqrt of its share) as
    ``loading_<method>`` and its share of squared loading as
    ``share_<method>``."""
    p = load("profiles")
    p = p[
        p["lr_pair"].isin(pairs)
        & ~((p["method"] == "cccRISE") & (p["annotation"] != "coarse"))
    ]
    p = p.sort_values(["lr_rank", "share"], ascending=[True, False])
    best = p.drop_duplicates(["method", "lr_pair"]).assign(
        method=lambda t: t["method"].map(lambda m: SHORT.get(m, m)),
        loading=lambda t: np.sqrt(t["share"]),
    )
    wide = best.pivot(index="lr_pair", columns="method")[
        ["lr_rank", "loading", "share"]
    ]
    wide.columns = [f"{'rank' if v == 'lr_rank' else v}_{m}" for v, m in wide.columns]
    return wide.reindex(pairs)


def _calls(table) -> pd.DataFrame:
    """Whether each pair is in each method's top `TOP_K` (any column) with at
    least `MIN_SHARE` of that column's squared loading."""
    ranks = table[[f"rank_{c}" for c in COLUMNS]].set_axis(list(COLUMNS), axis=1)
    shares = table[[f"share_{c}" for c in COLUMNS]].set_axis(list(COLUMNS), axis=1)
    return ranks.le(significance.TOP_K) & shares.ge(MIN_SHARE)


def rows() -> pd.DataFrame:
    """Pairs significant anywhere, with their scores per method (`COLUMNS`),
    best ranks (`rank_<method>`), group, colour and row position `y`."""
    table = significance.combined()
    table = table[table["region"].notna()].rename(columns={"x": "cccRISE"})
    table = table.join(_ranks(table.index))
    calls = _calls(table)
    n_comparators = calls[list(COLUMNS[1:])].sum(axis=1)
    palette = sns.color_palette("Dark2", 5)
    table = table.assign(
        group=np.select(
            [
                calls.all(axis=1),
                calls["cccRISE"] & (n_comparators > 0),
                calls["cccRISE"],
            ],
            [0, 1, 2],
            default=3,
        ),
        n_comparators=n_comparators,
        pattern=calls.astype(int).astype(str).agg("".join, axis=1),
        color=[
            palette[c - 1] if top else "black"
            for c, top in zip(table["component_x"], calls["cccRISE"], strict=True)
        ],
    )
    table = table.sort_values(
        ["group", "n_comparators", "pattern", "component_x", "cccRISE"],
        ascending=[True, False, False, True, False],
    )
    return table.assign(y=np.arange(len(table)) + GAP * table["group"].to_numpy())


def _dots(ax, table):
    for j, column in enumerate(COLUMNS):
        score = table[f"loading_{column}"].fillna(0).to_numpy()
        called = _calls(table)[column].to_numpy()
        ax.scatter(
            np.full(called.sum(), j), table["y"][called],
            s=_size(table[f"rank_{column}"].to_numpy()[called]),
            c=CMAP(NORM(score[called])), edgecolor=style.COLORS["ink"], lw=0.4,
            zorder=2,
        )  # fmt: skip


def _legends(fig):
    fig.colorbar(
        ScalarMappable(NORM, CMAP), ax=fig.axes, label="|loading| (unit-norm column)",
        location="bottom", shrink=0.8, pad=0.02, ticks=[0, 0.5, 1],
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
