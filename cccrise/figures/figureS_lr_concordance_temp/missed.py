"""Pairs one method weights and the other ranks low (draft panel D).

Every pair in an own-rank fit's top-10, one point per fit: left, comparator
top-10s ranked by cccRISE; right, cccRISE's top-10 ranked by each comparator.
x is the pair's share of its source column's squared loading (log); y its
best rank in any column of the other method (rank 1 at the top: ranks up to
`RANK_CUT` linear in the top `TOP` of the axis, the rest log-spaced down to
the worst rank of either direction). Shaded on the right of the
weight line: the other method has the pair in its top `RANK_CUT` (top) or not
(bottom); every pair in the bottom region is labelled."""

import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from .. import style
from ..figureS_lr_concordance._common import METHODS, SHORT, load

K = 10
RANK_CUT = 10
WEIGHT_CUT = 0.1
SHARE_FLOOR = 1e-4
TOP = 0.25  # share of the rank axis given to ranks 1..RANK_CUT
LABEL_GAP = 0.05  # axis units between stacked labels
RANK_TICKS = (1, 5, 10, 20, 50, 100, 200)
X_LIM = (6e-5, 1.6)
DIRECTIONS = {
    "comparator top-k, cccRISE rank": (
        "Comparator top-10 pairs",
        "LR loading share in comparator component",
        "best rank in cccRISE",
    ),
    "cccRISE top-k, comparator rank": (
        "cccRISE top-10 pairs",
        "LR loading share in cccRISE component",
        "best rank in comparator",
    ),
}
SHADES = {"top": style.COLORS["highlight"], "bottom": style.COLORS["up"]}


def _y(rank, worst):
    """Rank -> axis position in [0, 1] (0 = rank 1)."""
    r = np.asarray(rank, float)
    head = (r - 1) / (RANK_CUT - 1) * TOP
    tail = TOP + (1 - TOP) * np.log(np.maximum(r, RANK_CUT) / RANK_CUT) / np.log(
        worst / RANK_CUT
    )
    return np.where(r <= RANK_CUT, head, tail)


def _shade(ax):
    for (y0, y1), color in zip(
        ((-0.03, TOP), (TOP, 1.03)), SHADES.values(), strict=True
    ):
        ax.fill_between(
            [WEIGHT_CUT, X_LIM[1]], y0, y1, color=color, alpha=0.1, lw=0, zorder=0
        )
    ax.axhline(
        TOP, ls=style.LINE_STYLE_REFERENCE, color=style.COLORS["reference"], lw=0.5
    )
    ax.axvline(WEIGHT_CUT, ls=":", color=style.COLORS["reference"], lw=0.5)


def _rank_axis(ax, worst):
    ax.set_ylim(1.03, -0.03)
    ticks = [t for t in RANK_TICKS if t < worst * 0.8] + [worst]
    ax.yaxis.set_major_locator(FixedLocator(_y(ticks, worst)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda _v, i: f"{ticks[i]}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))


def _share_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1]))
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _p: (
                "≤0.01%"
                if v <= SHARE_FLOOR
                else f"{v:.{max(0, -int(np.floor(np.log10(v))) - 2)}%}"
            )
        )
    )


def _sources(rows, direction):
    """Per pair, where its source weight comes from: cccRISE components, or
    every comparator fit that has it in its top-k."""
    if direction.startswith("cccRISE"):
        names = rows["source_components"].str.split(",").explode().map("C{}".format)
    else:
        names = rows["method"].map(SHORT) + rows["annotation"].map(
            {"coarse": "", "fine": " fine"}
        )
    return names.groupby(rows["lr_pair"]).agg(lambda n: ", ".join(dict.fromkeys(n)))


def _spread(ys, gap, low, high):
    """`ys` (sorted) moved apart to at least `gap`, within [low, high]: pushed
    down from `low`, then the bottom of the stack pushed back up to `high`."""
    steps = gap * np.arange(len(ys))
    down = np.maximum.accumulate(np.maximum(ys, low) - steps) + steps
    back = steps[::-1]
    return np.minimum.accumulate((np.minimum(down, high) + back)[::-1])[::-1] - back


def _labels(ax, table, direction, worst):
    """One label per pair in the bottom-right region, at its worst rank,
    in a column right of the axes at about its point's height."""
    low = table[
        (table["source_max_share"] >= WEIGHT_CUT)
        & (table["best_target_rank"] > RANK_CUT)
    ]
    rows = low.loc[low.groupby("lr_pair")["best_target_rank"].idxmax()]
    rows = rows.assign(y=_y(rows["best_target_rank"], worst)).sort_values("y")
    sources = _sources(low, direction)
    ys = _spread(rows["y"].to_numpy(float), LABEL_GAP, TOP + 0.1, 1.0)
    for y, row in zip(ys, rows.itertuples(), strict=True):
        ax.annotate(
            f"{row.lr_pair.replace('^', '–')} ({sources[row.lr_pair]})",
            (row.source_max_share, row.y),
            xytext=(1.04, y),
            textcoords=("axes fraction", "data"),
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.35,
                "color": style.COLORS["reference"],
            },
        )


def _count(ax, table):
    """Distinct heavy pairs in each shaded region, right of the axes."""
    heavy = table[table["source_max_share"] >= WEIGHT_CUT]
    found = heavy["best_target_rank"] <= RANK_CUT
    for y, n, text in (
        (TOP / 2, heavy.loc[found, "lr_pair"].nunique(), "in"),
        (TOP + 0.04, heavy.loc[~found, "lr_pair"].nunique(), "not in"),
    ):
        ax.text(
            1.04,
            y,
            f"{n} pair{'' if n == 1 else 's'} {text} top {RANK_CUT}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold",
            color=SHADES["top" if text == "in" else "bottom"],
        )


def _panel(ax, table, direction, worst):
    title, xlabel, ylabel = DIRECTIONS[direction]
    colors = dict(zip(METHODS, style.CATEGORICAL[1:], strict=False))
    _shade(ax)
    for (annotation, method), sub in table.groupby(
        ["annotation", "method"], sort=False
    ):
        coarse = annotation == "coarse"
        ax.scatter(
            np.clip(sub["source_max_share"], SHARE_FLOOR, None),
            _y(sub["best_target_rank"], worst),
            s=style.POINT_SIZE,
            marker="o" if coarse else "^",
            facecolor=colors[method] if coarse else "none",
            edgecolor=colors[method],
            lw=0.6,
            alpha=0.85,
            label=f"{SHORT[method]} {annotation}",
            zorder=2,
        )
    _share_axis(ax)
    _rank_axis(ax, worst)
    _count(ax, table)
    _labels(ax, table, direction, worst)
    ax.set_title(title, fontsize=style.FONT_SIZE_TITLE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)


def plot(fig, annotations):
    table = load("missed")
    table = table[(table["k"] == K) & table["method"].isin(METHODS)]
    worst = int(table["best_target_rank"].max())  # shared by every version
    table = table[table["annotation"].isin(annotations)]
    axes = fig.subplots(1, 2)
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        _panel(ax, table[table["direction"] == direction], direction, worst)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        fontsize=style.FONT_SIZE_DENSE,
    )
