"""LR pairs one method weights and the other does not. Every pair in an
own-rank fit's top-10 (left: comparator top-10s ranked by cccRISE; right:
cccRISE's top-10 ranked by each comparator), by its share of its source
column's squared loading against its best rank in any column of the other
method. Above the dashed line (rank 50) the other method misses it; labelled
when it also carries >= 10% of a column."""

import numpy as np

from .. import common, style
from ._common import METHODS, SHORT, load

K = 10
SHARE_FLOOR = 1e-4
HEAVY = 0.1
DIRECTIONS = {
    "comparator top-k, cccRISE rank": "comparator top-10 → cccRISE rank",
    "cccRISE top-k, comparator rank": "cccRISE top-10 → comparator rank",
}


def _panel(ax, table, title):
    colors = dict(zip(METHODS, style.CATEGORICAL[1:], strict=False))
    for (annotation, method), sub in table.groupby(["annotation", "method"]):
        ax.scatter(
            np.clip(sub["source_max_share"], SHARE_FLOOR, None),
            sub["best_target_rank"],
            s=style.POINT_SIZE,
            marker="o" if annotation == "coarse" else "^",
            facecolor=colors[method] if annotation == "coarse" else "none",
            edgecolor=colors[method],
            lw=0.6,
            label=f"{SHORT[method]} {annotation}",
        )
    heavy = table[table["missed"] & (table["source_max_share"] >= HEAVY)]
    heavy = heavy.loc[heavy.groupby("lr_pair")["best_target_rank"].idxmax()]
    heavy = heavy.sort_values("best_target_rank")
    for i, row in enumerate(heavy.itertuples()):
        ax.annotate(
            row.lr_pair,
            (row.source_max_share, row.best_target_rank),
            xytext=(0.03, 0.30 - 0.06 * i),
            textcoords="axes fraction",
            fontsize=style.FONT_SIZE_DENSE - 1,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.4,
                "color": style.COLORS["reference"],
            },
        )
    ax.axhline(
        50, ls=style.LINE_STYLE_REFERENCE, color=style.COLORS["reference"], lw=0.5
    )
    ax.axvline(HEAVY, ls=":", color=style.COLORS["muted"], lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    n_missed = table.loc[table["missed"], "lr_pair"].nunique()
    ax.set_title(
        f"{title}\n{n_missed} of {table['lr_pair'].nunique()} pairs ranked > 50; "
        f"{len(heavy)} carry ≥ {HEAVY:.0%} of a column",
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.grid(False)


def plot(fig):
    table = load("missed")
    table = table[(table["k"] == K) & (table["method"].isin(METHODS))]
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (direction, title) in zip(axes, DIRECTIONS.items(), strict=True):
        _panel(ax, table[table["direction"] == direction], title)
        ax.set_xlabel("share of source column's loading")
    axes[0].set_ylabel("best rank in the other method")
    axes[0].invert_yaxis()
    axes[1].legend(
        fontsize=style.FONT_SIZE_DENSE - 1, loc="upper left", bbox_to_anchor=(1.02, 1)
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
