"""How much weight an LR column keeps down its ranking (why the
LR-concordance figure compares top-10s): for one method's own-rank fit on the
coarse annotation, each
column's |loading| relative to its largest (left) and the cumulative share of
its squared loading (right) by rank, one colour per column."""

import seaborn as sns

from .. import common, style
from ..figureS_lr_concordance._common import SHORT, load

DEPTH = 25
PANELS = {
    "rel_max": "|loading| / component max",
    "cum_share": "cumulative share\nof squared loading",
}


def palette(n: int) -> list:
    """cccRISE's five components keep the colours they have in `significance`."""
    return sns.color_palette("Dark2" if n <= 8 else "tab10", n)


def plot(fig, method: str = "cccRISE"):
    table = load("profiles")
    table = table[
        (table["method"] == method)
        & (table["annotation"] == "coarse")
        & (table["lr_rank"] <= DEPTH)
    ]
    prefix = "C" if method == "cccRISE" else f"{SHORT[method]} "
    columns = sorted(table["component"].unique())
    colors = dict(zip(columns, palette(len(columns)), strict=True))
    axes = fig.subplots(1, 2)
    for ax, (value, ylabel) in zip(axes, PANELS.items(), strict=True):
        for component, sub in table.groupby("component"):
            ax.plot(
                sub["lr_rank"],
                sub[value],
                color=colors[component],
                lw=style.LINE_WIDTH,
                label=f"{prefix}{component}",
            )
        ax.set_xlim(1, DEPTH)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("rank in component")
        ax.set_ylabel(ylabel)
        ax.grid(False)
    axes[0].legend(
        fontsize=style.FONT_SIZE_DENSE,
        loc="upper right",
        ncol=2 if len(columns) > 6 else 1,
    )
    fig.suptitle(method, fontsize=style.FONT_SIZE_TITLE)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 65 * style.MM))
