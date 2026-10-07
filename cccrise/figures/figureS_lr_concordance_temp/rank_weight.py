"""How much weight an LR column keeps down its ranking (companion to
`missed`): per column, each rank's |loading| relative to the column's largest
(left) and the cumulative share of its squared loading (right). cccRISE's
components in colour, every own-rank comparator column (coarse and fine) in
grey; dotted lines at the top-k and rank cuts of `missed`, dashed at
0.4 x max."""

import seaborn as sns

from .. import common, style
from ..figureS_lr_concordance._common import load
from .missed import RANK_CUT, K

DEPTH = 50
SIGNIFICANT = 0.4  # |loading| >= 0.4 x the column's largest
PANELS = {
    "rel_max": "|loading| / component max",
    "cum_share": "cumulative share of squared loading",
}


def _lines(ax, table, column, colors):
    comparators = table[table["method"] != "cccRISE"]
    for _key, sub in comparators.groupby(["annotation", "method", "component"]):
        ax.plot(
            sub["lr_rank"], sub[column], color=style.COLORS["muted"], lw=0.5, zorder=1
        )
    ours = table[(table["method"] == "cccRISE") & (table["annotation"] == "coarse")]
    for component, sub in ours.groupby("component"):
        n = int((sub["rel_max"] >= SIGNIFICANT).sum())
        ax.plot(
            sub["lr_rank"],
            sub[column],
            color=colors[component - 1],
            lw=style.LINE_WIDTH,
            label=f"C{component} ({n} ≥ {SIGNIFICANT}× max)",
            zorder=2,
        )


def plot(fig):
    table = load("profiles")
    table = table[table["lr_rank"] <= DEPTH]
    colors = sns.color_palette("Dark2", 5)
    axes = fig.subplots(1, 2)
    for ax, (column, ylabel) in zip(axes, PANELS.items(), strict=True):
        _lines(ax, table, column, colors)
        for cut in sorted({K, RANK_CUT}):
            ax.axvline(cut, ls=":", color=style.COLORS["reference"], lw=0.5)
        ax.set_xlim(1, DEPTH)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("rank in column")
        ax.set_ylabel(ylabel)
        ax.grid(False)
    axes[0].axhline(
        SIGNIFICANT,
        ls=style.LINE_STYLE_REFERENCE,
        color=style.COLORS["reference"],
        lw=0.5,
    )
    axes[0].plot(
        [], [], color=style.COLORS["muted"], lw=0.5, label="comparator columns"
    )
    axes[0].legend(fontsize=style.FONT_SIZE_DENSE, loc="upper right")


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 65 * style.MM))
