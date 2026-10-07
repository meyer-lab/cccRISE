"""Does a finer annotation let a cell-type method find C4 or its clinical
association? For each CP comparator: |cos| of its column closest to C4 on
the LR mode (left), that column's eta-squared with 6-month CLAD/death
(middle; line: C4's own, maxT p 0.024), and the share of its columns
carried by one sample (right); lines and marks as in the CKA panel."""

from .. import common, style
from ._common import METHODS, draw_ladder, load_fits

CP_METHODS = METHODS[:2]
C4_ETA2 = 0.133
COLUMNS = {
    "c4_lr_cos": "|cos| with C4 (LR)",
    "c4_eta2_6m": "$\\eta^2$, 6-month CLAD",
    "share_one_sample": "columns on one sample",
}


def plot(fig):
    table = load_fits()
    colors = dict(zip(METHODS, style.CATEGORICAL[1:], strict=False))
    axes = fig.subplots(1, len(COLUMNS))
    for ax, (column, label) in zip(axes, COLUMNS.items(), strict=True):
        for method in CP_METHODS:
            draw_ladder(ax, table, method, column, colors[method])
        ax.set_ylabel(label)
        ax.set_xlabel("cell types")
        ax.set_ylim(0, 1.02 if column != "c4_eta2_6m" else 0.16)
    axes[1].axhline(C4_ETA2, color=style.COLORS["emphasis"], lw=style.LINE_WIDTH_THIN)
    axes[1].text(
        8,
        C4_ETA2,
        " C4",
        va="bottom",
        fontsize=style.FONT_SIZE_DENSE,
        color=style.COLORS["emphasis"],
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
