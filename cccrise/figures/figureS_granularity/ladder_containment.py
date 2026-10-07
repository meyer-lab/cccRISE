"""Containment of cccRISE's C2, C3 and C4 in each comparator's span (rank 5)
as the annotation gets finer; lines and marks as in the CKA panel."""

from .. import common, style
from ._common import METHODS, draw_ladder, load_fits

COMPONENTS = (2, 3, 4)
MODES = {"condition": "condition", "lr": "LR"}


def plot(fig):
    table = load_fits()
    colors = dict(zip(METHODS, style.CATEGORICAL[1:], strict=False))
    axes = fig.subplots(len(MODES), len(COMPONENTS), sharex=True, sharey=True)
    for r, (mode, label) in enumerate(MODES.items()):
        for c, comp in enumerate(COMPONENTS):
            ax = axes[r, c]
            for method in METHODS:
                draw_ladder(
                    ax, table, method, f"contain_C{comp}_{mode}", colors[method]
                )
            ax.axhline(
                0.85,
                ls=style.LINE_STYLE_REFERENCE,
                color=style.COLORS["reference"],
                lw=style.LINE_WIDTH_THIN,
            )
            ax.set_ylim(0, 1.02)
            if r == 0:
                ax.set_title(f"C{comp}")
            else:
                ax.set_xlabel("cell types")
            if c == 0:
                ax.set_ylabel(f"containment\n({label})")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
