"""Supplementary figure: how each comparator column of Figure 5B gets its class.
The rules apply in order: contained in cccRISE's span (A), most of its loading on
one program's carrying pairs (B), broad or a single pair by its effective LR pairs
(C), homophilic (D); what is left is mixed pairs."""

from matplotlib.lines import Line2D

from ... import common, style
from .._programs import COLUMN_CLASSES
from . import column_breadth, homophilic_share, program_overlap, span_containment

LAYOUT = [[span_containment, program_overlap], [column_breadth, homophilic_share]]
#: The key's height under the upper row's axes (mm).
KEY_DROP = 18


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    upper = common.panel_axes(fig, "span_containment").get_position()
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=c, ms=3, label=k)
            for k, c in COLUMN_CLASSES.items()
        ],
        loc="center",
        bbox_to_anchor=(0.5, upper.y0 - KEY_DROP * style.MM / fig.get_figheight()),
        ncol=len(COLUMN_CLASSES),
        fontsize=style.FONT_SIZE_DENSE - 0.5,
        handletextpad=0.05,
        columnspacing=1.0,
    )
    return fig
