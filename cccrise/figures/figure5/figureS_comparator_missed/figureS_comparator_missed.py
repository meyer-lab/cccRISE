"""Supplementary figure: what the comparators find that CCC-RISE does not, each
panel under a bar in its cohort's color. Each PB and STAC LR column's largest
share of its squared loading on one CCC-RISE component's carrying pairs, ALAD
(A) and IBD (B), each fit starting at the same height in both; and, stacked, the
carrying pairs of the columns under 0.5 share that are associated with outcome
(q < 0.05) and carry a signalling pair CCC-RISE lacks, beside each pair's
largest loading in any CCC-RISE component, ALAD (C) over IBD (D)."""

from ... import common, style
from .._common import DATASET_COLORS, DATASET_LABELS
from . import alad_contents, alad_share, ibd_contents, ibd_share

LAYOUT = [[alad_share, ibd_share, [alad_contents, ibd_contents]]]  # C over D
#: The bars' height over A and B (in), as C and D draw their own
#: (`_missed.cohort_bar`, whose margins also leave the row room for these)
BAR_GAP = 2 * style.MM


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    for dataset in DATASET_LABELS:
        common.bar_over(
            fig,
            [common.panel_axes(fig, f"{dataset}_share")],
            BAR_GAP,
            DATASET_LABELS[dataset],
            DATASET_COLORS[dataset],
        )
    return fig
