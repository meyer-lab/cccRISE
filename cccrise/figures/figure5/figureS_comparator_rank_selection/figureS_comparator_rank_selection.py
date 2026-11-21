"""Supplementary figure: each annotation-pooled comparator's own rank
selection, on ALAD and IBD together, one column per method and labelling.
Coarse labels on the top row, fine on the bottom: pseudobulk CPD by cccRISE's
rank-selection sweep (fit R2X across ten seeded SVD starts; as Figure 2C and
D, the LR pair mode's collinearity per start and the share of starts at the
most common minimum); Tensor-cell2cell by the Kneedle elbow of its error;
STACCato by its eigenvalue shares. Each cohort's chosen rank is marked; a
bar in each method's color groups its panels under each row, the row's
bars naming method and labelling."""

from ... import common, style
from .._common import LABELLING_NAMES, METHOD_COLORS
from . import (
    pb_collinearity_coarse,
    pb_collinearity_fine,
    pb_fit_coarse,
    pb_fit_fine,
    pb_minima_coarse,
    pb_minima_fine,
    stac_shares_coarse,
    stac_shares_fine,
    tc2c_elbow_coarse,
    tc2c_elbow_fine,
)

LAYOUT = [
    [
        pb_fit_coarse,
        pb_collinearity_coarse,
        pb_minima_coarse,
        tc2c_elbow_coarse,
        stac_shares_coarse,
    ],
    [
        pb_fit_fine,
        pb_collinearity_fine,
        pb_minima_fine,
        tc2c_elbow_fine,
        stac_shares_fine,
    ],
]


#: Each method's panels, per labelling row, grouped by a bar in the method's
#: color under them named with method and labelling ("PB Coarse")
METHODS = {
    "PB": ("pb_fit", "pb_collinearity", "pb_minima"),
    "TC2C": ("tc2c_elbow",),
    "STAC": ("stac_shares",),
}
#: Room for a row of bars and names (in), and the bars' drop below the row's
#: lowest label
BAR_ROOM, BAR_DROP = 6 * style.MM, 2 * style.MM


def _bars(fig, lab: str) -> None:
    """`lab`'s row's method bars under it, each named with the labelling."""
    axes = {
        m: [common.panel_axes(fig, f"{n}_{lab}") for n in names]
        for m, names in METHODS.items()
    }
    y = (
        common.lowest_label(fig, [a for group in axes.values() for a in group])
        - BAR_DROP
    )
    for method, group in axes.items():
        name = f"{method} {LABELLING_NAMES[lab]}"
        common.bar_under(fig, group, y, name, METHOD_COLORS[method])


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    fine = [common.panel_axes(fig, f"{n}_fine") for ns in METHODS.values() for n in ns]
    # Room under each row: below the fine row, and between the rows
    common.grow_figure(fig, bottom=BAR_ROOM)
    common.open_figure_gap(fig, common.highest_label(fig, fine), BAR_ROOM)
    for lab in LABELLING_NAMES:
        _bars(fig, lab)
    return fig
