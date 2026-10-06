"""Supplementary figure: does a finer annotation make the comparators look
like cccRISE? Agreement (A) and per-component containment (B) with the
comparators fit on coarse vs fine cell types; how much of each cccRISE
component a label explains (C); and the clinically associated program and
the proliferating sub-states across fine types (D)."""

from .. import common, style
from ..figureS_eigenstates import selected
from . import cka, containment, eta_squared

PANELS = {
    "cka": cka,
    "containment": containment,
    "eta": eta_squared,
    "selected": selected,
}
LAYOUT = [["cka", "containment"], ["eta", "eta"], ["selected", "selected"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[1, 0.7, 1],
    )
