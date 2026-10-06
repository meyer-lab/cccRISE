"""Supplementary figure: cccRISE against the tensor comparators -- subspace
agreement (linear CKA), per-component containment (shared components), and
how cccRISE's components spread over cell types."""

from .. import common, style
from . import cell_fractions, containment, subspace

PANELS = {
    "subspace": subspace,
    "containment": containment,
    "cell_fractions": cell_fractions,
}
LAYOUT = [["subspace"], ["containment"], ["cell_fractions"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, 3 * 52 * style.MM), LAYOUT, PANELS
    )
