"""Supplementary figure: cccRISE against the tensor comparators -- subspace
agreement (linear CKA), per-component containment and shared components, and
how cccRISE's components spread over cell types."""

from .. import common, style
from . import cell_fractions, containment, shared, subspace

PANELS = {
    "subspace": subspace,
    "containment": containment,
    "shared": shared,
    "cell_fractions": cell_fractions,
}
LAYOUT = [["subspace"], ["containment"], ["shared"], ["cell_fractions"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, 4 * 52 * style.MM), LAYOUT, PANELS
    )
