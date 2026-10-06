"""Supplementary figure: the follow-ups. F1, the comparators refit on
tempered pseudobulk tensors: containment of cccRISE's components (A), how
concentrated their components are (B) and their best clinical association
(C). F2, cccRISE's components tested within single cell types (D)."""

from .. import common, style
from . import celltype_units, clinical, concentration, containment

PANELS = {
    "containment": containment,
    "concentration": concentration,
    "clinical": clinical,
    "units": celltype_units,
}
LAYOUT = [
    ["containment", "containment"],
    ["concentration", "clinical"],
    ["units", "units"],
]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[1.3, 0.8, 0.9],
    )
