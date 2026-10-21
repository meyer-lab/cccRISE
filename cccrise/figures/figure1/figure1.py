"""Figure 1: CCC-RISE deconvolves cell communication variability across
single cells and conditions. Its panels are placeholders sized for the
hand-drawn illustrations; building it records their letters for the
manuscript."""

from .. import common, style
from . import cell_mapping, component_interpretation, embedding, workflow

LAYOUT = [
    [workflow],
    [embedding, component_interpretation, cell_mapping],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
