"""Supplementary figure: the EE cells behind Figure 5E-J. How EE cells are
called L-hormone+ (markers, as Figure 2 calls epithelium); where the zoomed
EE/EC map sits among all cells; and hormone expression on the zoomed map."""

from ... import common, style
from . import cell_map, hormone_maps, l_hormone_call

LAYOUT = [[l_hormone_call, cell_map], [hormone_maps]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
