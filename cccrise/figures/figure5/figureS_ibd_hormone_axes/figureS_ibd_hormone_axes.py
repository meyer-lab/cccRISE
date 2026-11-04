"""Supplementary figure: the hormone programs' communication axes. The
eigenstates C3 and C4 load on belong to EE, EC and tuft cells; both programs
send from EE eigenstates, C3 to an EC eigenstate and C4 to tuft eigenstates;
and the ligands they send (PYY, GCG) are expressed, mostly together, by a
subset of EE cells. Figure 5E-G show the two programs splitting those cells."""

from ... import common, style
from . import (
    ee_hormones,
    eigenstate_association,
    eigenstate_composition,
    eigenstate_loadings,
)

LAYOUT = [
    [eigenstate_association, eigenstate_loadings],
    [eigenstate_composition, ee_hormones],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
