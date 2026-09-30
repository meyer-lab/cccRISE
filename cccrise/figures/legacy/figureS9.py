"""
Figure S9: cccRISE on BALF alad data. Showing ligand-receptor factors.
"""

from ..common import ALAD_FACTORS_PATH
from .commonFuncs.plotFactors import make_lr_factors_partial_figure


def makeFigure():
    return make_lr_factors_partial_figure(ALAD_FACTORS_PATH, grid=(6, 6))
