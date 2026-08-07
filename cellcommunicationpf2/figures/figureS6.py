"""
Figure S6: CCC-RISE on BALF alad data. Showing ligand-receptor factors.
"""

from .common import ALAD_PATH
from .commonFuncs.plotFactors import make_lr_factors_partial_figure


def makeFigure():
    return make_lr_factors_partial_figure(ALAD_PATH, grid=(6, 6))
