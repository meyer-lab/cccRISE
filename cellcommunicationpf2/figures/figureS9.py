"""
Figure S9: CCC-RISE on BALF alad data. Showing ligand-receptor factors.
"""

from .common import ALAD_STABLE_PATH
from .commonFuncs.plotFactors import make_lr_factors_partial_figure


def makeFigure():
    return make_lr_factors_partial_figure(ALAD_STABLE_PATH, grid=(6, 6))
