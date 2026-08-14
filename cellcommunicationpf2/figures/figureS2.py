"""
Figure S2: CCC-RISE on BALF COVID-19 data. Showing ligand-receptor factors.
"""

from .common import COVID19_FACTORS_PATH
from .commonFuncs.plotFactors import make_lr_factors_partial_figure


def makeFigure():
    return make_lr_factors_partial_figure(COVID19_FACTORS_PATH, grid=(5, 4))
