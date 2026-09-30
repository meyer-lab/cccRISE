"""
Figure S2: cccRISE on IBD data. Showing ligand-receptor factors.
"""

from .common import IBD_FACTORS_PATH
from .commonFuncs.plotFactors import make_lr_factors_partial_figure


def makeFigure():
    # Top and bottom LR pairs for each of IBD's 2 CP components
    return make_lr_factors_partial_figure(IBD_FACTORS_PATH, grid=(2, 2))
