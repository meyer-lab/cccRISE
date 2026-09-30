"""
Figure S10: cccRISE on BALF alad data. Showing weighted sender and receiver cell factors.
"""

from ..common import ALAD_FACTORS_PATH
from .commonFuncs.plotPaCMAP import make_wc_pacmap_figure


def makeFigure():
    return make_wc_pacmap_figure(ALAD_FACTORS_PATH, grid=(6, 6), figsize=(18, 18))
