"""
Figure S3: CCC-RISE on BALF COVID-19 data. Showing weighted sender and receiver cell factors.
"""

from .common import COVID19_FACTORS_PATH
from .commonFuncs.plotPaCMAP import make_wc_pacmap_figure


def makeFigure():
    return make_wc_pacmap_figure(COVID19_FACTORS_PATH, grid=(5, 4), figsize=(18, 18))
