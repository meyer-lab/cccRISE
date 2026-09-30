"""
Figure S3: cccRISE on IBD data. Showing weighted sender and receiver cell factors.
"""

from .common import IBD_FACTORS_PATH
from .commonFuncs.plotPaCMAP import make_wc_pacmap_figure


def makeFigure():
    # Sender and receiver PaCMAPs for each of IBD's 2 CP components
    return make_wc_pacmap_figure(IBD_FACTORS_PATH, grid=(2, 2), figsize=(10, 10))
