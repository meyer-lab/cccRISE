"""
Figure A2c_d: CCC-RISE on BALF COVID-19 data. Showing PaCMAP of cells colored by cell type and disease condition.
"""

from .common import COVID19_PATH
from .commonFuncs.plotPaCMAP import make_celltype_condition_pacmap_figure


def makeFigure():
    return make_celltype_condition_pacmap_figure(
        COVID19_PATH,
        celltype_col="celltype",
        condition_col="condition",
        sample_col="sample",
    )
