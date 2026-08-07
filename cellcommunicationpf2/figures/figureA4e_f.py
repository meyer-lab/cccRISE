"""
Figure A2b-c: CCC-RISE on BALF COVID-19 data. Showing PaCMAP of cells colored by cell type.
"""

from .common import ALAD_PATH
from .commonFuncs.plotPaCMAP import make_celltype_condition_pacmap_figure


def makeFigure():
    return make_celltype_condition_pacmap_figure(
        ALAD_PATH,
        celltype_col="broad_cell_type",
        condition_col="ALADstatus",
    )
