"""
Figure A63: Scatter plot of factor values for cccRISE
"""

from ..common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotFactors import plot_pair_lr_factors
from .commonFuncs.plotGeneral import symmetrize_axes


def makeFigure():
    ax, f = getSetup((8, 8), (3, 3))  # 1 row, 3 columns for 3 L-R  pairs
    subplotLabel(ax)

    X = get_alad_data(project_cells=False)
    ccc_rise_cmp1 = 5
    ccc_rise_cmp2 = 9

    plot_pair_lr_factors(X, ccc_rise_cmp1, ccc_rise_cmp2, ax[6])
    symmetrize_axes(ax[6])
    return f
