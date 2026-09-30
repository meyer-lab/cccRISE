"""
Figure A5c: Scatter plot of factor values for cccRISE components 12 and 14
"""

from ..common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)
from ..commonFuncs.plotFactors import plot_pair_lr_factors
from ..commonFuncs.plotGeneral import symmetrize_axes


def makeFigure():
    ax, f = getSetup((8, 8), (3, 3))  # 1 row, 3 columns for 3 L-R  pairs
    subplotLabel(ax)

    X = get_alad_data(project_cells=False)
    ccc_rise_cmp1 = 13
    ccc_rise_cmp2 = 17

    plot_pair_lr_factors(X, ccc_rise_cmp1, ccc_rise_cmp2, ax[6])
    symmetrize_axes(ax[6])
    return f
