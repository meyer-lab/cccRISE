"""
Figure S1c_d: FMS across CPD ranks (only) for COVID-19
"""

from ...ccc_rise import calculate_interaction_tensor
from ...import_data import import_ligand_receptor_pairs
from ...tensor import run_fms_r2x_analysis
from ..common import get_covid_data, getSetup, subplotLabel
from .commonFuncs.plotGeneral import plot_fms_r2x_stability


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    X = get_covid_data(project_cells=True)
    lr_pairs = import_ligand_receptor_pairs()

    interaction_tensor, _ = calculate_interaction_tensor(X, lr_pairs)
    df = run_fms_r2x_analysis(
        interaction_tensor, rank_list=list(range(1, 11)), runs=3, svd_init="random"
    )

    plot_fms_r2x_stability(df, ax)
    return f
