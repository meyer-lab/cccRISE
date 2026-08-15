"""
Figure S5f_g: Decomposition of the communication tensor from Tensorcell2cell on ALAD data
"""

import pandas as pd

from ..import_data import import_ligand_receptor_pairs
from ..tensor import calculate_interaction_tensor, run_fms_r2x_analysis
from .common import get_alad_data, getSetup, load_or_compute_dataframe, subplotLabel
from .commonFuncs.plotGeneral import plot_fms_r2x_stability


def compute_alad_cpd_stability_data(
    rank_list: list[int] | None = None,
    runs: int = 3,
) -> pd.DataFrame:
    """Compute FMS and R2X stability metrics across CPD ranks on ALAD."""
    if rank_list is None:
        rank_list = list(range(1, 27, 1))

    X = get_alad_data(project_cells=True)
    lr_pairs = import_ligand_receptor_pairs()

    interaction_tensor = calculate_interaction_tensor(X, lr_pairs)
    return run_fms_r2x_analysis(
        interaction_tensor, rank_list=rank_list, runs=runs, svd_init="random"
    )


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    df = load_or_compute_dataframe(
        "figS5f_g_cpd_stability.csv",
        compute_alad_cpd_stability_data,
    )

    plot_fms_r2x_stability(df, ax)
    return f
