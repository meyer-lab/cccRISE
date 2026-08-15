"""
Figure S1c_d: FMS across CPD ranks (only) for COVID-19
"""

import numpy as np
import pandas as pd
import seaborn as sns

from ..import_data import import_ligand_receptor_pairs
from ..tensor import calculate_interaction_tensor, run_fms_r2x_analysis
from .common import get_covid_data, getSetup, load_or_compute_dataframe, subplotLabel


def compute_covid_cpd_stability_data(
    rank_list: list[int] | None = None,
    runs: int = 3,
) -> pd.DataFrame:
    """Compute FMS and R2X stability metrics across CPD ranks on COVID-19."""
    if rank_list is None:
        rank_list = list(range(1, 20))

    X = get_covid_data(project_cells=True)
    lr_pairs = import_ligand_receptor_pairs()

    interaction_tensor = calculate_interaction_tensor(X, lr_pairs)
    return run_fms_r2x_analysis(
        interaction_tensor, rank_list=rank_list, runs=runs, svd_init="random"
    )


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    df = load_or_compute_dataframe(
        "figS1c_d_cpd_stability.csv",
        compute_covid_cpd_stability_data,
    )

    sns.lineplot(data=df, x="Component", y="FMS", ax=ax[0], label="FMS")
    ax[0].set_ylim(0, 1)

    sns.lineplot(data=df, x="Component", y="R2X", ax=ax[1], color="orange", label="R2X")
    ax[1].set_ylim(0, float(np.max(df["R2X"])) + 0.02)

    return f
