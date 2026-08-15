"""
Figure 8: FMS and R2X across data percentage with fixed CPD and RISE ranks for COVID-19
"""

import numpy as np
import pandas as pd
import seaborn as sns

from ..import_data import add_cond_idxs, import_balf_covid, import_ligand_receptor_pairs
from ..tensor import run_fms_r2x_data_percentage_analysis
from .common import getSetup, load_or_compute_dataframe, subplotLabel


def compute_fms_r2x_percentage_data(
    rise_rank: int = 35,
    cp_rank: int = 8,
    runs: int = 3,
) -> pd.DataFrame:
    """Run FMS and R2X analysis across data subsampling percentages."""
    X = import_balf_covid(gene_threshold=0.001, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()
    condition_column = "sample"
    X_filtered = add_cond_idxs(X, condition_column)
    percentage_list = list(range(100, 0, -5))

    return run_fms_r2x_data_percentage_analysis(
        X_filtered,
        lr_pairs,
        rise_rank=rise_rank,
        cp_rank=cp_rank,
        percentage_list=percentage_list,
        runs=runs,
        svd_init="random",
    )


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    df = load_or_compute_dataframe(
        "fig8_data_percentage.csv",
        compute_fms_r2x_percentage_data,
    )

    sns.lineplot(data=df, x="Data_Percentage", y="FMS", ax=ax[0])
    ax[0].set_xlabel("Data Percentage (%)")
    ax[0].set_ylabel("FMS")
    ax[0].set_ylim(0, 1)
    ax[0].invert_xaxis()

    sns.lineplot(data=df, x="Data_Percentage", y="R2X", ax=ax[1], color="orange")
    ax[1].set_xlabel("Data Percentage (%)")
    ax[1].set_ylabel("R2X")
    ax[1].set_ylim(0, float(np.max(df["R2X"])) + 0.02)
    ax[1].invert_xaxis()

    return f
