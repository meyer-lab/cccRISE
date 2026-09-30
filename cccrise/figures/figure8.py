"""
Figure 8: FMS and R2X across data percentage with fixed CPD and RISE ranks for IBD
"""

import pandas as pd

from ..import_data import add_cond_idxs, import_ligand_receptor_pairs
from ..tensor import run_fms_r2x_data_percentage_analysis
from .common import IBD_SPEC, getSetup, load_or_compute_dataframe, subplotLabel
from .commonFuncs.plotGeneral import plot_fms_r2x_stability


def compute_fms_r2x_percentage_data(
    # IBD's selected ranks; see ibd_ti_factors_cap2000.provenance.yaml
    rise_rank: int = 55,
    cp_rank: int = 2,
    runs: int = 3,
) -> pd.DataFrame:
    """Run FMS and R2X analysis across data subsampling percentages."""
    X = IBD_SPEC.load()
    lr_pairs = import_ligand_receptor_pairs()
    X_filtered = add_cond_idxs(X, IBD_SPEC.condition_column)
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

    plot_fms_r2x_stability(
        df,
        ax,
        x_col="Data_Percentage",
        x_label="Data Percentage (%)",
        y_label_fms="FMS",
        y_label_r2x="R2X",
        invert_xaxis=True,
    )
    return f
