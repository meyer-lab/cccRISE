"""
Figure S5h_i: Decomposition of pseudobulk communication data and tensor for ALAD data
"""

import pandas as pd

from ..import_data import (
    add_cond_idxs,
    import_alad,
)
from ..tensor import calculate_fms_rise, rise_store_r2x
from ..utils import resample
from .common import getSetup, load_or_compute_dataframe, subplotLabel
from .commonFuncs.plotGeneral import plot_fms_r2x_stability


def compute_alad_rise_stability_data(
    rank_list: list[int] | None = None,
    runs: int = 3,
) -> pd.DataFrame:
    """Compute FMS and R2X stability metrics across RISE ranks on ALAD."""
    if rank_list is None:
        rank_list = list(range(1, 30, 2))

    X = import_alad()
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    rows = []
    for i in rank_list:
        dataX, r2x = rise_store_r2x(X, rank=i, n_iter_max=100, tolerance=1e-6)

        for j in range(runs):
            sampledX, _ = rise_store_r2x(
                resample(X, condition_name=condition_column),
                rank=i,
                n_iter_max=100,
                tolerance=1e-6,
            )
            fmsScore = calculate_fms_rise(dataX, sampledX)
            rows.append({"Run": j, "Component": i, "FMS": fmsScore, "R2X": r2x})
            print(f"Rank {i}, Run {j}: FMS={fmsScore:.4f}, R2X={r2x:.4f}")

    return pd.DataFrame(rows)


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    df = load_or_compute_dataframe(
        "figS5h_i_rise_stability.csv",
        compute_alad_rise_stability_data,
    )

    plot_fms_r2x_stability(df, ax, title_prefix="RISE on ALAD scRNA-seq")
    return f
