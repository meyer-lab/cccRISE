"""
Figure S1a_b: FMS across RISE ranks (only) for IBD
"""

import pandas as pd

from ..ccc_rise import fit_rise
from ..import_data import add_cond_idxs
from ..tensor import calculate_fms_rise
from ..utils import resample
from .common import IBD_SPEC, getSetup, load_or_compute_dataframe, subplotLabel
from .commonFuncs.plotGeneral import plot_fms_r2x_stability


def compute_rise_stability_data(
    ranks: list[int] | None = None,
    runs: int = 3,
) -> pd.DataFrame:
    """Compute FMS and R2X stability metrics across RISE ranks on IBD."""
    if ranks is None:
        ranks = list(range(1, 20))

    X = IBD_SPEC.load()
    condition_column = IBD_SPEC.condition_column
    X = add_cond_idxs(X, condition_column)

    rows = []
    for i in ranks:
        print(f"Computing rank {i}")
        dataX, r2x = fit_rise(X, rank=i, n_iter_max=100, tolerance=1e-6)

        for j in range(runs):
            sampledX, _ = fit_rise(
                resample(X, condition_name=condition_column),
                rank=i,
                n_iter_max=100,
                tolerance=1e-6,
            )
            fmsScore = calculate_fms_rise(dataX, sampledX)
            rows.append({"Run": j, "Component": i, "FMS": fmsScore, "R2X": r2x})
            print(f"  Run {j}, FMS={fmsScore:.4f}, R2X={r2x:.4f}")

    return pd.DataFrame(rows)


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    df = load_or_compute_dataframe(
        "figS1a_b_rise_stability.csv",
        compute_rise_stability_data,
    )

    plot_fms_r2x_stability(df, ax, title_prefix="RISE on IBD")
    return f
