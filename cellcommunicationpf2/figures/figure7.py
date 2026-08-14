"""
Figure 7: RISE Projection Boxplots by Cell Type

This figure runs RISE, stacks the projections, and for each eigen-state,
plots boxplots of the projection values for each cell type.
"""

import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd

from ..import_data import add_cond_idxs, import_balf_covid
from .common import getSetup, load_or_compute_dataframe, subplotLabel


def compute_rise_projections_data(n_eigenstates: int = 30) -> pd.DataFrame:
    """Run RISE decomposition and return stacked projections per cell type."""
    adata = import_balf_covid()
    adata = add_cond_idxs(adata, "sample")

    pf2_output, _ = parafac2_nd(
        adata, rank=n_eigenstates, n_iter_max=100, tol=1e-6, random_state=42
    )
    _, _, projections = pf2_output
    stacked = np.concatenate(
        [proj.T for proj in projections], axis=1
    )  # (n_eigenstates, total_cells)

    celltype_labels = np.concatenate(
        [
            adata.obs.loc[adata.obs["condition_unique_idxs"] == i, "celltype"].values
            for i in range(len(projections))
        ]
    )

    data_dict = {f"eigenstate_{r + 1}": stacked[r, :] for r in range(n_eigenstates)}
    data_dict["celltype"] = celltype_labels
    return pd.DataFrame(data_dict)


def makeFigure():
    """Generate Figure 7 showing RISE projection boxplots for each cell type."""
    n_eigenstates = 30
    df = load_or_compute_dataframe(
        "fig7_rise_projections.parquet",
        compute_rise_projections_data,
        n_eigenstates=n_eigenstates,
    )

    celltypes = sorted(df["celltype"].unique())
    min_val, max_val = -0.5, 0.5

    # Setup figure: 6 rows x 5 columns
    nrows, ncols = 6, 5
    ax, f = getSetup((3 * ncols, 2 * nrows), (nrows, ncols))
    subplotLabel(ax)

    # For each eigen-state, plot horizontal boxplots for all cell types
    for r in range(n_eigenstates):
        col = f"eigenstate_{r + 1}"
        data = [df.loc[df["celltype"] == ct, col].values for ct in celltypes]
        ax[r].boxplot(
            data, tick_labels=celltypes, orientation="horizontal", showfliers=False
        )

        if r % ncols == 0:
            ax[r].set_ylabel("Cell Type")
        else:
            ax[r].set_yticklabels([])
        ax[r].set_xlabel(f"Eigen-state {r + 1} Projection Value")
        ax[r].set_xlim(min_val, max_val)

    return f
