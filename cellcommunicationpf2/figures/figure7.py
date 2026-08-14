"""
Figure 7: RISE Projection Boxplots by Cell Type

This figure plots boxplots of single-cell projection values for each cell type
across eigen-states using projections built directly from the NetCDF factor files.
"""

import numpy as np

from .common import get_covid_data, getSetup, subplotLabel


def makeFigure():
    """Generate Figure 7 showing RISE projection boxplots for each cell type."""
    X = get_covid_data(project_cells=True, do_embedding=False)

    projections = X.obsm["projections"]
    celltype_labels = X.obs["celltype"].values
    unique_celltypes = np.unique(celltype_labels)

    n_eigenstates = min(30, projections.shape[1])
    min_val, max_val = -0.5, 0.5

    # Setup figure: 6 rows x 5 columns
    nrows, ncols = 6, 5
    ax, f = getSetup((3 * ncols, 2 * nrows), (nrows, ncols))
    subplotLabel(ax)

    # For each eigen-state, plot horizontal boxplots for all cell types
    for r in range(n_eigenstates):
        data = [projections[celltype_labels == ct, r] for ct in unique_celltypes]
        ax[r].boxplot(
            data,
            tick_labels=unique_celltypes,
            orientation="horizontal",
            showfliers=False,
        )

        if r % ncols == 0:
            ax[r].set_ylabel("Cell Type")
        else:
            ax[r].set_yticklabels([])
        ax[r].set_xlabel(f"Eigen-state {r + 1} Projection Value")
        ax[r].set_xlim(min_val, max_val)

    return f
