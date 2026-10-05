"""
Figure A6f: Heatmaps of ligand-receptor expression
"""

import numpy as np
import seaborn as sns

from ...utils import (
    binned_expression_product_matrix,
)
from ..common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = get_alad_data(project_cells=True)
    ccc_rise_cmp = 9

    X_mdc_sender = X[X.obs["broad_cell_type"] == "Dendritic Cells"]
    X_mdc_sender = X_mdc_sender[
        np.argsort(X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Dendritic Cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(-X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    pairs = [["CADM1", "CADM1"], ["ESAM", "ESAM"]]
    for i, (lig, rec) in enumerate(pairs):
        df_grouped = binned_expression_product_matrix(
            X_mdc_sender, X_mdc_receiver, lig, rec
        )
        # Keep max value consistent across heatmaps for better comparison
        sns.heatmap(df_grouped, ax=ax[i], cmap="rocket", vmax=0.12)
        ax[i].set_title(f"{lig}-{rec} Interaction")
        ax[i].set_xlabel("Dendritic Cells")
        ax[i].set_ylabel("Dendritic Cells")

    return f
