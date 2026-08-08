"""
Figure S8a-b: Heatmaps of ligand-receptor interaction scores between sender and receiver cells for specific cell types and components in CCC-RISE on BALF alad data.
"""

import anndata
import numpy as np
import seaborn as sns

from ..utils import (
    binned_expression_product_matrix,
)
from .common import (
    ALAD_PATH,
    getSetup,
    subplotLabel,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad(ALAD_PATH)
    ccc_rise_cmp = 13

    X_mdc_sender = X[X.obs["broad_cell_type"] == "Dendritic Cells"]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Dendritic Cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    pairs = [["CD200", "CD200R1"]]
    for i, (lig, rec) in enumerate(pairs):
        df_grouped = binned_expression_product_matrix(
            X_mdc_sender, X_mdc_receiver, lig, rec
        )
        # Keep max value consistent across heatmaps for better comparison
        sns.heatmap(df_grouped, ax=ax[i], cmap="rocket", vmax=0.015)
        ax[i].set_title(f"{lig}-{rec} Interaction")
        ax[i].set_xlabel("Dendritic Cells")
        ax[i].set_ylabel("Dendritic Cells")

    ccc_rise_cmp = 13

    X_mdc_sender = X[X.obs["broad_cell_type"] == "Dendritic Cells"]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Epithelial cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    pairs = [["CD200", "CD200R1"]]
    for i, (lig, rec) in enumerate(pairs):
        df_grouped = binned_expression_product_matrix(
            X_mdc_sender, X_mdc_receiver, lig, rec
        )
        # Keep max value consistent across heatmaps for better comparison
        sns.heatmap(df_grouped, ax=ax[i + 2], cmap="rocket", vmax=0.015)
        ax[i + 2].set_title(f"{lig}-{rec} Interaction")
        ax[i + 2].set_xlabel("Epithelial Cells")
        ax[i + 2].set_ylabel("Dendritic Cells")

    return f
