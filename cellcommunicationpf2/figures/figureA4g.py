"""
Figure A4g: Heatmaps of ligand-receptor expression products for CCL17-CCR4 and CCL22-CCR4
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
    ccc_rise_cmp = 5

    X_mdc_sender = X[(X.obs["broad_cell_type"] == "CD4 T cells")]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Dendritic Cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    pairs = ["CD40LG", "CD40"]
    df_grouped = binned_expression_product_matrix(
        X_mdc_sender, X_mdc_receiver, pairs[0], pairs[1]
    )
    # Keep max value consistent across heatmaps for better comparison
    sns.heatmap(df_grouped, ax=ax[0], cmap="rocket")
    ax[0].set_title(f"{pairs[0]}-{pairs[1]} Interaction")
    ax[0].set_xlabel("Receiver Epithelial Cells")
    ax[0].set_ylabel("Sender CD4 T Cells")

    X_mdc_sender = X[(X.obs["broad_cell_type"] == "Epithelial cells")]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Epithelial cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    df_grouped = binned_expression_product_matrix(
        X_mdc_sender, X_mdc_receiver, pairs[0], pairs[1]
    )
    sns.heatmap(df_grouped, ax=ax[1], cmap="rocket")
    ax[1].set_title(f"{pairs[0]}-{pairs[1]} Interaction")
    ax[1].set_xlabel("Receiver Epithelial Cells")
    ax[1].set_ylabel("Sender Macrophages")

    return f
