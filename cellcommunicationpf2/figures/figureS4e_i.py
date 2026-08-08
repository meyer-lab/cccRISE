"""
Figure S4e-i: Heatmaps of ligand-receptor expression products for NK cell to Dendritic Cell interactions in CCC-RISE on BALF alad data.
"""

import anndata
import numpy as np
import seaborn as sns

from ..utils import (
    binned_expression_product_matrix,
)
from .common import (
    COVID19_PATH,
    getSetup,
    subplotLabel,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad(COVID19_PATH)
    ccc_rise_cmp = 3

    X_mdc_sender = X[X.obs["celltype"] == "Epithelial"]

    # Alter order based on factor value high to low
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["celltype"] == "Epithelial")]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    print("Epithelial sender cells:", X_mdc_sender.shape)

    pairs = [["PTN", "PTPRZ1"], ["PTN", "SDC1"]]
    for i, (lig, rec) in enumerate(pairs):
        df = binned_expression_product_matrix(X_mdc_sender, X_mdc_receiver, lig, rec)
        sns.heatmap(df, ax=ax[i], cmap="viridis")

        ax[i].set_xlabel("Receiver Epithelial Cells")
        ax[i].set_ylabel("Sender Epithelial Cells")
        ax[i].set_title(f"{lig}-{rec} Interaction")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    return f
