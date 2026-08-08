"""
Figure A5k_l: Heatmaps of ligand-receptor expression products for NK cell to Dendritic Cell interactions.
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
    ccc_rise_cmp = 17

    X_mdc_sender = X[X.obs["broad_cell_type"] == "NK cells"]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "NK cells")]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    pairs = [["GZMA", "F2R"]]
    for i, (lig, rec) in enumerate(pairs):
        df_grouped = binned_expression_product_matrix(
            X_mdc_sender, X_mdc_receiver, lig, rec
        )
        sns.heatmap(df_grouped, ax=ax[i], cmap="rocket")
        ax[i].set_title(f"{lig}-{rec} Interaction")
        ax[i].set_xlabel("NK Cells")
        ax[i].set_ylabel("NK Cells")

    X_mdc_sender = X[X.obs["broad_cell_type"] == "CD4 T cells"]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "CD4 T cells")]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    pairs = [["GZMA", "F2R"]]
    for i, (lig, rec) in enumerate(pairs):
        df_grouped = binned_expression_product_matrix(
            X_mdc_sender, X_mdc_receiver, lig, rec
        )
        sns.heatmap(df_grouped, ax=ax[1], cmap="rocket")
        ax[1].set_title(f"{lig}-{rec} Interaction")
        ax[1].set_xlabel("CD4 T Cells")
        ax[1].set_ylabel("CD4 T Cells")

    return f
