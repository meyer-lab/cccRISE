"""
Figure A4g: Heatmaps of ligand-receptor expression products for CCL17-CCR4 and CCL22-CCR4
"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import pandas as pd
import seaborn as sns
from ..utils import (
    expression_product_matrix,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    ccc_rise_cmp = 5

    X_mdc_sender = X[(X.obs["broad_cell_type"] == "CD4 T cells")]
    sender_weights = X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-np.abs(sender_weights))
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Dendritic Cells")]
    receiver_weights = X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1]
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(np.abs(receiver_weights))
    ]


    pairs = ["CD40LG", "CD40"]
    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, pairs[0], pairs[1])
    df_grouped = group_matrix(df)
    sns.heatmap(df_grouped, ax=ax[0], cmap="rocket")
    ax[0].set_title(f"{pairs[0]}-{pairs[1]} Interaction")
    ax[0].set_xlabel("Receiver DCs")
    ax[0].set_ylabel("Sender CD4 T Cells")

    # Weight verification: sender + receiver component weight sums
    s_sorted = np.abs(sender_weights[np.argsort(-np.abs(sender_weights))])
    r_sorted = np.abs(receiver_weights[np.argsort(np.abs(receiver_weights))])
    weight_matrix = group_weight_sum(s_sorted, r_sorted)
    sns.heatmap(weight_matrix, ax=ax[2], cmap="viridis")
    ax[2].set_title("Component Weight (sender + receiver)")
    ax[2].set_xlabel("Receiver DCs")
    ax[2].set_ylabel("Sender CD4 T Cells")


    X_mdc_sender = X[(X.obs["broad_cell_type"] == "Epithelial cells")]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-np.abs(X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1]))
    ]

    X_mdc_receiver = X[(X.obs["broad_cell_type"] == "Epithelial cells")]

    X_mdc_receiver = X_mdc_receiver[
        np.argsort(np.abs(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1]))
    ]

    df = expression_product_matrix(X_mdc_sender, X_mdc_receiver, pairs[0], pairs[1])
    df_grouped = group_matrix(df)
    sns.heatmap(df_grouped, ax=ax[1], cmap="rocket")
    ax[1].set_title(f"{pairs[0]}-{pairs[1]} Interaction")
    ax[1].set_xlabel("Receiver Epithelial Cells")
    ax[1].set_ylabel("Sender Epithelial Cells")

    # Component weight distributions for panel (a) cell types
    ax[3].hist(sender_weights, bins=30, alpha=0.6, label=f"Sender sc_B (CD4 T)\nn={len(sender_weights)}, {np.sum(sender_weights<0)} neg")
    ax[3].hist(receiver_weights, bins=30, alpha=0.6, label=f"Receiver rc_C (DCs)\nn={len(receiver_weights)}, {np.sum(receiver_weights<0)} neg")
    ax[3].axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax[3].set_xlabel("Component Weight")
    ax[3].set_ylabel("Count")
    ax[3].set_title(f"Cmp {ccc_rise_cmp} Weight Distribution")
    ax[3].legend(fontsize=6)

    return f


def group_matrix(df):
    """
    Groups a DataFrame into a 10x10 matrix by binning rows and columns and averaging within bins.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    row_group_size = n_rows // 10
    col_group_size = n_cols // 10
    row_groups = np.arange(n_rows) // row_group_size
    col_groups = np.arange(n_cols) // col_group_size
    row_groups = np.clip(row_groups, 0, 9)
    col_groups = np.clip(col_groups, 0, 9)
    df_grouped = df.groupby(row_groups).mean()
    df_grouped = df_grouped.groupby(col_groups, axis=1).mean()
    return df_grouped


def group_weight_sum(sender_weights, receiver_weights):
    """Bin weights into 10 groups each, return 10x10 matrix of sender + receiver means."""
    n_s, n_r = len(sender_weights), len(receiver_weights)
    s_groups = np.clip(np.arange(n_s) // (n_s // 10), 0, 9)
    r_groups = np.clip(np.arange(n_r) // (n_r // 10), 0, 9)
    s_means = pd.Series(sender_weights).groupby(s_groups).mean().values
    r_means = pd.Series(receiver_weights).groupby(r_groups).mean().values
    return pd.DataFrame(s_means[:, None] + r_means[None, :])