"""
Figure A3zb: XXX
"""

import anndata
import numpy as np
import scanpy as sc
import seaborn as sns

from ..utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
)
from .common import (
    getSetup,
    subplotLabel,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/ccc/bal_alad.h5ad")
    cmp5 = [5, "CD4 T cells"]
    cmp7 = [7, "CD4 T cells"]
    cmp13 = [13, "Dendritic Cells"]
    cmp17 = [17, "NK cells"]
    ccc_rise_cmp = cmp13[0]

    X_epi_sender = X[X.obs["broad_cell_type"] == cmp13[1]]
    X_epi_sender_top = add_obs_cmp_label(
        X_epi_sender, cmp=ccc_rise_cmp, pos=True, top_perc=10, type="sender"
    )
    X_epi_sender_top = add_obs_cmp_unique_one(X_epi_sender_top, cmp=ccc_rise_cmp)
    X_epi_sender_top = X_epi_sender_top[
        X_epi_sender_top.obs["Label"].isin([f"Cmp{ccc_rise_cmp}"])
    ]
    X_epi_sender_top.obs["Label"] = f"Cmp{ccc_rise_cmp}_Top"

    X_epi_sender_bottom = add_obs_cmp_label(
        X_epi_sender, cmp=ccc_rise_cmp, pos=False, top_perc=10, type="sender"
    )
    X_epi_sender_bottom = add_obs_cmp_unique_one(X_epi_sender_bottom, cmp=ccc_rise_cmp)
    X_epi_sender_bottom = X_epi_sender_bottom[
        X_epi_sender_bottom.obs["Label"].isin([f"Cmp{ccc_rise_cmp}"])
    ]
    X_epi_sender_bottom.obs["Label"] = "NoLabel"

    combined_X = anndata.concat([X_epi_sender_top, X_epi_sender_bottom])

    method_test = ["logreg", "wilcoxon"]
    axs = 0
    for method in method_test:
        sc.tl.rank_genes_groups(combined_X, "Label", method=method)
        results = combined_X.uns["rank_genes_groups"]
        top_n = 20
        top_idx_top = np.argsort(results["scores"][f"Cmp{ccc_rise_cmp}_Top"])[-top_n:][
            ::-1
        ]
        top_genes_top = results["names"][f"Cmp{ccc_rise_cmp}_Top"][top_idx_top]
        top_scores_top = results["scores"][f"Cmp{ccc_rise_cmp}_Top"][top_idx_top]

        sns.barplot(
            x=top_scores_top,
            y=top_genes_top,
            ax=ax[axs],
        )
        ax[axs].set_xlabel(f"Score: {method}")
        ax[axs].set_ylabel("Gene")
        axs += 1

    X_epi_sender_top.obs["Label"] = "NoLabel"
    X_epi_sender_bottom.obs["Label"] = f"Cmp{ccc_rise_cmp}_Bottom"
    combined_X = anndata.concat([X_epi_sender_bottom, X_epi_sender_top])

    for method in method_test:
        sc.tl.rank_genes_groups(combined_X, "Label", method=method)
        results = combined_X.uns["rank_genes_groups"]
        top_n = 20
        top_idx_bottom = np.argsort(results["scores"][f"Cmp{ccc_rise_cmp}_Bottom"])[
            -top_n:
        ][::-1]
        top_genes_bottom = results["names"][f"Cmp{ccc_rise_cmp}_Bottom"][top_idx_bottom]
        top_scores_bottom = results["scores"][f"Cmp{ccc_rise_cmp}_Bottom"][
            top_idx_bottom
        ]

        sns.barplot(
            x=top_scores_bottom,
            y=top_genes_bottom,
            ax=ax[axs],
        )
        ax[axs].set_xlabel(f"Score: {method}")
        ax[axs].set_ylabel("Gene")
        axs += 1

    ax[0].set_title("Top 10 perc vs Bot 10 perc: logreg")
    ax[1].set_title("Top 10 perc vs Bot 10 perc: Wilcoxon")
    ax[2].set_title("Bottom 10 perc vs Top 10 perc: logreg")
    ax[3].set_title("Bottom 10 perc vs Top 10 perc: Wilcoxon")

    return f
