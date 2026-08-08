"""
Figure A2j: Violin plot of cell weight distributions for mDCs in CCC-RISE on BALF COVID-19 data.
"""

import anndata
import seaborn as sns

from .common import (
    COVID19_PATH,
    getSetup,
    subplotLabel,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import Anndata file
    X = anndata.read_h5ad(COVID19_PATH)
    ccc_rise_cmp = 6

    # Violin plot of cell weighting distribution for Mast cells for a component
    # Keep only cells with mast cells for the violin plot
    X_mdc = X[X.obs["celltype"] == "mDC"]
    X_mdc = X_mdc.obsm["sc_B"][:, ccc_rise_cmp - 1]

    sns.violinplot(data=X_mdc, ax=ax[0])
    ax[0].set_ylim(-0.1, 0.85)
    ax[0].set_xlabel("mDC Weight Distribution")
    ax[0].set_ylabel(f"Sender Cell Component {ccc_rise_cmp} Association")

    return f
