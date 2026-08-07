"""
Figure A3a: Comparison of CCC-RISE and Pseudobulk CPD on BALF COVID-19
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .common import (
    COVID19_PATH,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotGeneral import get_sample_to_group


def makeFigure():
    ax, f = getSetup((3, 3), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad(COVID19_PATH)
    condition_column = "sample"
    group_col = "condition"
    sample_to_group = get_sample_to_group(X, condition_column, group_col)
    A_factor = X.uns["A"]

    severity_map = {k: i + 1 for i, k in enumerate(np.unique(sample_to_group))}
    severity_numeric = sample_to_group.map(severity_map).values

    cmp = A_factor.shape[1]
    spearman_corr = np.zeros(cmp)
    for i in range(cmp):
        spearman_corr[i] = spearmanr(A_factor[:, i], severity_numeric)[0]
    spearman_df = pd.DataFrame(
        spearman_corr,
        index=pd.Index([f"{i + 1}" for i in range(cmp)]),
        columns=pd.Index(["Spearman Correlation"]),
    )
    spearman_df["Component"] = spearman_df.index
    sns.barplot(
        x="Component",
        y="Spearman Correlation",
        data=spearman_df,
        ax=ax[0],
        color="black",
    )
    ax[0].set_ylim(-0.1, 1)
    print(spearman_df)

    return f
