"""
Figure A4d: Logistic regression weights for ALAD vs Control classification based on CCC-RISE components.
"""

import itertools

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegression

from ..import_data import add_cond_idxs
from ..utils import correct_conditions
from .common import ALAD_PATH, getSetup, subplotLabel
from .commonFuncs.plotGeneral import (
    binarize_group,
    get_sample_to_group,
    rotate_xaxis,
    rotate_yaxis,
)


def makeFigure():
    ax, f = getSetup((6, 6), (1, 1))
    subplotLabel(ax)

    # Import and prepare data
    X = anndata.read_h5ad(ALAD_PATH)

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    group_col = "ALADstatus"
    sample_to_group = get_sample_to_group(X, condition_column, group_col)
    sample_to_group = binarize_group(sample_to_group, "control", "control", "alad")
    print(np.unique(sample_to_group, return_counts=True))

    print(sample_to_group.shape)
    print()

    pair_logistic_regression(correct_conditions(X), sample_to_group, ax[0])

    return f


def pair_logistic_regression(X: np.ndarray, status_df: pd.DataFrame, ax: Axes):
    """Plot factor weights for donor SLE prediction"""
    lrmodel = LogisticRegression(penalty=None)
    all_comps = np.arange(X.shape[1])
    acc = np.zeros((X.shape[1], X.shape[1]))

    for comps in itertools.product(all_comps, all_comps):
        if comps[0] >= comps[1]:
            compFacs = X[:, [comps[0], comps[1]]]
            LR_CoH = lrmodel.fit(compFacs, status_df.values)
            acc[comps[0], comps[1]] = LR_CoH.score(compFacs, status_df.values)
            acc[comps[1], comps[0]] = acc[comps[0], comps[1]]

    mask = np.triu(np.ones_like(acc, dtype=bool))

    for i in range(len(mask)):
        mask[i, i] = False

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        data=acc,
        vmin=0.5,
        # vmax=1,
        xticklabels=all_comps + 1,
        yticklabels=all_comps + 1,
        mask=mask,
        cbar_kws={"label": "Prediction Accuracy"},
        ax=ax,
        cmap=cmap,
    )

    ax.set(xlabel="Component", ylabel="Component")
    rotate_xaxis(ax, rotation=0)
    rotate_yaxis(ax, rotation=0)
