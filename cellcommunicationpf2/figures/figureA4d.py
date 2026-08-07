"""
Figure A4d: Logistic regression weights for ALAD vs Control classification based on CCC-RISE components.
"""

import anndata
import numpy as np

from ..import_data import add_cond_idxs
from ..logreg import ccc_rise_logreg_weights
from ..utils import correct_conditions
from .common import ALAD_PATH, getSetup, subplotLabel
from .commonFuncs.plotGeneral import binarize_group, get_sample_to_group


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import and prepare data
    X = anndata.read_h5ad(ALAD_PATH)

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    group_col = "ALADstatus"
    sample_to_group = get_sample_to_group(X, condition_column, group_col)
    sample_to_group = binarize_group(sample_to_group, "control", "control", "alad")

    X.uns["A"] = correct_conditions(X)

    auc_roc_weights, _ = ccc_rise_logreg_weights(X, sample_to_group)

    ax[0].bar(np.arange(len(auc_roc_weights)) + 1, auc_roc_weights)

    ax[0].set_xlabel("Component")
    ax[0].set_ylabel("Logistic Regression Weight")

    ym = np.max(np.abs(auc_roc_weights)) * 1.1
    ax[0].set_ylim(-ym, ym)

    return f
