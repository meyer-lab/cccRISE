"""
Figure A6b: Prediction accuracy of CPD ranks using logistic regression on ALAD data.
"""

import numpy as np
import pandas as pd

from ..import_data import add_cond_idxs, import_alad, import_ligand_receptor_pairs
from ..logreg import cpd_ranks_logreg
from ..tensor import calculate_interaction_tensor
from .common import getSetup, load_or_compute_dataframe, subplotLabel
from .commonFuncs.plotGeneral import binarize_group, get_sample_to_group


def compute_figA6b_cpd_ranks() -> pd.DataFrame:
    """Compute 10-fold CV ROC-AUC and accuracy across CPD ranks on ALAD."""
    X = import_alad(gene_threshold=0.01, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    group_col = "ALADstatus"
    sample_to_group = get_sample_to_group(X, condition_column, group_col)
    sample_to_group = binarize_group(sample_to_group, "declined", "declined", "stable")

    interaction_tensor = calculate_interaction_tensor(X, lr_pairs, rise_rank=30)
    rank_list = list(range(2, 52, 2))
    scoring = ["roc_auc", "accuracy"]

    scores_aucroc, scores_accuracy = cpd_ranks_logreg(
        X,
        interaction_tensor,
        rank_list,
        sample_to_group,
        scoring,
        n_iter_max=10000,
        random_state=1,
    )

    return pd.DataFrame(
        {"rank": rank_list, "roc_auc": scores_aucroc, "accuracy": scores_accuracy}
    )


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    df = load_or_compute_dataframe("figA6b_cpd_ranks.csv", compute_figA6b_cpd_ranks)

    ax[0].plot(df["rank"], df["roc_auc"], "o-", color="steelblue")
    ax[0].set_ylim(0, float(np.max(df["roc_auc"])) + 0.05)
    ax[0].set_xlabel("CPD Rank")
    ax[0].set_ylabel("10-Fold CV: roc_auc")

    ax[1].plot(df["rank"], df["accuracy"], "o-", color="coral")
    ax[1].set_ylim(0, float(np.max(df["accuracy"])) + 0.05)
    ax[1].set_xlabel("CPD Rank")
    ax[1].set_ylabel("10-Fold CV: accuracy")

    return f
