"""
Figure A6a: RISE Rank Analysis on ALAD data distinguishing 'declined' vs 'stable' conditions.
"""

import numpy as np
import pandas as pd

from ..import_data import add_cond_idxs, import_alad
from ..logreg import rise_ranks_logreg
from .common import getSetup, load_or_compute_dataframe, subplotLabel
from .commonFuncs.plotGeneral import binarize_group, get_sample_to_group


def compute_figA6a_rise_ranks() -> pd.DataFrame:
    """Compute 10-fold CV ROC-AUC and accuracy across RISE ranks on ALAD."""
    X = import_alad(gene_threshold=0.01, normalize=True)
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    group_col = "ALADstatus"
    sample_to_group = get_sample_to_group(X, condition_column, group_col)
    sample_to_group = binarize_group(sample_to_group, "declined", "declined", "stable")

    rank_list = list(np.append([1], range(5, 66, 5)))
    scoring = ["roc_auc", "accuracy"]
    scores_aucroc, scores_accuracy = rise_ranks_logreg(
        X, rank_list, sample_to_group, scoring, n_iter_max=10000, tolerance=1e-9
    )

    return pd.DataFrame(
        {"rank": rank_list, "roc_auc": scores_aucroc, "accuracy": scores_accuracy}
    )


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    df = load_or_compute_dataframe("figA6a_rise_ranks.csv", compute_figA6a_rise_ranks)

    ax[0].plot(df["rank"], df["roc_auc"], "o-", color="steelblue")
    ax[0].set_xlabel("RISE Rank")
    ax[0].set_ylabel("10-Fold CV: roc_auc")
    ax[0].set_ylim(0, float(np.max(df["roc_auc"])) + 0.05)

    ax[1].plot(df["rank"], df["accuracy"], "o-", color="coral")
    ax[1].set_xlabel("RISE Rank")
    ax[1].set_ylabel("10-Fold CV: accuracy")
    ax[1].set_ylim(0, float(np.max(df["accuracy"])) + 0.05)

    return f
