"""
Figure A4a: Prediction accuracy of RISE ranks using logistic regression on ALAD data.
"""

from ..import_data import add_cond_idxs, import_alad
from .common import getSetup, subplotLabel
from .commonFuncs.plotGeneral import binarize_group, get_sample_to_group


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_alad(gene_threshold=0.01, normalize=True)
    print(X)

    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    group_col = "ALADstatus"
    sample_to_group = get_sample_to_group(X, condition_column, group_col)
    sample_to_group = binarize_group(sample_to_group, "control", "control", "alad")

    # rank_list = list(np.append([1], range(5, 66, 5)))
    # rank_list = list(range(1, 4, 2))

    # scores_aucroc, scores_accuracy = rise_ranks_logreg(
    #     X, rank_list, sample_to_group, scoring, n_iter_max=10000, tolerance=1e-9
    # )

    # ax[0].plot(rank_list, scores_aucroc)
    # ax[0].set_xlabel("RISE Rank")
    # ax[0].set_ylabel("10-Fold CV: roc_auc")
    # ax[0].set_ylim(0, np.max(scores_aucroc) + 0.05)

    # ax[1].plot(rank_list, scores_accuracy)
    # ax[1].set_xlabel("RISE Rank")
    # ax[1].set_ylabel("10-Fold CV: accuracy")
    # ax[1].set_ylim(0, np.max(scores_accuracy) + 0.05)

    return f
