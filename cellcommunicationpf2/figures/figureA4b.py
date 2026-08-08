"""
Figure A4b: Prediction accuracy of CPD ranks using logistic regression on ALAD data.
"""

from ..import_data import add_cond_idxs, import_alad, import_ligand_receptor_pairs
from .common import getSetup, subplotLabel
from .commonFuncs.plotGeneral import binarize_group, get_sample_to_group


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    # Import and prepare data
    print("Importing and preparing")
    X = import_alad(gene_threshold=0.001, normalize=True)
    lr_pairs = import_ligand_receptor_pairs()  # noqa: F841

    # Add numerical indices for each patient sample, which is the primary condition
    condition_column = "dsco_id"
    X = add_cond_idxs(X, condition_column)

    group_col = "ALADstatus"
    sample_to_group = get_sample_to_group(X, condition_column, group_col)
    sample_to_group = binarize_group(sample_to_group, "control", "control", "alad")

    # interaction_tensor = calculate_interaction_tensor(X, lr_pairs, rise_rank=15)

    # # rank_list = list(range(2, 42, 2))
    # rank_list = list(range(1, 4, 2))

    # scoring = ["roc_auc", "accuracy"]
    # scores_aucroc, scores_accuracy = cpd_ranks_logreg(
    #     X,
    #     interaction_tensor,
    #     rank_list,
    #     sample_to_group,
    #     scoring,
    #     n_iter_max=10000,
    # )

    # ax[0].plot(rank_list, scores_aucroc)
    # ax[0].set_ylim(0, np.max(scores_aucroc) + 0.05)
    # ax[0].set_xlabel("CPD Rank")
    # ax[0].set_ylabel("10-Fold CV: roc_auc")

    # ax[1].plot(rank_list, scores_accuracy)
    # ax[1].set_ylim(0, np.max(scores_accuracy) + 0.05)
    # ax[1].set_xlabel("CPD Rank")
    # ax[1].set_ylabel("10-Fold CV: accuracy")

    return f
