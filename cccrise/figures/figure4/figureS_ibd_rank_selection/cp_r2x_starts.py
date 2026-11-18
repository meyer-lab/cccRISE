"""Fit R2X of each seeded SVD start at each CPD rank, IBD cohort."""

from ... import common

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    common.plot_r2x_starts(axs[0], DATASET, common.selected_ranks(DATASET)[1])


def makeFigure():
    return common.standalone_fixed(__name__)
