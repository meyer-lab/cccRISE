"""Number of seeded SVD starts reaching each distinct minimum at each CPD rank,
IBD cohort."""

from ... import common

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    common.plot_minima(axs[0], DATASET, common.selected_ranks(DATASET)[1])


def makeFigure():
    return common.standalone_fixed(__name__)
