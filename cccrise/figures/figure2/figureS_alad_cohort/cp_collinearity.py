"""Largest absolute cosine between two components in each CPD mode, per rank
and seeded SVD start, ALAD cohort."""

from ... import common

DATASET = "alad"
MODES = ("A", "B", "C", "D")
AXES = ("wide",)


def plot(axs):
    common.plot_collinearity_medians(
        axs[0], DATASET, modes=MODES, rank=common.selected_ranks(DATASET)[1]
    )


def makeFigure():
    return common.standalone_fixed(__name__)
