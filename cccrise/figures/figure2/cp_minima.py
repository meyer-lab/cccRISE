"""Distinct minima the ten seeded SVD starts reach at each CP rank: how many
starts reach each, and its FMS to the one most starts reach (f_max)."""

from .. import common, cp_rank

DATASET = "alad"
AXES = ("square",)


def plot(axs):
    cp_rank.plot_minima(axs[0], cp_rank.dataset_minima(DATASET))


def makeFigure():
    return common.standalone_fixed(__name__)
