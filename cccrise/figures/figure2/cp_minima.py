"""Distinct minima the ten seeded SVD starts reach at each CPD rank: how many
starts reach each, stacked from the one most starts reach (f_max)."""

from .. import common

DATASET = "alad"
AXES = ("square",)


def plot(axs):
    common.plot_minima(axs[0], DATASET, common.selected_ranks(DATASET)[1])


def makeFigure():
    return common.standalone_fixed(__name__)
