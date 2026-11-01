"""Distinct minima the ten seeded SVD starts reach at each CP rank: how many
starts reach each, stacked from the one most starts reach (f_max)."""

from .. import common

DATASET = "alad"
AXES = ("square",)


def plot(axs):
    common.plot_minima(axs[0], DATASET)


def makeFigure():
    return common.standalone_fixed(__name__)
