"""Distinct minima the ten seeded SVD starts reach at each IBD CP rank, from
``uv run cccrise-cpdrank``."""

from ... import common

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    common.plot_minima(axs[0], DATASET)


def makeFigure():
    return common.standalone_fixed(__name__)
