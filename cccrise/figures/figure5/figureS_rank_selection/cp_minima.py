"""Distinct minima the ten seeded SVD starts reach at each IBD CP rank, from
``uv run cccrise-cpdrank``."""

from ... import common, cp_rank

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    cp_rank.plot_minima(axs[0], cp_rank.dataset_minima(DATASET))


def makeFigure():
    return common.standalone_fixed(__name__)
