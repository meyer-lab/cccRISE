"""IBD CP fit R2X across ten seeded SVD starts per rank, from ``uv run
cccrise-cpdrank``, with the Kneedle elbow of the median curve."""

from ... import common

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    common.plot_r2x_starts(axs[0], DATASET, common.selected_ranks(DATASET)[1])


def makeFigure():
    return common.standalone_fixed(__name__)
