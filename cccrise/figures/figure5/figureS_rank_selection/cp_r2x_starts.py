"""IBD CP fit R2X across ten seeded SVD starts per rank, from ``uv run
cccrise-cpdrank``, with the Kneedle elbow of the median curve."""

from ... import common, cp_rank

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    cp_rank.plot_r2x_starts(
        axs[0], common.load_cpd_rank(DATASET), common.selected_ranks(DATASET)[1]
    )


def makeFigure():
    return common.standalone_fixed(__name__)
