"""IBD CP collinearity by mode across ten seeded SVD starts per rank, from
``uv run cccrise-cpdrank``."""

from ... import common, cp_rank

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    cp_rank.plot_mode_collinearity(
        axs[0], common.load_cpd_rank(DATASET), common.selected_ranks(DATASET)[1]
    )


def makeFigure():
    return common.standalone_fixed(__name__)
