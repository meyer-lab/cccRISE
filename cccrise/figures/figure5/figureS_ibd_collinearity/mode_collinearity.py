"""IBD CP collinearity of the condition, sender, and receiver modes across
ten seeded SVD starts per rank, from ``uv run cccrise-cpdrank``; the LR pair
mode's is in `figureS_rank_selection`."""

from ... import common

DATASET = "ibd"
MODES = ("A", "B", "C")
AXES = ("wide",)


def plot(axs):
    common.plot_collinearity_medians(
        axs[0], DATASET, modes=MODES, rank=common.selected_ranks(DATASET)[1]
    )


def makeFigure():
    return common.standalone_fixed(__name__)
