"""CP collinearity of every mode across ten seeded SVD starts per rank, from
``uv run cccrise-cpdrank``; Figure 2 shows the LR pair mode's per start."""

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
