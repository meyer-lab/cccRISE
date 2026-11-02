"""CP collinearity of every mode across ten seeded SVD starts per rank, from
``uv run cccrise-cpdrank``; Figure 2 shows the LR pair mode's per start."""

from ... import common, style

DATASET = "alad"
MODES = ("A", "B", "C", "D")


def plot(fig):
    ax = fig.subplots()
    common.plot_collinearity_medians(
        ax, DATASET, modes=MODES, rank=common.selected_ranks(DATASET)[1]
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
