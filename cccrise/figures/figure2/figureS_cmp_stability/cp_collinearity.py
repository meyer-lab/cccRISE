"""CP collinearity of the condition, sender, and receiver modes across ten
seeded SVD starts per rank, from ``uv run cccrise-cpdrank``; the LR pair
mode's is in Figure 2."""

from ... import common, style

DATASET = "alad"
MODES = ("A", "B", "C")


def plot(fig):
    ax = fig.subplots()
    common.plot_collinearity(
        ax, DATASET, modes=MODES, rank=common.selected_ranks(DATASET)[1]
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
