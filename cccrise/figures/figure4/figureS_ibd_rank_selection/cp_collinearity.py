"""IBD CP collinearity of the LR pair mode across ten seeded SVD starts per
rank, from ``uv run cccrise-cpdrank``; `mode_collinearity` shows every
mode's median."""

from ... import common

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    common.plot_collinearity_boxes(
        axs[0],
        DATASET,
        rank=common.selected_ranks(DATASET)[1],
        wrap_key=True,
        key_loc="center right",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
