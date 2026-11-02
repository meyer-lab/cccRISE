"""IBD CP collinearity of every mode across ten seeded SVD starts per rank,
from ``uv run cccrise-cpdrank``; `cp_collinearity` shows the LR pair mode's
per start."""

from ... import common

DATASET = "ibd"
MODES = ("A", "B", "C", "D")
AXES = ("rect",)


def plot(axs):
    # Panel A's key already names the dashed selected-rank line
    common.plot_collinearity_medians(
        axs[0],
        DATASET,
        modes=MODES,
        rank=common.selected_ranks(DATASET)[1],
        key_rank=False,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
