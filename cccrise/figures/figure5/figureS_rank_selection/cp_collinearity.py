"""IBD CP collinearity of the LR pair mode across ten seeded SVD starts per
rank, from ``uv run cccrise-cpdrank``; the other modes' are in
`figureS_ibd_collinearity`."""

from ... import common

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    common.plot_collinearity(
        axs[0], DATASET, modes=("D",), rank=common.selected_ranks(DATASET)[1]
    )


def makeFigure():
    return common.standalone_fixed(__name__)
