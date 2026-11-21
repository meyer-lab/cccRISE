"""Each ALAD CCC-RISE component's largest cos^2 with a single LR column of each
comparator fit.

Figure 5A: each ALAD CCC-RISE component's best single LR column in every
fit, to read against 5B's containment: containment well above the best
column's cos^2 is a component held across several columns rather than in one.
IBD's is in `figureS_comparator_ibd_and_depth`."""

from .. import common
from ._recovery import COS2_LABEL, best_column_grid, colorbar_beneath

AXES = ("fig5_best",)


def plot(axs):
    best_column_grid(axs[0], ("alad",), bars=False)
    colorbar_beneath(axs[0], COS2_LABEL, like_5c=True)


def makeFigure():
    return common.standalone_fixed(__name__)
