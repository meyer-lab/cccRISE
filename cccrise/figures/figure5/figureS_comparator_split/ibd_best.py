"""Each IBD CCC-RISE component's largest cos^2 with a single LR column of each
comparator fit.

Each IBD component's best single LR column in every comparator fit
(`_recovery.best_column_grid`; ALAD's is Figure 5A)."""

from ... import common
from .._recovery import COS2_LABEL, best_column_grid, colorbar_below

AXES = ("split_ibd_grid",)


def plot(axs):
    best_column_grid(axs[0], ("ibd",), bars=False)
    colorbar_below(axs[0], COS2_LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
