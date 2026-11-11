"""Figure 5A: each CCC-RISE component's best single LR column in every fit, to
read against 5B's containment: containment well above the best column's
cos^2 is a component held across several columns rather than in one."""

from .. import common
from ._recovery import best_column_grid

AXES = ("fig4_best",)


def plot(axs):
    best_column_grid(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
