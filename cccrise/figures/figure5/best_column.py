"""Figure 5C: each program's best column in every fit."""

from .. import common
from ._recovery import best_column_grid

AXES = ("fig4_best",)


def plot(axs):
    best_column_grid(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
