"""The pre-registered main-text selection: the clinically associated
component C4 (sender S4, receiver R4), the eigen-state carrying it (E16), and
the two other eigen-states with the largest share of active cells in partly
active fine types (E4, E5)."""

from .. import common, style
from ._common import plot_grid

UNITS = ["S4", "R4", "E16", "E4", "E5"]


def plot(fig):
    plot_grid(fig, UNITS, n_cols=5)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 60 * style.MM))
