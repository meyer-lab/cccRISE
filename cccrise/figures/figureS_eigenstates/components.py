"""Every cccRISE component's sender (S) and receiver (R) per-cell loading,
over its home coarse types, stacked by fine type (see `_common`)."""

from .. import common, style
from ._common import load, plot_grid


def plot(fig):
    units = list(load("summary")["unit"])
    units = [u for u in units if u[0] == "S"] + [u for u in units if u[0] == "R"]
    plot_grid(fig, units, n_cols=5)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 2 * style.PANEL_HEIGHT))
