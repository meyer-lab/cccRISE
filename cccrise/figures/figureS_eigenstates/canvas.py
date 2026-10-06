"""Every RISE eigen-state's projection over its home coarse types, stacked
by fine type (see `_common`)."""

from .. import common, style
from ._common import load, plot_grid


def plot(fig):
    units = [u for u in load("summary")["unit"] if u.startswith("E")]
    plot_grid(fig, units, n_cols=4)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX))
