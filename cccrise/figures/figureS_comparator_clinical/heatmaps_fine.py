"""As `heatmaps`, with the comparators fit on the 26 fine cell types (cccRISE
does not use cell types and is unchanged)."""

from .. import common, style
from .heatmaps import plot_granularity


def plot(fig):
    plot_granularity(fig, "fine")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
