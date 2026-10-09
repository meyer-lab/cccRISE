"""Granularity (b): the fraction of each fine type's cells active in the
cccRISE component, against the comparator's weight (`granularity`)."""

from .. import common, style
from . import granularity


def plot(fig):
    axes = granularity.plot(fig, granularity.active_fraction)
    for ax in axes[:, 0]:
        ax.set_ylabel("cells active")


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 150 * style.MM))
