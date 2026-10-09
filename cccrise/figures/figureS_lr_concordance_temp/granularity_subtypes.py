"""Granularity (a): cccRISE's mean loading per fine type against the
comparator's coarse weight (`granularity`)."""

from .. import common, style
from . import granularity


def plot(fig):
    axes = granularity.plot(fig, granularity.mean_loading)
    for ax in axes[:, 0]:
        ax.set_ylabel("loading / max")


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 150 * style.MM))
