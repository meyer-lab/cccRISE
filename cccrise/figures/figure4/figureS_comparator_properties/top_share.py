"""Each component's top pair's share of squared LR loading."""

from ... import common
from .._strips import top_share

AXES = ("half",)


def plot(axs):
    top_share(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
