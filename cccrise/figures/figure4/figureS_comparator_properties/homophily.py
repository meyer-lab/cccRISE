"""Each component's homophilic share of squared LR loading."""

from ... import common
from .._strips import homophilic_share

AXES = ("half",)


def plot(axs):
    homophilic_share(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
