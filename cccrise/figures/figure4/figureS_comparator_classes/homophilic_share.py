"""Homophilic share of the columns the earlier rules leave."""

from ... import common
from .._classes import homophily

AXES = ("half",)


def plot(axs):
    homophily(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
