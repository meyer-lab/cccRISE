"""Weighted score percentile of every column."""

from ... import common
from .._nichenet import prior_shuffle_strip

AXES = ("mid",)


def plot(axs):
    prior_shuffle_strip(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
