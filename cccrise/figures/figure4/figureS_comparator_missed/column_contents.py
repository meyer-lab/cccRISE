"""LR contents of the columns no cccRISE program carries, both cohorts."""

from ... import common
from .._missed import contents_matrix

AXES = ("missed_matrix",)


def plot(axs):
    contents_matrix(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
