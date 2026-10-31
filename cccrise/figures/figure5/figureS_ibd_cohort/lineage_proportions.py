"""Each IBD biopsy's lineage proportions by inflammation status, with
each lineage's Kruskal-Wallis q across the three statuses
(Benjamini-Hochberg corrected across lineages)."""

from ... import common
from ._common import plot_level

AXES = ("full",)


def plot(axs):
    plot_level(axs[0], "lineage")


def makeFigure():
    return common.standalone_fixed(__name__)
