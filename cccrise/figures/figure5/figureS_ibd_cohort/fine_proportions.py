"""Each IBD biopsy's fine cell type proportions by inflammation status, with
each cell type's Kruskal-Wallis q across the three statuses
(Benjamini-Hochberg corrected across cell types)."""

from ... import common
from ._common import plot_level

AXES = ("full",)


def plot(axs):
    plot_level(axs[0], "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
