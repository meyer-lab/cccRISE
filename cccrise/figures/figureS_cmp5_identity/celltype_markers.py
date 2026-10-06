"""Canonical markers of each annotated broad cell type
(`common.CELLTYPE_MARKERS`) per sender, receiver, and rest group."""

from .. import common

AXES = ("strip",)


def plot(axs):
    common.plot_markers(axs[0], common.CELLTYPE_MARKERS)


def makeFigure():
    return common.standalone_fixed(__name__)
