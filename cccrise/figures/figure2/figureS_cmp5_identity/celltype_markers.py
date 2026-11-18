"""Fraction of cells detecting and mean normalized expression of canonical
markers of each annotated broad cell type in the sender, receiver, and rest
groups of CPD component 6."""

from ... import common
from .. import _common as S

AXES = ("strip",)


def plot(axs):
    S.plot_markers(axs[0], S.CELLTYPE_MARKERS, show_legend=True)


def makeFigure():
    return common.standalone_fixed(__name__)
