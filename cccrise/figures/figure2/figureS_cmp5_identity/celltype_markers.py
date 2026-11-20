"""Canonical markers of each annotated broad cell type
(`S.CELLTYPE_MARKERS`) per sender, receiver, and rest group."""

from ... import common, style
from .. import _common as S

AXES = ("strip",)


def plot(axs):
    S.plot_markers(
        axs[0],
        S.CELLTYPE_MARKERS,
        show_legend=True,
        fontsize=style.FONT_SIZE_DENSE,
        key_below=True,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
