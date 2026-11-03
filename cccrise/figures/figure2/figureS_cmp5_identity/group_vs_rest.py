"""Genes distinguishing the sender (left) and receiver (right) group from
every other cell (`V.plot_volcano`), on shared axes, labeling the canonical
pan-epithelial markers (`S.CELLTYPE_MARKERS`), with the color key shared with
the sender vs. receiver volcano below the receiver's."""

import numpy as np

from ... import common
from .. import _common as S
from . import _common as V

AXES = ("square", "square")


def plot(axs):
    # Shared limits, so the two groups' volcanoes compare directly
    lims = np.max(
        [V.limits(*V.volcano_data(g, "rest")[1:]) for g in S.DE_GROUPS], axis=0
    )
    markers = S.CELLTYPE_MARKERS["Epithelial cells"]
    for part, (ax, group) in enumerate(zip(axs, S.DE_GROUPS, strict=True)):
        V.plot_volcano(ax, group, "rest", markers, lims)
        V.save_volcano(group, "rest", part)
    V.legend(axs[1])


def makeFigure():
    return common.standalone_fixed(__name__)
