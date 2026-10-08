"""Genes distinguishing the sender (left) and receiver (right) group from the
rest of the cells (`V.plot_volcano`), on shared axes."""

import numpy as np

from ... import common
from .. import _common as S
from . import _volcano as V

AXES = ("square", "square")


def plot(axs):
    # Shared limits, so the two groups' volcanoes compare directly
    lims = np.max(
        [V.limits(*V.volcano_data(g, "rest")[1:]) for g in S.DE_GROUPS], axis=0
    )
    for ax, group in zip(axs, S.DE_GROUPS, strict=True):
        V.plot_volcano(ax, group, "rest", lims)


def makeFigure():
    return common.standalone_fixed(__name__)
