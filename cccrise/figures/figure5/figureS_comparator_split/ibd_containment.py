"""Containment of each IBD CCC-RISE component's LR pair loadings in each
comparator fit's LR column span.

Each IBD component's containment in every comparator fit's LR span
(`span_containment.containment_grid`; ALAD's is Figure 5B)."""

from ... import common
from .._recovery import colorbar_below
from ..span_containment import LABEL, containment_grid

AXES = ("split_ibd_grid",)


def plot(axs):
    containment_grid(axs[0], ("ibd",), bars=False)
    colorbar_below(axs[0], LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
