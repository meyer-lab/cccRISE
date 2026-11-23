"""Containment of each IBD CCC-RISE component's LR pair loadings in each
comparator fit's LR column span.

Each IBD component's containment in every comparator fit's LR span
(`span_containment.containment_grid`; ALAD's is Figure 5B)."""

from ... import common
from .._recovery import colorbar_beneath
from ..span_containment import LABEL, containment_grid

AXES = ("split_ibd_grid",)
JOIN_PREVIOUS = True  # beside A, sharing its component axis, as 5B


def plot(axs):
    containment_grid(axs[0], ("ibd",), bars=False)
    axs[0].tick_params(labelleft=False)  # the cos^2 grid's labels serve both
    colorbar_beneath(axs[0], LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
