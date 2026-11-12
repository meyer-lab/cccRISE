"""Each IBD component's containment in every comparator fit's LR span
(`span_containment.containment_grid`; ALAD's is Figure 5B)."""

from ... import common
from .._recovery import colorbar_beneath
from ..span_containment import LABEL, containment_grid

AXES = ("ibd_grid",)


def plot(axs):
    containment_grid(axs[0], ("ibd",))
    colorbar_beneath(axs[0], LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
