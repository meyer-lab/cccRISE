"""Figure 4G: each component's strongest outcome association."""

from .. import common
from ._columns import strongest_association

AXES = ("rect",)


def plot(axs):
    strongest_association(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
