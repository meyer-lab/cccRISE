"""How much of each cccRISE program a (sample x cell type) pseudobulk could hold (coarse labels)."""

from ... import common
from .._panels import representability

AXES = ("square",)


def plot(axs):
    representability(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
