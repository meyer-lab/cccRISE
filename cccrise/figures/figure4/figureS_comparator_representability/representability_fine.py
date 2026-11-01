"""How much of each cccRISE program a (sample x cell type) pseudobulk could hold (fine labels)."""

from ... import common
from .._panels import representability

AXES = ("square",)


def plot(axs):
    representability(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
