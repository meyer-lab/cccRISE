"""The share of each cccRISE program's weight in the cell types published Tensor-cell2cell drops (fine labels)."""

from ... import common
from .._panels import inner_program_loss

AXES = ("square",)


def plot(axs):
    inner_program_loss(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
