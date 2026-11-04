"""Which cccRISE programs published and masked Tensor-cell2cell contain (fine labels)."""

from ... import common
from .._panels import reach
from ._common import METHODS

AXES = ("square",)


def plot(axs):
    reach(axs, "fine", methods=METHODS)


def makeFigure():
    return common.standalone_fixed(__name__)
