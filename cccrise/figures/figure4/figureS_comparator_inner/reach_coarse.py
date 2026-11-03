"""Which cccRISE programs published and masked Tensor-cell2cell contain (coarse labels)."""

from ... import common
from .._panels import reach
from ._common import METHODS

AXES = ("square",)


def plot(axs):
    reach(axs, "coarse", methods=METHODS)


def makeFigure():
    return common.standalone_fixed(__name__)
