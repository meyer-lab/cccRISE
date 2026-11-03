"""What published Tensor-cell2cell's construction drops from the masked tensor (coarse and fine labels)."""

from ... import common
from .._panels import inner_loss

AXES = ("square",)


def plot(axs):
    inner_loss(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
