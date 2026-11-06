"""Figure 5D: how STAC arranges the mixed programs' pairs."""

from .. import common
from ._recovery import pair_columns

AXES = ("fig4_columns",)


def plot(axs):
    pair_columns(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
