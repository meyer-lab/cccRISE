"""Figure 4E on IBD: each comparator LR column's containment in cccRISE's LR
span."""

from ... import common
from .._columns import columns_in_cccrise

AXES = ("rect",)


def plot(axs):
    columns_in_cccrise(axs, "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
