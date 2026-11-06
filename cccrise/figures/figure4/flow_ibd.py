"""Figure 4G: PYY and GCG in STAC and in cccRISE."""

from .. import common
from ._flows import flow_ibd

AXES = ("fig4_flow_ibd",)


def plot(axs):
    flow_ibd(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
