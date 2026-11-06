"""Figure 4F: BTC-EGFR in STAC and in cccRISE."""

from .. import common
from ._flows import flow_alad

AXES = ("fig4_flow_alad",)


def plot(axs):
    flow_alad(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
