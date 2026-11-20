"""ALAD epithelial cells assigned BTC-EGFR communication by STAC (one
epithelial label) and by CCC-RISE component 6 (the cells past its sender and
receiver cutoffs)."""

from .. import common
from ._flows import flow_alad

AXES = ("fig5_flow_alad",)


def plot(axs):
    flow_alad(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
