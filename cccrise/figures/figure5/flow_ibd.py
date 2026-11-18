"""IBD cells assigned PYY and GCG communication by STAC (one EE label) and by
CCC-RISE components 3 and 4 (EE cells by hormone state past each sender cutoff,
and EC and tuft cells past each receiver cutoff)."""

from .. import common
from ._flows import flow_ibd

AXES = ("fig5_flow_ibd",)


def plot(axs):
    flow_ibd(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
