"""Genes differing between the sender and receiver groups
(`_common.plot_volcano`), labeling the most enriched in each
(`_common.top_genes`)."""

from ... import common
from . import _common as V

AXES = ("square",)


def plot(axs):
    V.plot_volcano(axs[0], "sender", "receiver", V.top_genes("sender", "receiver"))


def makeFigure():
    return common.standalone_fixed(__name__)
