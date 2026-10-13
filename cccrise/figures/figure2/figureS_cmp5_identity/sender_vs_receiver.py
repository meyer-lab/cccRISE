"""Genes differing between the sender and receiver groups
(`_common.plot_volcano`), labeling the most enriched in each
(`_common.top_genes`)."""

from ... import common
from . import _common as V

AXES = ("square",)

LABELED = [
    "C20orf85",  # ciliated
    "RSPH1",  # ciliated
    "SNTN",  # ciliated
    "TPPP3",  # ciliated
    "CCDC146",  # ciliated
    "CDHR3",  # ciliated
    "CCDC78",  # ciliated
    "CXCL17",  # secratory
    "SERPINB3",  # secratory
    "KRT19",  # secratory
    "KRT7",  # alveolar
]


def plot(axs):
    # V.plot_volcano(axs[0], "sender", "receiver", V.top_genes("sender", "receiver"))
    V.plot_volcano(axs[0], "sender", "receiver", LABELED)


def makeFigure():
    return common.standalone_fixed(__name__)
