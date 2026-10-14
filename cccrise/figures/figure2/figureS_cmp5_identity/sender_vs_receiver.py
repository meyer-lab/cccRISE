"""Genes differing between the sender and receiver groups
(`_common.plot_volcano`), labeling `CURATED_DE_MARKERS`."""

from ... import common
from . import _common as V

AXES = ("square",)

# Epithelium-specific, coordinated genes hand-picked from all the
# differentially expressed ones
CURATED_DE_MARKERS = [
    "C20orf85",  # ciliated
    "RSPH1",  # ciliated
    "SNTN",  # ciliated
    "TPPP3",  # ciliated
    "CCDC146",  # ciliated
    "CDHR3",  # ciliated
    "CCDC78",  # ciliated
    "CXCL17",  # secretory
    "SERPINB3",  # secretory
    "KRT19",  # secretory
    "KRT7",  # alveolar
]


def plot(axs):
    # V.plot_volcano(axs[0], "sender", "receiver", V.top_genes("sender", "receiver"))
    V.plot_volcano(axs[0], "sender", "receiver", CURATED_DE_MARKERS)


def makeFigure():
    return common.standalone_fixed(__name__)
