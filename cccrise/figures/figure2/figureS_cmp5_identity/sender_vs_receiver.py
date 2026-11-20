"""Differential expression between the sender and receiver groups of CPD
component 6 (Wilcoxon rank-sum, Benjamini-Hochberg q), curated markers labeled
with their epithelial subtype.

Genes differing between the sender and receiver groups
(`_common.plot_volcano`), labeling `CURATED_DE_MARKERS`, each tagged with its
epithelial subtype."""

from ... import common
from . import _common as V

AXES = ("rect",)  # wide enough for the tagged labels

# Epithelium-specific, coordinated genes hand-picked from all the
# differentially expressed ones, and their epithelial subtype
CURATED_DE_MARKERS = {
    "C20orf85": "ciliated",
    "RSPH1": "ciliated",
    "SNTN": "ciliated",
    "TPPP3": "ciliated",
    "CCDC146": "ciliated",
    "CDHR3": "ciliated",
    "CCDC78": "ciliated",
    "SERPINB3": "basal",
    "CXCL17": "secretory",
    "KRT19": "secretory",
    "KRT7": "secretory",
}
SUBTYPE_TAGS = {"ciliated": "C", "basal": "B", "secretory": "S"}
TAG_ROOM = 1.9  # x range widening for the tagged labels


def plot(axs):
    x_limit, y_limit = V.limits(*V.volcano_data("sender", "receiver")[1:])
    V.plot_volcano(
        axs[0],
        "sender",
        "receiver",
        {g: f"{g} ({SUBTYPE_TAGS[s]})" for g, s in CURATED_DE_MARKERS.items()},
        (x_limit * TAG_ROOM, y_limit),
    )
    V.legend(axs[0], ("sender", "receiver"))
    V.save_volcano("sender", "receiver")


def makeFigure():
    return common.standalone_fixed(__name__)
