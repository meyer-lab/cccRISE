"""Label-free identity of the granularity component's sender and receiver
groups: canonical pan-epithelial markers (`S.CELLTYPE_MARKERS`) and airway
epithelial subtype markers (`SUBTYPE_MARKERS`, from the same atlas), and the genes of the component's top LR pairs, per group."""

from .. import common, style
from . import _common as S

# Differentially expressed markers
SUBTYPE_MARKERS = {
    "Ciliated": ["C20orf85", "RSPH1", "SNTN", "TPPP3", "CCDC146", "CDHR3"],
    "Secretory": ["CXCL17", "SERPINB3", "KRT19"],
    "Alveolar": ["KRT7"],
}
AXES = ("strip",)


def plot(axs):
    S.plot_markers(
        axs[0],
        SUBTYPE_MARKERS | {"Top LR pairs": S.top_pair_genes()},
        show_legend=True,
        fontsize=style.FONT_SIZE_TICK,  # as the group labels
    )


def makeFigure():
    return common.standalone_fixed(__name__)
