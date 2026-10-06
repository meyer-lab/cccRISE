"""Label-free identity of the granularity component's sender and receiver
groups: canonical pan-epithelial markers (`S.CELLTYPE_MARKERS`) and airway
epithelial subtype markers (`SUBTYPE_MARKERS`, from the same atlas), and the genes of the component's top LR pairs, per group."""

from .. import common, style
from . import _common as S

SUBTYPE_MARKERS = {
    "Ciliated": ["FOXJ1", "CAPS", "TPPP3"],
    "Basal": ["KRT5", "TP63", "KRT15"],
    "Secretory": ["SCGB1A1", "SCGB3A1", "MUC5B"],
}
AXES = ("strip",)


def plot(axs):
    S.plot_markers(
        axs[0],
        {"Pan-epithelial": S.CELLTYPE_MARKERS["Epithelial cells"]}
        | SUBTYPE_MARKERS
        | {"Top LR pairs": S.top_pair_genes()},
        show_legend=True,
        fontsize=style.FONT_SIZE_TICK,  # as the group labels
    )


def makeFigure():
    return common.standalone_fixed(__name__)
