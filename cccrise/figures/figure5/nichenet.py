"""The IBD weight-averaged NicheNet target-score percentile per CP component
against shuffled priors, as `figure2.nichenet` draws ALAD's, with receivers
chosen among fine cell types (`cccrise-nichenet --receiver-celltype-column
celltype`); the legend sits beside the axes."""

from .. import common, style
from ..figure2.nichenet import AXES, draw  # noqa: F401  (AXES: this panel's)
from ._common import DATASET


def plot(axs):
    draw(axs, DATASET)
    (ax,) = axs
    legend = ax.get_legend()
    handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
    legend.remove()
    common.legend(
        ax, "right", handles=handles, labels=labels, fontsize=style.FONT_SIZE_DENSE
    )


def makeFigure():
    return common.standalone_fixed(__name__)
