"""Supplementary figure: how the granularity component's sender and receiver
weights relate to the RISE eigenstates and cell types, how the label-free
sender and receiver groups (Figure 2) depend on the cutoff, and the genes
they express."""

from matplotlib.transforms import Bbox

from ... import common, style
from . import (
    celltype_auroc,
    celltype_markers,
    cmp_eigenstate_loadings,
    de_genes,
    eigenstate_scatter,
    eigenstate_split,
    sender_vs_receiver,
    tau_sweep,
)

LAYOUT = [
    [eigenstate_scatter, cmp_eigenstate_loadings],
    [eigenstate_split, tau_sweep],
    [sender_vs_receiver, celltype_auroc],
    [[celltype_markers, de_genes]],
]


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    _share_dotplot_legends(fig)
    return fig


def _share_dotplot_legends(fig) -> None:
    """Stretch `de_genes`' colorbar up through the `celltype_markers` dot plot
    stacked above it, and center its size legend on both, which share them."""
    upper = common.panel_axes(fig, celltype_markers.__name__.rsplit(".", 1)[-1])
    lower = common.panel_axes(fig, de_genes.__name__.rsplit(".", 1)[-1])
    top, bottom = upper.get_position().y1, lower.get_position().y0
    width = fig.get_figwidth()
    spanning = Bbox.from_bounds(
        lower.get_position().x1 + style.COLORBAR_PAD / width,
        bottom,
        style.COLORBAR_WIDTH / width,
        top - bottom,
    )
    (cax,) = lower.child_axes
    cax.set_axes_locator(lambda ax, renderer: spanning)
    legend = lower.get_legend()
    assert legend is not None
    anchor_x = legend.get_bbox_to_anchor().x0 / fig.bbox.width
    legend.set_bbox_to_anchor((anchor_x, (top + bottom) / 2), transform=fig.transFigure)
