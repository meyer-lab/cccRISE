"""Supplementary figure: the hormone programs' communication axes. The
eigenstates C3 and C4 load on belong to EE, EC and tuft cells: both programs
send from EE eigenstates, C3 to an EC eigenstate and C4 to tuft eigenstates.
Where Figure 5's zoomed EE/EC/tuft map sits among all cells, beside the
programs' receiver weights and their top pairs' receptors on that map, and
the villus-like hormones on it. The programs use far fewer sender eigenstates
than the cells they come from, and the two hormone states' markers. Figure
5E-H show the two programs splitting the EE cells expressing their ligands."""

from matplotlib.transforms import Bbox

from ... import common, style
from . import (
    cell_map,
    eigenstate_association,
    eigenstate_composition,
    eigenstate_loadings,
    eigenstate_use,
    hormone_maps,
    hormone_state_genes,
    receiver_maps,
    receptor_maps,
)

LAYOUT = [
    [eigenstate_loadings, eigenstate_association, eigenstate_composition],
    [cell_map, [receiver_maps], [receptor_maps]],
    [hormone_maps],
    [eigenstate_use],
    [hormone_state_genes],
]
#: Stacked map pairs whose one colorbar spans both maps
SPANNED = (receiver_maps, receptor_maps)


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    for module in SPANNED:
        span_colorbar(fig, module.__name__.rsplit(".", 1)[-1])
    return fig


def span_colorbar(fig, panel: str) -> None:
    """Stretch the colorbar beside panel module `panel`'s last (bottom) axes
    up to its first's top."""
    axes = [ax for ax in fig.axes if ax.get_label() == panel]
    top, bottom = axes[0].get_position(), axes[-1].get_position()
    (cax,) = axes[-1].child_axes
    span = Bbox.from_bounds(
        bottom.x1 + style.COLORBAR_PAD / fig.get_figwidth(),
        bottom.y0,
        style.COLORBAR_WIDTH / fig.get_figwidth(),
        top.y1 - bottom.y0,
    )
    cax.set_axes_locator(lambda *_: span)  # figure coords
