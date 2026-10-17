"""Supplementary figure: the canonical ALAD RISE factor matrices."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from ... import common, style
from . import factors, genes

LAYOUT = [[[factors, genes]]]  # one stack, so the two heatmaps left-align


def makeFigure():
    fig = common.compose_fixed(LAYOUT)
    top, bottom = common.panel_axes(fig, "factors"), common.panel_axes(fig, "genes")
    _close_gap(fig, top, bottom)
    _span_colorbar(fig, top, bottom)
    return fig


def _close_gap(fig: Figure, top: plt.Axes, bottom: plt.Axes) -> None:
    """Shrink the gap the stack leaves above `bottom` (for its letter) to
    `style.ROW_GAP`, cutting it from the figure's height, and letter `bottom`
    in its left margin beside its top edge instead."""
    _, height = fig.get_size_inches()
    upper, lower = top.get_position(), bottom.get_position()
    cut = (upper.y0 - lower.y1) * height - style.ROW_GAP
    new_height = height - cut

    def inches_up(y: float, shift: float = 0) -> float:  # fraction -> new fraction
        return (y * height - shift) / new_height

    top.set_position(
        (upper.x0, inches_up(upper.y0, cut), upper.width, inches_up(upper.height))
    )
    bottom.set_position(
        (lower.x0, inches_up(lower.y0), lower.width, inches_up(lower.height))
    )
    letters = {text.get_gid(): text for text in fig.texts}
    letters["factors"].set_y(inches_up(letters["factors"].get_position()[1], cut))
    letters["genes"].set_y(inches_up(lower.y1))
    fig.set_size_inches(fig.get_size_inches()[0], new_height)


def _span_colorbar(fig: Figure, top: plt.Axes, bottom: plt.Axes) -> None:
    """Stretch `bottom`'s inset colorbar (`common.colorbar`) from its bottom
    edge to `top`'s top."""
    (cax,) = bottom.child_axes
    lower, upper = bottom.get_position(), top.get_position()
    width = fig.get_figwidth()
    span = Bbox.from_bounds(
        lower.x1 + style.COLORBAR_PAD / width,
        lower.y0,
        style.COLORBAR_WIDTH / width,
        upper.y1 - lower.y0,
    )
    cax.set_axes_locator(lambda *_: span)  # figure coords
