"""Per-cell sender vs. receiver weight on the granularity component, colored
by the label-free sender and receiver groups (`S.split_cells`): the
cells passing either cutoff (`TAU` x each weight's maximum), split by the line
of equal relative weight."""

from .. import common, style
from . import _common as S

LEGEND_MARKER_SIZE = 12  # pt^2, alike for the small grey and larger group points
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    sender, receiver = S.component_weights()
    groups = S.split_cells(sender, receiver)
    S.plot_weight_scatter(
        ax,
        {g.capitalize(): mask for g, mask in groups.items()},
        {g.capitalize(): color for g, color in S.GROUP_COLORS.items()},
        rest_label="Rest",
    )
    _plot_boundary(ax, sender.max(), receiver.max())
    handles, labels = ax.get_legend_handles_labels()  # rest first: drawn beneath
    legend = common.legend(
        ax,
        "below",
        handles=handles[1:] + handles[:1],
        labels=labels[1:] + labels[:1],
        ncols=2,
        columnspacing=0.8,
        handletextpad=0.2,
    )
    legend.set_in_layout(False)  # hangs into the row's spare height below
    for handle in legend.legend_handles:
        handle.set_sizes([LEGEND_MARKER_SIZE])


def _plot_boundary(ax, sender_max: float, receiver_max: float) -> None:
    """The edges of the region passing neither cutoff, and the equal relative
    weight line beyond it."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    corner_x, corner_y = S.GROUP_TAU * sender_max, S.GROUP_TAU * receiver_max
    reach = max(x1 / sender_max, y1 / receiver_max)
    for xs, ys in (
        ((x0, corner_x), (corner_y, corner_y)),
        ((corner_x, corner_x), (y0, corner_y)),
        ((corner_x, reach * sender_max), (corner_y, reach * receiver_max)),
    ):
        ax.plot(
            xs,
            ys,
            color=style.COLORS["reference"],
            linestyle=style.LINE_STYLE_REFERENCE,
            linewidth=style.LINE_WIDTH_THIN,
        )
    ax.set(xlim=(x0, x1), ylim=(y0, y1))


def makeFigure():
    return common.standalone_fixed(__name__)
