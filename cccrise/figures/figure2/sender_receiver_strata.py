"""Per-cell sender vs. receiver weight on the granularity component, colored
by the label-free sender and receiver groups (`S.split_cells`): the
cells passing either cutoff (`TAU` x each weight's maximum), split by the line
of equal relative weight."""

from .. import common
from . import _common as S

LEGEND_MARKER_SIZE = 12  # pt^2, alike for the small grey and larger group points
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    sender, receiver = S.component_weights()
    groups = S.split_cells(sender, receiver)
    S.plot_weight_scatter(
        ax,
        sender,
        receiver,
        {g.capitalize(): mask for g, mask in groups.items()},
        {g.capitalize(): color for g, color in S.GROUP_COLORS.items()},
        rest_label="Rest",
    )
    ax.set(
        xlabel=f"Cmp. {S.GRANULARITY_COMPONENT} sender weight",
        ylabel=f"Cmp. {S.GRANULARITY_COMPONENT} receiver weight",
    )
    S.plot_split_boundary(ax, sender.max(), receiver.max())
    common.source_data("cell_weights_groups", S.cell_table(), S.CELL_TABLE_DESCRIPTION)
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


def makeFigure():
    return common.standalone_fixed(__name__)
