"""Per-cell sender vs. receiver weight for the granularity CP component, with
the shared cell type legend for this row."""

from .. import common
from ._common import DATASET, GRANULARITY_COMPONENT, HIGHLIGHT_CELLTYPE

AXES = ("square",)


def plot(axs):
    common.plot_component_weight_scatter(
        DATASET, GRANULARITY_COMPONENT, axs[0], highlight=(HIGHLIGHT_CELLTYPE,)
    )


def makeFigure():
    return common.standalone_fixed(__name__)
