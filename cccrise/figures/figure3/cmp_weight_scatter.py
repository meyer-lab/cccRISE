"""Per-cell sender vs. receiver weight on the outcome-associated CP
component, its predominant cell types enlarged, with the cell type legend
shared by this figure's cell type panels."""

from .. import common
from ._common import COMPONENT, COMPONENT_CELLTYPES, DATASET

AXES = ("square",)


def plot(axs):
    common.plot_component_weight_scatter(
        DATASET, COMPONENT, axs[0], highlight=COMPONENT_CELLTYPES
    )


def makeFigure():
    return common.standalone_fixed(__name__)
