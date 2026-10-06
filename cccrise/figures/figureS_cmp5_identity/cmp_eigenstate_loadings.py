"""The granularity CP component's sender and receiver loadings on each RISE
eigenstate."""

from ...runners.datasets import DATASETS
from .. import common

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    adata = common.load_factors_to_anndata(
        DATASETS[common.GRANULARITY_DATASET].factors_path()
    )
    common.plot_component_eigenstate_loadings(
        adata,
        common.GRANULARITY_COMPONENT,
        ax,
        (common.GRANULARITY_SENDER_EIGENSTATE, common.GRANULARITY_RECEIVER_EIGENSTATE),
    )
    ax.set_ylabel(f"Cmp. {common.GRANULARITY_COMPONENT} loading")


def makeFigure():
    return common.standalone_fixed(__name__)
