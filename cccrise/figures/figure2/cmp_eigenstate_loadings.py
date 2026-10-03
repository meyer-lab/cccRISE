"""The granularity CP component's sender and receiver loadings on each RISE
eigenstate."""

from ...runners.datasets import DATASETS
from .. import common
from ._common import (
    DATASET,
    GRANULARITY_COMPONENT,
    RECEIVER_EIGENSTATE,
    SENDER_EIGENSTATE,
)

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    adata = common.load_factors_to_anndata(DATASETS[DATASET].factors_path())
    common.plot_component_eigenstate_loadings(
        adata, GRANULARITY_COMPONENT, ax, (SENDER_EIGENSTATE, RECEIVER_EIGENSTATE)
    )
    ax.set_ylabel(f"Cmp. {GRANULARITY_COMPONENT} loading")


def makeFigure():
    return common.standalone_fixed(__name__)
