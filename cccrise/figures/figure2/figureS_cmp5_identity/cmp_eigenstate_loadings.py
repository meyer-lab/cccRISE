"""The granularity CP component's sender and receiver loadings on each RISE
eigenstate."""

from ....runners.datasets import DATASETS
from ... import common
from .. import _common as S

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    adata = common.load_factors_to_anndata(
        DATASETS[S.GRANULARITY_DATASET].factors_path()
    )
    common.plot_component_eigenstate_loadings(
        adata,
        S.GRANULARITY_COMPONENT,
        ax,
        (S.GRANULARITY_SENDER_EIGENSTATE, S.GRANULARITY_RECEIVER_EIGENSTATE),
    )
    ax.set_ylabel(f"Cmp. {S.GRANULARITY_COMPONENT} loading")


def makeFigure():
    return common.standalone_fixed(__name__)
