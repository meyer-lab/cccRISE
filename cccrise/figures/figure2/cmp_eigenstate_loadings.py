"""The granularity CP component's sender and receiver loadings on each RISE
eigenstate."""

from ...runners.datasets import DATASETS
from .. import common, style
from ._shared import COMPONENT, DATASET


def plot(fig):
    ax = fig.subplots()
    adata = common.load_factors_to_anndata(DATASETS[DATASET].factors_path())
    common.plot_component_eigenstate_loadings(adata, COMPONENT, ax)
    ax.set_title(f"Cmp. {COMPONENT} eigenstate loadings")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
