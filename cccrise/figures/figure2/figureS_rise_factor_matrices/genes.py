"""The RISE gene factor (C), each component divided by its largest magnitude
(keeping sign). Only the `TOP_GENE_FRACTION` of genes with the largest such
loading on any component are shown, ordered by their most heavily loaded
component, then by that loading."""

import numpy as np

from ... import common
from . import _common as S

TOP_GENE_FRACTION = 0.1
AXES = ("tall",)


def plot(axs):
    draw(axs, S.DATASET)


def draw(axs, dataset: str) -> None:
    (ax,) = axs
    factor = S.rise_factors(dataset)["pf2_C"]
    loadings = factor.values.astype(float)
    loadings = loadings / np.abs(loadings).max(axis=0)
    n_top = round(TOP_GENE_FRACTION * len(loadings))
    top = np.argsort(-np.abs(loadings).max(axis=1))[:n_top]
    rows = top[common.dominant_order(loadings[top])]
    image = common.plot_factor_heatmap(ax, loadings[rows], S.XLABEL)
    common.source_data(
        "rise_gene_factor_top_genes",
        S.normalized_table(factor["gene"].values[rows], loadings[rows], "Gene"),
        f"RISE gene factor (C) of the top {TOP_GENE_FRACTION:.0%} of genes by "
        "largest normalized |loading| on any component (rows top to bottom as "
        "plotted): each gene's loading divided by the component's largest "
        "|loading| over all genes (sign kept).",
    )
    ax.set_yticks([])
    ax.set_ylabel(f"Top {TOP_GENE_FRACTION:.0%} genes (n={n_top})")
    common.colorbar(ax, image, "Loading / max |loading|")


def makeFigure():
    return common.standalone_fixed(__name__)
