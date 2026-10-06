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
    (ax,) = axs
    loadings = S.rise_factors()["pf2_C"].values.astype(float)
    loadings = loadings / np.abs(loadings).max(axis=0)
    n_top = round(TOP_GENE_FRACTION * len(loadings))
    shown = loadings[np.argsort(-np.abs(loadings).max(axis=1))[:n_top]]
    image = common.plot_factor_heatmap(
        ax, shown[common.dominant_order(shown)], S.XLABEL
    )
    ax.set_yticks([])
    ax.set_ylabel(f"Top {TOP_GENE_FRACTION:.0%} genes (n={n_top})")
    common.colorbar(ax, image, "Loading / max |loading|")


def makeFigure():
    return common.standalone_fixed(__name__)
