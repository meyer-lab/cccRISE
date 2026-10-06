"""The cells with high projections (above each projection's
`EIGENSTATE_PERCENTILE`) onto the `N_EIGENSTATES` RISE eigenstates the
granularity component weighs most (largest summed squared sender and receiver
loading), on the component's per-cell sender vs. receiver weight: how the
eigenstates behind Figure 2's eigenstate scatter make up the weights."""

import numpy as np

from .. import common, style

N_EIGENSTATES = 3
EIGENSTATE_PERCENTILE = 99.8
EIGENSTATE_COLORS = [
    style.COLORS["receiver"],
    style.COLORS["sender"],
    style.CATEGORICAL[4],
]
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(common.GRANULARITY_DATASET)
    eigenstates = top_eigenstates(adata)
    projections = adata.obsm["projections"][:, eigenstates - 1]
    high = projections > np.percentile(projections, EIGENSTATE_PERCENTILE, axis=0)
    single = high.sum(axis=1) == 1
    groups = {f"Eig. {e}": high[:, i] & single for i, e in enumerate(eigenstates)}
    groups["Multiple"] = high.sum(axis=1) > 1
    colors = dict(zip(groups, [*EIGENSTATE_COLORS, style.COLORS["ink"]], strict=True))
    common.plot_weight_scatter(ax, groups, colors)
    common.legend(
        ax,
        "right",
        title=f"High projection\n(> p{EIGENSTATE_PERCENTILE:g})",
        markerscale=2,
    )


def top_eigenstates(adata) -> np.ndarray:
    """1-based indices of the `N_EIGENSTATES` eigenstates with the largest
    summed squared sender (B) and receiver (C) loading on the component."""
    B, C = (np.asarray(adata.uns[k])[:, common.GRANULARITY_COMPONENT - 1] for k in "BC")
    return np.argsort(-(B**2 + C**2))[:N_EIGENSTATES] + 1


def makeFigure():
    return common.standalone_fixed(__name__)
