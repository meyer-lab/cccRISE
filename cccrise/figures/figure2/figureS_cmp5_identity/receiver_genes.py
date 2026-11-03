"""Every gene's loading on the RISE eigenstate behind the granularity
component's receiver weights (pf2_C) against its rank: the S-shaped curve,
the genes up to its elbow named in a column over the flat tail. The elbow is
the Kneedle elbow (`common.kneedle_elbow`) of the genes loading at least
`ELBOW_WINDOW` x the top gene's: the head declines gradually, so over the
whole curve the elbow falls much later (rank ~90)."""

import numpy as np
import pandas as pd

from ....factors import load_factors_netcdf
from ....runners.datasets import DATASETS
from ... import common, style
from ...figure3._common import label_ranked_head, plot_ranked_loadings
from .. import _common as S

EIGENSTATE = S.GRANULARITY_RECEIVER_EIGENSTATE
ELBOW_WINDOW = 0.25
LABEL_RANK = 3000  # labels' column, above the flat tail
LABEL_FLOOR = 0.1  # labels spread from the top loading down to this fraction
AXES = ("rect_tall",)


def plot(axs):
    (ax,) = axs
    factor = load_factors_netcdf(DATASETS[S.GRANULARITY_DATASET].factors_path())
    loadings = factor["pf2_C"].values[:, EIGENSTATE - 1].astype(float)
    genes = np.asarray(factor["pf2_C"]["gene"].values).astype(str)
    head = np.sort(loadings)[::-1]
    head = head[head >= ELBOW_WINDOW * head[0]]
    n_top = common.kneedle_elbow(np.arange(1, len(head) + 1), head)
    plot_ranked_loadings(ax, loadings, n_top, point_size=style.POINT_SIZE)
    ax.set_xlim(0, len(loadings) + 1)
    ax.set(xlabel="Gene rank", ylabel=f"Eigenstate {EIGENSTATE}\ngene loading")
    label_ranked_head(ax, loadings, genes, n_top, LABEL_RANK, LABEL_FLOOR)
    order = np.argsort(-loadings)
    common.source_data(
        f"eigenstate_{EIGENSTATE}_gene_loadings_ranked",
        pd.DataFrame(
            {
                "Rank": np.arange(1, len(order) + 1),
                "Gene": genes[order],
                "Loading": loadings[order],
                "Named": np.arange(len(order)) < n_top,
            }
        ),
        f"Every gene's loading on RISE eigenstate {EIGENSTATE} (gene factor C), "
        "the eigenstate with the largest receiver loading on CP component "
        f"{S.GRANULARITY_COMPONENT}, by rank; 'Named' marks the genes up to "
        f"the curve's elbow ({n_top}), labeled in the plot.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
