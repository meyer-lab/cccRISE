"""Every LR pair's loading on the outcome-associated CP component (cp_D)
against its rank, on a log rank axis so the head is not lost to the flat
tail; the component's `N_TOP` most heavily weighted pairs (the pairs tested
for co-variation in `coordinated_changes`) marked and named."""

import numpy as np
import pandas as pd

from ...factors import load_factors_netcdf
from .. import common, style
from ._common import COMPONENT, SPEC, label_ranked_head, plot_ranked_loadings
from .coordinated_changes import N_TOP

# Labels start in a column at this rank, in the empty space above the flat
# tail, spread evenly from the top loading down to this fraction of it
LABEL_RANK = 20
LABEL_FLOOR = 0.15
AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    factor = load_factors_netcdf(SPEC.factors_path())["cp_D"]
    loadings = factor.values[:, COMPONENT - 1].astype(float)
    names = np.asarray(factor["lr_pair"].values).astype(str)
    plot_ranked_loadings(ax, loadings, N_TOP, point_size=style.POINT_SIZE)
    ax.set_xscale("log")
    ax.set_xlim(0.8, len(loadings) * 1.2)
    ax.set(xlabel="LR pair rank", ylabel=f"Cmp. {COMPONENT}\nLR pair loading")

    order = np.argsort(-loadings)
    ranks = np.arange(1, len(order) + 1)
    common.source_data(
        "cp_lr_pair_loadings_ranked",
        pd.DataFrame(
            {
                "Rank": ranks,
                "LR pair": names[order],
                f"Cmp {COMPONENT} loading": loadings[order],
                "Named": ranks <= N_TOP,
            }
        ),
        f"Every LR pair's loading on CP component {COMPONENT} (cp_D), by rank; "
        f"'Named' marks the {N_TOP} most heavily weighted, labeled in the plot.",
    )
    label_ranked_head(ax, loadings, names, N_TOP, LABEL_RANK, LABEL_FLOOR)


def makeFigure():
    return common.standalone_fixed(__name__)
