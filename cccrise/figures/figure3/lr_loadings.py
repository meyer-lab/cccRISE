"""Every LR pair's loading on the outcome-associated CP component (cp_D)
against its rank, on a log rank axis so the head is not lost to the flat
tail; the component's `N_TOP` most heavily weighted pairs (the pairs tested
for co-variation in `coordinated_changes`) marked and named."""

import numpy as np
from adjustText import adjust_text

from ...factors import load_factors_netcdf
from .. import common, style
from ._common import COMPONENT, SPEC, plot_ranked_loadings
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
    top = order[:N_TOP]
    heights = np.linspace(loadings[top[0]], LABEL_FLOOR * loadings[top[0]], N_TOP)
    texts = [
        ax.text(
            LABEL_RANK, y, names[i], va="center", fontsize=common.FACTOR_LABEL_FONT_SIZE
        )
        for y, i in zip(heights, top, strict=True)
    ]
    # Every point on the curve is an obstacle, so no label sits on it; labels
    # move only vertically, keeping their column
    adjust_text(
        texts,
        x=ranks,
        y=loadings[order],
        target_x=ranks[:N_TOP],
        target_y=loadings[top],
        ax=ax,
        only_move={"text": "y", "static": "y", "explode": "y", "pull": "y"},
        time_lim=10,
        arrowprops={
            "arrowstyle": "-",
            "lw": style.LINE_WIDTH_THIN / 2,
            "color": style.COLORS["reference"],
        },
    )


def makeFigure():
    return common.standalone_fixed(__name__)
