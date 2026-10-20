"""Every unit's association with each contrast, maxT over every unit of every
fit: the share (and count) of each method's units passing maxT p < 0.05,
unadjusted and adjusted for the first five composition PCs."""

import numpy as np

from .. import common, style
from . import _common

AXES = ("rect", "rect")
ALPHA = 0.05


def plot(axs):
    rows = _common.table("composition_association")
    contrasts = list(dict.fromkeys(rows["contrast"]))
    for ax, adjusted in zip(axs, (False, True), strict=True):
        subset = rows[rows["adjusted"] == adjusted]
        hits = subset.groupby(["contrast", "method"])["p_maxT"].apply(lambda p: (p < ALPHA).sum())
        sizes = subset.groupby(["contrast", "method"]).size()
        grid = (hits / sizes).unstack().reindex(index=contrasts, columns=list(_common.METHODS))
        ax.imshow(grid.to_numpy(float), cmap=style.CMAPS["weight"], vmin=0, vmax=0.7, aspect="auto")
        for (i, j), share in np.ndenumerate(grid.to_numpy(float)):
            key = (contrasts[i], _common.METHODS[j])
            ax.text(j, i, f"{hits.get(key, 0)}/{sizes.get(key, 0)}", ha="center", va="center",
                    fontsize=style.FONT_SIZE_DENSE,
                    color="white" if share > 0.4 else style.COLORS["ink"])  # fmt: skip
        ax.set_xticks(range(len(_common.METHODS)), [_common.SHORT[m] for m in _common.METHODS])
        ax.set_yticks(range(len(contrasts)), contrasts if not adjusted else [])
        ax.set_title("Adjusted for composition" if adjusted else "Unadjusted", fontsize=style.FONT_SIZE_DENSE + 1)
        ax.tick_params(length=0)


def makeFigure():
    return common.standalone_fixed(__name__)
