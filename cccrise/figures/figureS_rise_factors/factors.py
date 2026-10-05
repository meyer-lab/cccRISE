"""The RISE condition factor (A; conditions ordered by their most heavily
loaded component, then by that loading) and mixing factor (B; columns are the
components, and the fit's is the identity), each component divided by its
largest magnitude (keeping sign)."""

import numpy as np

from .. import common
from . import _common as S

AXES = ("rect", "square")


def plot(axs):
    factors = S.rise_factors()
    conditions = factors["pf2_A"].values.astype(float)
    common.plot_factor_heatmap(
        axs[0], conditions[common.dominant_order(conditions)], S.XLABEL
    )
    axs[0].set_yticks([])
    axs[0].set_ylabel(f"Condition (n={len(conditions)})")

    common.plot_factor_heatmap(axs[1], factors["pf2_B"].values.astype(float), S.XLABEL)
    axs[1].set_yticks(np.array(S.TICKS) - 1, S.TICKS)
    axs[1].set_ylabel("B row")
    for ax in axs:
        ax.set_xticks(np.array(S.TICKS) - 1, S.TICKS)  # all 17 would collide


def makeFigure():
    return common.standalone_fixed(__name__)
