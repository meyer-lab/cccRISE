"""The RISE condition factor (A; conditions ordered by their most heavily
loaded component, then by that loading), each component divided by its
largest magnitude (keeping sign)."""

from ... import common
from . import _common as S

AXES = ("wide",)  # as wide as `genes`' "tall", which it sits above


def plot(axs):
    (ax,) = axs
    conditions = S.rise_factors()["pf2_A"].values.astype(float)
    common.plot_factor_heatmap(
        ax, conditions[common.dominant_order(conditions)], S.XLABEL
    )
    ax.set_xticks([])  # shares `genes`' component axis below it
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_ylabel(f"Condition (n={len(conditions)})")


def makeFigure():
    return common.standalone_fixed(__name__)
