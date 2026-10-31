"""Association of every IBD RISE component's condition loading (`pf2_A`)
with every inflammation-status association (one-way ANOVA eta-squared,
FDR-corrected across all associations and components); a star marks FDR <
0.05, the 60 columns leaving no room for the values."""

from ... import common
from .. import _common as S
from ._common import save_association_data
from .cp_associations import FDR

AXES = ("wide",)
TICK_STEP = 10


def plot(axs):
    (ax,) = axs
    table = S.rise_association_table()
    common.plot_association_heatmap(table, ax, xlabel="RISE component", annotate=False)
    n = table["Component"].nunique()
    ticks = [1, *range(TICK_STEP, n + 1, TICK_STEP)]
    ax.set_xticks([t - 1 for t in ticks], ticks)
    save_association_data(
        table,
        S.corrected_loadings("pf2_A"),
        "rise_component",
        "each RISE component's condition loading (pf2_A), read-depth corrected "
        "as tested",
        FDR,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
