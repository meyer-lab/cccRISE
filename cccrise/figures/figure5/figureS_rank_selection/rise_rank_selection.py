"""IBD RISE rank selection by bi-cross-validation, from ``uv run cccrise-riserank``,
with BiCV test on its own y axis: its R2X is far below the fit's and BiCV
train's, so on theirs it runs flat along the bottom."""

from ... import common, style
from ...figure2.rise_rank_selection import draw

DATASET = "ibd"
AXES = ("square",)
#: Fraction of the axes height BiCV test's own axis spans, at the bottom,
#: below the fit and BiCV train curves
TEST_BAND = 0.3


def plot(axs):
    ax, color = axs[0], style.COLORS["heldout"]
    test_ax = ax.twinx()
    draw(ax, DATASET, test_ax=test_ax)
    lo, hi = test_ax.dataLim.y0, test_ax.dataLim.y1
    ticks = [t for t in test_ax.get_yticks() if lo <= t <= hi]
    test_ax.set_ylim(lo - 0.05 * (hi - lo), lo + (hi - lo) / TEST_BAND)
    test_ax.set_yticks(ticks)
    test_ax.tick_params(axis="y", colors=color)
    test_ax.spines["right"].set_visible(True)
    test_ax.spines["right"].set_color(color)
    test_ax.spines["right"].set_bounds(lo, hi)


def makeFigure():
    return common.standalone_fixed(__name__)
