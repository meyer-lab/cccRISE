"""Supplementary figure: CCC-RISE across the four communication scores --
how closely each alternate score's CP components match the default's, how
stable each score's fit is under the jackknife, and what each component
represents under each score."""

from string import ascii_uppercase

from matplotlib.text import Annotation
from matplotlib.transforms import Bbox

from ... import common, style
from . import _common as S
from . import factor_match, interpretation, jackknife

PANELS = {
    "factor_match": factor_match,
    "jackknife": jackknife,
    "interpretation": interpretation,
}
LAYOUT = [
    ["factor_match", "factor_match", "jackknife"],
    ["interpretation"] * 3,
]
# mm per row; the interpretation grid needs ~30 mm per CP component.
ROW_HEIGHTS = [52, 188]
# Panels whose second axes (and those right of it) get a letter of their own
SPLIT_PANELS = {"factor_match", "interpretation"}


def makeFigure():
    fig = common.compose(
        (style.FIGURE_WIDTH_DOUBLE, sum(ROW_HEIGHTS) * style.MM),
        LAYOUT,
        PANELS,
        height_ratios=ROW_HEIGHTS,
    )
    S.method_legend(fig)
    letters = iter(ascii_uppercase)
    for name, subfig in zip(PANELS, fig.subfigs, strict=True):
        subfig.texts[0].set_text(next(letters))
        if name in SPLIT_PANELS:
            _letter_beside(subfig, subfig.axes[1], next(letters))
    return fig


def _letter_beside(subfig, ax, letter: str) -> None:
    """Letter `ax` (and the axes right of it) as a panel of its own, level
    with `subfig`'s panel letter and left of `ax`'s tick labels, tagged with
    the same panel module for `record_panel_letters`."""
    panel_label = subfig.texts[0]

    def left_of_ax(renderer):
        x = ax.get_tightbbox(renderer).x0
        return Bbox.from_extents(x, 0, x + 1, 1)

    subfig.add_artist(
        Annotation(
            letter,
            (0, 1),
            xycoords=(left_of_ax, panel_label),
            ha="left",
            va="top",
            fontsize=panel_label.get_fontsize(),
            fontweight=panel_label.get_fontweight(),
            gid=panel_label.get_gid(),
        )
    )
