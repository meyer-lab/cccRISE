"""Supplementary figure: CCC-RISE across the four communication scores --
how closely each alternate score's CP components match the default's, and
what each component represents under each score."""

from string import ascii_uppercase

from matplotlib.text import Annotation
from matplotlib.transforms import Bbox

from ... import common, style
from . import _common as S
from . import factor_match, interpretation

PANELS = {
    "factor_match": factor_match,
    "interpretation": interpretation,
}
LAYOUT = [
    ["factor_match"],
    ["interpretation"],
]
# mm per row; the interpretation grid needs ~30 mm per CP component.
ROW_HEIGHTS = [46, 188]
KEY_SPACE = 6  # mm between A-B's letters and axes, for the methods' legend
# Panels whose second axes (and those right of it) get a letter of their own
SPLIT_PANELS = {"factor_match", "interpretation"}


def makeFigure():
    fig = common.compose(
        (style.FIGURE_WIDTH_DOUBLE, sum(ROW_HEIGHTS) * style.MM),
        LAYOUT,
        PANELS,
        height_ratios=ROW_HEIGHTS,
    )
    letters = iter(ascii_uppercase)
    for name, subfig in zip(PANELS, fig.subfigs, strict=True):
        subfig.texts[0].set_text(next(letters))
        if name in SPLIT_PANELS:
            _letter_beside(subfig, subfig.axes[1], next(letters))
    _fit_match_row(fig)
    return fig


def _fit_match_row(fig) -> None:
    """With the layout frozen, A-B's axes narrowed to span from C's LR pair
    axes' left to D's receiver axes' right, and lowered under the methods'
    legend, which sits under their letters; A's letter kept left of its
    labels."""
    fig.draw_without_rendering()
    fig.set_layout_engine("none")
    top, bottom = fig.subfigs
    a, b = top.axes[:2]
    left, right = bottom.axes[0].get_position().x0, bottom.axes[2].get_position().x1
    pa, pb = a.get_position(), b.get_position()
    gap = pb.x0 - pa.x1 + style.PANEL_GAP_MIN / fig.get_figwidth()
    width = (right - left - gap) / 2
    # Positions are fractions of each subfigure, which span the figure's width
    key = KEY_SPACE * style.MM / (top.bbox.height / fig.dpi)
    for ax, x0 in ((a, left), (b, left + width + gap)):
        ax.set_position((x0, pa.y0, width, pa.height - key))
    S.method_legend(
        fig,
        loc="upper center",
        bbox_to_anchor=((left + right) / 2, pa.y1),
        bbox_transform=top.transSubfigure,
    )
    tight = a.get_tightbbox()
    letter = top.texts[0]
    letter.set_x((tight.x0 - top.bbox.x0) / top.bbox.width)


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
