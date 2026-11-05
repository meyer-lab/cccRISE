"""The angles of EE cells past the magnitude floor (`hormone_angle_scatter`),
bars colored by the class their angle bins a cell into
(`_common.hormone_extremes`): C3 high up to `EXTREME_SECTOR` degrees, C4
high from 90 - `EXTREME_SECTOR`, both between."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.typing import LineStyleType

from .. import common, style
from . import _common as S

AXES = ("square",)
ANGLE_BINS = np.arange(-45, 135.1, 7.5)
HEADROOM = 1.3  # y limit over the tallest bar's, for the lines' key
CLASS_KEY_SHIFT = 0.06  # axes widths the class key moves left of center
#: Reference angles: {label: (angles, line style)}
REFERENCES: dict[str, tuple[tuple[float, ...], LineStyleType]] = {
    "Thresholds": ((S.EXTREME_SECTOR, 90 - S.EXTREME_SECTOR), "--"),
    "Equal weight": ((45,), ":"),
}


def sector(angle: float) -> str:
    """The `_common.SECTOR_COLORS` key of a bin centered at `angle`."""
    if angle <= S.EXTREME_SECTOR:
        return "Cmp. 3 high"
    return "Cmp. 4 high" if angle >= 90 - S.EXTREME_SECTOR else "Both"


def plot(axs):
    (ax,) = axs
    polar = S.hormone_angles(S.programs("enteroendocrine_cells"))
    angles = polar.loc[
        polar["Angle"].notna() & (polar["Magnitude"] >= S.EXTREME_FLOOR), "Angle"
    ]
    common.source_data(
        "hormone_angle_distribution",
        angles.to_frame(),
        f"Angles (degrees) of EE cells with magnitude at least {S.EXTREME_FLOOR}.",
    )
    bars = S.cell_histogram(ax, angles, ANGLE_BINS)
    for bar in bars:
        bar.set_facecolor(S.SECTOR_COLORS[sector(bar.get_x() + bar.get_width() / 2)])
    # Room above the bars for the lines' key inside the axes; the lines stop
    # at the bars' top, below it
    ax.set_ylim(top=HEADROOM * ax.get_ylim()[1])
    for marks, linestyle in REFERENCES.values():
        for mark in marks:
            ax.axvline(
                mark,
                ymax=1 / HEADROOM,
                color=style.COLORS["ink"],
                linestyle=linestyle,
                linewidth=style.LINE_WIDTH_THIN,
            )
    ax.set_xticks([0, 45, 90])
    ax.set_xlabel(
        f"Angle (°), magnitude ≥ {S.EXTREME_FLOOR}", fontsize=style.FONT_SIZE_DENSE
    )
    lines = ax.legend(
        handles=[
            Line2D([], [], color=style.COLORS["ink"], linestyle=ls, label=label)
            for label, (_, ls) in REFERENCES.items()
        ],
        loc="upper right",
        fontsize=style.FONT_SIZE_DENSE,
        ncols=len(REFERENCES),
        handlelength=1.1,
        handletextpad=0.25,
        columnspacing=0.5,
        borderaxespad=0.1,
        borderpad=0.1,
    )
    ax.add_artist(lines)  # kept when the class key below replaces `ax.legend_`
    classes = common.legend(
        ax,
        "above",
        handles=[
            Patch(facecolor=color, edgecolor=style.COLORS["ink"], label=label)
            for label, color in S.SECTOR_COLORS.items()
        ],
        fontsize=style.FONT_SIZE_DENSE,
        ncols=len(S.SECTOR_COLORS),
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.6,
    )
    # Wider than the axes: hangs over the gaps instead of widening them, a
    # little left of center to stay clear of the next panel's letter
    classes.set_in_layout(False)
    x, y = classes.get_bbox_to_anchor().transformed(ax.transAxes.inverted()).p0
    classes.set_bbox_to_anchor((x - CLASS_KEY_SHIFT, y), transform=ax.transAxes)


def makeFigure():
    return common.standalone_fixed(__name__)
