"""Each EE cell's C3 and C4 sender weight, each over its largest EE value (as
Figure 2's cutoff scales them), as a magnitude and an angle
(`_common.hormone_angles`; 0 degrees C3 only, 90 C4 only), colored by angle
past the magnitude floor (dotted arc, `_common.EXTREME_FLOOR`). The angle is
undefined below the dashed line (shaded), where the two sum to at most 0; the dotted
ray and point mark an example cell at 45 degrees on the floor."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .. import common, style
from . import _common as S

AXES = ("square",)
FLOOR = S.EXTREME_FLOOR
LIMITS = (-0.6, 1.1)
OUTSIDE_SHADE = "0.93"
EXAMPLE_ANGLE = 45
RAY_LENGTH = 1.25
ANGLE_CMAP = LinearSegmentedColormap.from_list(
    "c3_c4", [S.HORMONE_COLORS[3], "0.85", S.HORMONE_COLORS[4]]
)


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells")
    ee = S.hormone_angles(cells).assign(Class=S.hormone_extremes(cells))
    common.source_data(
        "hormone_angle_cells",
        ee.join(cells[["Cell", "Biopsy", "Status"]]),
        "Each EE cell's C3 and C4 sender weight over its largest EE value, their "
        "magnitude and angle (degrees; 0 C3 only, 90 C4 only; NaN where their sum "
        "is not positive), and its class (`_common.hormone_extremes`).",
    )
    shown = ee["Angle"].notna() & (ee["Magnitude"] >= FLOOR)
    ax.scatter(
        ee.loc[~shown, "C3 relative"],
        ee.loc[~shown, "C4 relative"],
        c=style.COLORS["muted"],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    points = ax.scatter(
        ee.loc[shown, "C3 relative"],
        ee.loc[shown, "C4 relative"],
        c=ee.loc[shown, "Angle"],
        cmap=ANGLE_CMAP,
        vmin=0,
        vmax=90,
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    _guides(ax)
    ax.set(
        xlim=LIMITS,
        ylim=LIMITS,
        xlabel="C3 sender weight / max",
        ylabel="C4 sender weight / max",
    )
    common.colorbar(ax, points, "Angle (°)")


def _guides(ax) -> None:
    """The undefined half-plane (shaded), the magnitude floor and the example
    cell."""
    line = {"color": style.COLORS["ink"], "linewidth": style.LINE_WIDTH_THIN}
    low = LIMITS[0]
    ax.fill(
        [low, low, -low], [-low, low, low], color=OUTSIDE_SHADE, linewidth=0, zorder=0
    )
    ax.plot([low, -low], [-low, low], linestyle=style.LINE_STYLE_REFERENCE, **line)
    arc = np.radians(np.linspace(-45, 135, 100))
    ax.plot(FLOOR * np.cos(arc), FLOOR * np.sin(arc), linestyle=":", **line)
    example = np.radians(EXAMPLE_ANGLE)
    point = (FLOOR * np.cos(example), FLOOR * np.sin(example))
    ax.plot(
        [0, RAY_LENGTH * np.cos(example)],
        [0, RAY_LENGTH * np.sin(example)],
        linestyle=":",
        **line,
    )
    ax.scatter(*point, color=style.COLORS["ink"], s=style.POINT_SIZE, zorder=3)
    ax.annotate(
        f"{FLOOR}  {EXAMPLE_ANGLE}°",
        point,
        xytext=(-4, -4),  # 45 degrees down-left, onto the low-weight cells
        textcoords="offset points",
        ha="center",
        va="top",
        rotation=-45,  # along the half-plane's edge
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
