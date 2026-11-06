"""The magnitudes of all EE cells' C3/C4 sender weights (Figure 4's angle
scatter, `hormone_angle_scatter`), with the floor past which their angle
bins them (`_common.EXTREME_FLOOR`, Figure 2's cutoff)."""

import numpy as np
from matplotlib.lines import Line2D

from ... import common, style
from .. import _common as S

AXES = ("square",)
MAGNITUDE_BINS = np.arange(0, 1.3, 0.05)


def plot(axs):
    (ax,) = axs
    polar = S.hormone_angles(S.programs("enteroendocrine_cells"))
    common.source_data(
        "hormone_magnitude_distribution",
        polar[["Magnitude"]],
        "Magnitude of each EE cell's C3 and C4 sender weights over their largest "
        f"EE values; the floor is {S.EXTREME_FLOOR}.",
    )
    S.cell_histogram(ax, polar["Magnitude"], MAGNITUDE_BINS)
    color, linestyle = style.COLORS["ink"], style.LINE_STYLE_REFERENCE
    ax.axvline(
        S.EXTREME_FLOOR,
        color=color,
        linestyle=linestyle,
        linewidth=style.LINE_WIDTH_THIN,
    )
    ax.set_xlabel("Magnitude, all EE cells", fontsize=style.FONT_SIZE_DENSE)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                label=f"Floor ({S.EXTREME_FLOOR})",
            )
        ],
        fontsize=style.FONT_SIZE_DENSE,
        loc="upper right",
        handlelength=1.5,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
