"""Replicate FMS histogram of every unstable RISE and CP component -- one with
a replicate below `UNSTABLE_FMS` or a bimodal FMS distribution -- separating
components that lose one or two replicates from ones that split into two
populations. Counts are on a log scale, so a single replicate stays visible
beside the dozens that re-identify the component."""

import numpy as np

from ... import common, style
from . import _common as S

N_COLS = 6
BINS = np.linspace(0, 1, 21)


def plot(fig):
    df = S.stability()
    unstable = S.unstable_components(df)
    axes = S.axes_present(df)
    n_rows = max(1, -(-len(unstable) // N_COLS))
    grid = fig.subplots(n_rows, N_COLS, sharex=True, squeeze=False)
    for ax, (stage, component) in zip(grid.flat, unstable, strict=False):
        rows = df[(df["Stage"] == stage) & (df["Component"] == component)]
        ax.hist(
            [rows.loc[rows["Axis"] == axis, "FMS"] for axis in axes],
            bins=BINS,
            color=[style.STABILITY_AXIS_COLORS[axis] for axis in axes],
            log=True,
        )
        common.reference_line(ax, S.UNSTABLE_FMS, horizontal=False)
        ax.set_title(
            f"{S.COMPONENT_NAMES[stage]} {component}", fontsize=style.FONT_SIZE_DENSE
        )
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
    for ax in grid.flat[len(unstable) :]:
        ax.set_visible(False)
    fig.supxlabel("Factor match score")
    fig.supylabel("Replicates")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"], legend=S.legend)
