"""Replicate FMS histogram of every RISE and CP component whose FMS is
bimodal (`_common.is_bimodal`): a second population of replicates that lose
the component, rather than one or two that do."""

import numpy as np

from ... import common, style
from . import _common as S

N_COLS = 6
BINS = np.linspace(0, 1, 11)


def plot(fig):
    df = S.stability()
    bimodal = S.bimodal_components(df)
    axes = S.axes_present(df)
    n_cols = min(N_COLS, max(1, len(bimodal)))
    n_rows = max(1, -(-len(bimodal) // n_cols))
    grid = fig.subplots(n_rows, n_cols, sharex=True, squeeze=False)
    for ax, (stage, component) in zip(grid.flat, bimodal, strict=False):
        rows = df[(df["Stage"] == stage) & (df["Component"] == component)]
        ax.hist(
            [rows.loc[rows["Axis"] == axis, "FMS"] for axis in axes],
            bins=BINS,
            color=[style.STABILITY_AXIS_COLORS[axis] for axis in axes],
        )
        ax.set_title(
            f"{S.COMPONENT_NAMES[stage]} {component}", fontsize=style.FONT_SIZE_DENSE
        )
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
    for ax in grid.flat[len(bimodal) :]:
        ax.set_visible(False)
    fig.supxlabel("Factor match score")
    fig.supylabel("Replicates")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"], legend=S.legend)
