"""Program recovery: how much of each cccRISE program's LR column lies in the
span of each comparator fit's LR columns (1 when some combination of them
reproduces it). Outlined: recovered (>= 0.85); dot: its sample pattern is
recovered too."""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ...comparators.agreement import CONTAINED
from .. import common, style
from . import _common

AXES = ("rect",)


def containment(mode: str):
    """Programs x fits, the containment of each cccRISE column in `mode`."""
    frames = []
    for annotation in _common.ANNOTATIONS:
        table = _common.validation("comparator_containment", annotation)
        table = table[(table["direction"] == "cccRISE in comparator")
                      & (table["mode"] == mode) & (table["rank_mode"] == "own")]  # fmt: skip
        frames.append(table.assign(fit=[_common.fit_label(m, annotation) for m in table["method"]]))
    grid = pd.concat(frames).pivot_table(index="component", columns="fit", values="containment")
    return grid[[_common.fit_label(m, a) for m, a in _common.FITS]]


def plot(axs):
    (ax,) = axs
    lr, condition = containment("lr"), containment("condition")
    ax.imshow(lr.to_numpy(), cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(lr.to_numpy()):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=style.FONT_SIZE_DENSE,
                color="white" if value > 0.6 else style.COLORS["ink"])  # fmt: skip
        if value >= CONTAINED:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                          ec=style.COLORS["ink"], lw=1))  # fmt: skip
        if condition.to_numpy()[i, j] >= CONTAINED:
            ax.plot(j + 0.33, i - 0.3, marker="o", ms=2, color="white")
    ax.set_xticks(range(lr.shape[1]), lr.columns, rotation=45, ha="right")
    ax.set_yticks(range(lr.shape[0]), [_common.PROGRAMS[c] for c in lr.index])
    ax.tick_params(length=0)


def makeFigure():
    return common.standalone_fixed(__name__)
