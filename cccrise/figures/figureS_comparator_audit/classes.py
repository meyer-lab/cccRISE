"""Each fit's missed LR columns by pre-registered class."""

import numpy as np

from .. import common
from . import _common

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    table = _common.audit()
    order = [f"{m} {a}" for a in ("coarse", "fine") for m in _common.SHORT.values()]
    counts = table.groupby(["fit", "class"]).size().unstack(fill_value=0).reindex(order, fill_value=0)
    left = np.zeros(len(counts))
    for name, color in _common.CLASSES.items():
        values = counts.get(name, 0 * left)
        ax.barh(range(len(counts)), values, left=left, color=color, label=name.capitalize(), lw=0)
        left += np.asarray(values)
    ax.set_yticks(range(len(counts)), counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Missed LR columns")
    common.legend(ax, "right")


def makeFigure():
    return common.standalone_fixed(__name__)
