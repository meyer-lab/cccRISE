"""How much of each cccRISE program's per-cell sender and receiver weight a
label-pooled method can represent: the R2 of its sample x cell-type means
(`run_granularity_ladder`), at fine labels (bars) and coarse (ticks)."""

import numpy as np

from .. import common, style
from . import _common

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    table = _common.validation("granularity_representability")
    width = 0.38
    for offset, side in ((-1, "sender"), (1, "receiver")):
        for level, kwargs in (("fine", {}), ("coarse", {"fill": False})):
            rows = table[(table["side"] == side) & (table["kind"] == level)].sort_values("component")
            x = rows["component"].to_numpy() + offset * width / 2
            if level == "fine":
                ax.bar(x, rows["r2_sample_type"], width, color=style.COLORS[side],
                       label=side.capitalize())  # fmt: skip
            else:
                ax.scatter(x, rows["r2_sample_type"], marker="_", s=40, color=style.COLORS["ink"],
                           lw=1, label="Coarse labels" if side == "sender" else None)  # fmt: skip
    ax.set_xticks(np.arange(1, 6), [f"C{c}" for c in range(1, 6)])
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Per-cell variance\nrepresentable by labels")
    ax.legend(loc="upper left", ncols=3, frameon=False)


def makeFigure():
    return common.standalone_fixed(__name__)
