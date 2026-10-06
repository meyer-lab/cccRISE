"""Linear CKA of each comparator's condition and LR modes with cccRISE's, on
coarse and on fine cell types; filled where above the row-permutation null's
95th percentile."""

import numpy as np

from .. import common, style
from ._common import METHODS, SHORT, load

MODES = {"condition": "Condition mode", "lr": "LR mode"}


def plot(fig):
    table = load("subspace")
    table = table[table["statistic"] == "linear_cka"]
    axes = fig.subplots(1, 2, sharey=True)
    colors = dict(zip(METHODS, style.CATEGORICAL[1:], strict=False))
    for ax, (mode, title) in zip(axes, MODES.items(), strict=True):
        for method in METHODS:
            rows = table[(table["mode"] == mode) & (table["method"] == method)]
            rows = rows.set_index("granularity").reindex(["coarse", "fine"])
            ax.plot([0, 1], rows["observed"], color=colors[method], lw=1)
            for x, (_g, row) in enumerate(rows.iterrows()):
                ax.scatter(
                    x,
                    row["observed"],
                    s=18,
                    color=colors[method],
                    facecolor=colors[method] if row["above_null"] else "white",
                    zorder=3,
                    label=SHORT[method] if x == 0 else None,
                )
            ax.scatter(
                [0, 1],
                rows["perm_q95"],
                marker="_",
                s=60,
                color=colors[method],
                alpha=0.6,
            )
        ax.set_xticks([0, 1], ["coarse (8)", "fine (26)"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel("linear CKA with cccRISE\n(dash: null 95th pct)")
    axes[1].legend(fontsize=style.FONT_SIZE_DENSE, frameon=False)
    axes[0].set_ylim(0, np.ceil(table["observed"].max() * 10 + 1) / 10)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
