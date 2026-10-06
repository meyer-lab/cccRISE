"""Containment of each cccRISE component in each comparator's span, coarse
(c) and fine (f) cell types; starred where contained (>= 0.85 and above the
row-permutation null)."""

import numpy as np

from .. import common, style
from ._common import METHODS, SHORT, load

MODES = {"condition": "Condition mode", "lr": "LR mode"}


def plot(fig):
    table = load("containment")
    table = table[table["direction"] == "cccRISE in comparator"]
    columns = [(m, g) for m in METHODS for g in ("coarse", "fine")]
    labels = [f"{SHORT[m]} {g[0]}" for m, g in columns]
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (mode, title) in zip(axes, MODES.items(), strict=True):
        sub = table[table["mode"] == mode]
        grid = sub.pivot_table(
            index="component", columns=["method", "granularity"], values="containment"
        ).reindex(columns=columns)
        calls = sub.pivot_table(
            index="component", columns=["method", "granularity"], values="contained"
        ).reindex(columns=columns)
        image = ax.imshow(
            grid.to_numpy(float),
            cmap=style.CMAPS["sequential"],
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        for (i, j), v in np.ndenumerate(grid.to_numpy(float)):
            star = "*" if calls.iloc[i, j] else ""
            ax.text(
                j,
                i,
                f"{v:.2f}{star}",
                ha="center",
                va="center",
                fontsize=style.FONT_SIZE_DENSE - 1,
                color="white" if v > 0.6 else style.COLORS["ink"],
            )
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(grid)), [f"C{k}" for k in grid.index])
        ax.set_title(title)
        ax.grid(False)
    fig.colorbar(image, ax=axes, label="containment", shrink=0.8)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
