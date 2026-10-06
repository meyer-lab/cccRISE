"""How much of each cccRISE component's per-cell loading (sample-scaled) a
cell-type label explains: eta-squared by coarse type, the extra explained by
fine types, and the residual no annotation-based method can represent."""

import numpy as np

from .. import common, style
from ._common import load

SIDES = {"sender": "Sender ($sc_B$)", "receiver": "Receiver ($rc_C$)"}


def plot(fig):
    table = load("cell_fractions")
    eta = table.groupby(["granularity", "side", "component"])[
        "type_eta_squared"
    ].first()
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (side, title) in zip(axes, SIDES.items(), strict=True):
        coarse = eta.loc["coarse", side].to_numpy(float)
        fine = eta.loc["fine", side].to_numpy(float)
        x = np.arange(1, len(coarse) + 1)
        ax.bar(x, coarse, color=style.COLORS["emphasis"], label="coarse type")
        ax.bar(
            x,
            fine - coarse,
            bottom=coarse,
            color=style.COLORS["highlight"],
            label="+ fine type",
        )
        ax.bar(
            x,
            1 - fine,
            bottom=fine,
            color=style.COLORS["muted"],
            label="within fine type",
        )
        ax.set_xticks(x, [f"C{k}" for k in x])
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel(r"share of per-cell variance ($\eta^2$)")
    axes[1].legend(
        fontsize=style.FONT_SIZE_DENSE,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
