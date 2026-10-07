"""How much of each cccRISE component a cell-type mean represents, as the
annotation gets finer. Top: variance of the sample-scaled per-cell loading
explained by its (sample x type) means -- what a pseudobulk tensor at that
granularity sees; bottom: by type alone. Solid: each coarse type split by
its own expression (bisecting k-means, >= 100 cells per subtype); dashed and
band: random splits of the same sizes (median, 5-95% over 20 seeds); star:
the fine annotation (26 types); first point: the coarse annotation (8)."""

import numpy as np

from .. import common, style
from ._common import load_ladder

SIDES = {"sender": "Sender ($sc_B$)", "receiver": "Receiver ($rc_C$)"}
STATS = {"r2_sample_type": "$R^2$ (sample × type)", "eta2_type": "$\\eta^2$ (type)"}


def _series(table, side, component, stat):
    sub = table[(table["side"] == side) & (table["component"] == component)]
    coarse = sub[sub["level"] == "coarse"]
    expr = sub[sub["kind"] == "expr"].sort_values("n_types")
    expr = expr.loc[:, ["n_types", stat]]
    expr = np.vstack([coarse[["n_types", stat]].to_numpy(), expr.to_numpy()])
    rand = sub[sub["kind"] == "rand"].groupby("m")
    band = rand[stat].quantile([0.05, 0.5, 0.95]).unstack()
    n_types = rand["n_types"].first()
    fine = sub.loc[sub["level"] == "fine", ["n_types", stat]].to_numpy()[0]
    return expr, n_types.to_numpy(), band, fine


def plot(fig):
    table = load_ladder("representability")
    components = sorted(table["component"].unique())
    colors = dict(zip(components, style.CATEGORICAL, strict=False))
    axes = fig.subplots(2, 2, sharex=True)
    for row, (stat, ylabel) in enumerate(STATS.items()):
        for col, (side, title) in enumerate(SIDES.items()):
            ax = axes[row, col]
            for c in components:
                expr, n_rand, band, fine = _series(table, side, c, stat)
                ax.plot(
                    expr[:, 0],
                    expr[:, 1],
                    color=colors[c],
                    marker="o",
                    ms=style.MARKER_SIZE,
                    lw=style.LINE_WIDTH,
                    label=f"C{c}",
                )
                ax.plot(
                    n_rand,
                    band[0.5],
                    color=colors[c],
                    ls="--",
                    lw=style.LINE_WIDTH_THIN,
                )
                ax.fill_between(
                    n_rand, band[0.05], band[0.95], color=colors[c], alpha=0.15, lw=0
                )
                ax.scatter(
                    *fine,
                    marker="*",
                    s=40,
                    color=colors[c],
                    edgecolor="white",
                    lw=0.3,
                    zorder=4,
                )
            ax.set_xscale("log")
            ax.set_xticks([8, 13, 26, 38, 79], ["8", "13", "26", "38", "79"])
            ax.minorticks_off()
            ax.grid(False)
            if row == 0:
                ax.set_title(title)
            else:
                ax.set_xlabel("cell types")
            if col == 0:
                ax.set_ylabel(ylabel)
    axes[0, 1].legend(
        fontsize=style.FONT_SIZE_DENSE, loc="upper left", bbox_to_anchor=(1.02, 1)
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
