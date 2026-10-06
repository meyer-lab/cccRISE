"""cccRISE components within single fine cell types (F2): each unit's
maxT-adjusted permutation p (all units per association) for samples with
at least 3 cells of the type (primary) and at least 1 (sensitivity). Labelled: units
with p < 0.05 and the pre-registered cDC1 unit (C2 receiver)."""

import numpy as np

from ...runners.datasets import VALIDATION_DIR
from .. import common, style

FILENAME = "celltype_condition_anova_alad.parquet"
PRIMARY = ("receiver", "cDC1", 2, "Control vs. ALAD (All)")


def plot(fig):
    table = common.load_runner_output(FILENAME, VALIDATION_DIR)
    order = list(dict.fromkeys(table["association"]))
    axes = fig.subplots(1, 2, sharey=True)
    rng = np.random.default_rng(0)
    for ax, cells in zip(axes, (3, 1), strict=True):
        sub = table[table["min_cells"] == cells]
        y = sub["association"].map({a: i for i, a in enumerate(order)})
        y = y + rng.uniform(-0.25, 0.25, len(sub))
        hit = sub["maxt_p"] < 0.05
        ax.scatter(-np.log10(sub["maxt_p"]), y, s=4, color=style.COLORS["muted"])
        ax.scatter(
            -np.log10(sub.loc[hit, "maxt_p"]), y[hit], s=10, color=style.COLORS["up"]
        )
        key = (
            (sub["side"] == PRIMARY[0])
            & (sub["type"] == PRIMARY[1])
            & (sub["component"] == PRIMARY[2])
            & (sub["association"] == PRIMARY[3])
        )
        for _, row in sub[key].iterrows():
            ax.annotate(
                "C2 rec. cDC1",
                (-np.log10(row["maxt_p"]), order.index(row["association"])),
                xytext=(8, -6),
                textcoords="offset points",
                fontsize=style.FONT_SIZE_DENSE - 1,
                color=style.COLORS["emphasis"],
                arrowprops={
                    "arrowstyle": "-",
                    "color": style.COLORS["emphasis"],
                    "lw": 0.6,
                },
            )
        ax.axvline(-np.log10(0.05), color=style.COLORS["reference"], ls="--", lw=0.8)
        ax.set_title(
            rf"$\geq${cells} cell{'s' if cells > 1 else ''} per sample "
            f"({hit.sum()} units p < .05)"
        )
        ax.set_xlabel(r"$-\log_{10}$ maxT p")
        ax.grid(False)
    axes[0].set_yticks(range(len(order)), order, fontsize=style.FONT_SIZE_DENSE - 1)
    axes[0].invert_yaxis()


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
