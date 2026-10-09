"""Which comparator columns cccRISE does not contain, and why (draft).

Left: each LR column's effective number of pairs (1 / sum of squared
loading shares squared: 1 for a single-pair column), for every comparator
column (own rank, coarse and fine) and for cccRISE's components. Right: per
comparator column, how much cccRISE weights the column's top pair (its best
|loading| / component max over cccRISE's components) against the column's
LR containment in cccRISE's span; each distinct top pair of a contained
column is named beside it, those of the uncontained columns (containment
below `CONTAINED_FLOOR`) are listed in one block with their column counts."""

import numpy as np
import pandas as pd

from .. import common, style
from ..figureS_lr_concordance import containment
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS, SHORT, load

COLORS = dict(zip(METHODS, style.CATEGORICAL[1:4], strict=True))
CONTAINED_FLOOR = 0.2


def columns() -> pd.DataFrame:
    """Per LR column (cccRISE and own-rank comparators, both annotations): its
    top pair, effective number of pairs, cccRISE's weight on the top pair and
    (comparators) containment in cccRISE's span."""
    p = load("profiles")
    keys = ["method", "annotation", "component"]
    eff = p.groupby(keys)["share"].apply(lambda s: 1 / (s**2).sum()).rename("eff_pairs")
    top = p[p["lr_rank"] == 1].set_index(keys)["lr_pair"].rename("top_pair")
    ccc = p[p["method"] == "cccRISE"].groupby("lr_pair")["rel_max"].max()
    table = pd.concat([top, eff], axis=1).join(
        containment.load("comparator in cccRISE")["containment"].rename_axis(keys)
    )
    return table.assign(ccc_weight=table["top_pair"].map(ccc)).reset_index()


def _effective(ax, table):
    rng = np.random.default_rng(0)
    groups = [("cccRISE", "cccRISE")] + [(m, SHORT[m]) for m in METHODS]
    for i, (method, _label) in enumerate(groups):
        sub = table[table["method"] == method]
        if method == "cccRISE":
            sub = sub[sub["annotation"] == "coarse"]  # the same fit on both
        for annotation in ANNOTATIONS:
            part = sub[sub["annotation"] == annotation] if method != "cccRISE" else sub
            ax.scatter(
                part["eff_pairs"], i + rng.uniform(-0.15, 0.15, len(part)),
                s=style.POINT_SIZE, lw=0.7,
                facecolor=COLORS.get(method, "black") if annotation == "coarse" else "none",
                edgecolor=COLORS.get(method, "black"),
            )  # fmt: skip
            if method == "cccRISE":
                break
    ax.set_yticks(range(len(groups)), [g[1] for g in groups])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5], ["1", "2", "5"])
    ax.xaxis.set_minor_formatter("")
    ax.axvline(1, color=style.COLORS["reference"], lw=style.LINE_WIDTH_THIN, ls=":")
    ax.set_xlabel("effective number of LR pairs")
    ax.grid(False)


def _labels(ax, comp):
    """Contained columns' top pairs beside them; the uncontained ones'
    listed in one block, most columns first."""
    size = style.FONT_SIZE_DENSE - 1.5
    low = comp["containment"] < CONTAINED_FLOOR
    for pair, sub in comp[~low].groupby("top_pair"):
        ax.annotate(
            f"{pair.replace('^', '–')} ({len(sub)})",
            (sub["ccc_weight"].mean(), sub["containment"].mean()), xytext=(5, 0),
            textcoords="offset points", fontsize=size, va="center",
            color=style.COLORS["ink"],
        )  # fmt: skip
    counts = comp[low]["top_pair"].value_counts(sort=False)
    counts = counts.sort_values(ascending=False, kind="stable")
    lines = [f"{p.replace('^', '–')} ({n})" for p, n in counts.items()]
    ax.text(
        0.45, 0.04, "not contained, top pair (columns):\n" + "\n".join(lines),
        fontsize=size, va="bottom", color=style.COLORS["ink"], linespacing=1.25,
    )  # fmt: skip


def _containment(ax, table):
    comp = table[table["method"] != "cccRISE"]
    rng = np.random.default_rng(1)
    for (method, annotation), sub in comp.groupby(["method", "annotation"]):
        ax.scatter(
            sub["ccc_weight"] + rng.uniform(-0.01, 0.01, len(sub)),
            sub["containment"] + rng.uniform(-0.01, 0.01, len(sub)),
            s=style.POINT_SIZE_FOREGROUND, lw=0.7,
            facecolor=COLORS[method] if annotation == "coarse" else "none",
            edgecolor=COLORS[method], label=f"{SHORT[method]} {annotation}",
        )  # fmt: skip
    _labels(ax, comp)
    ax.set_xlim(-0.04, 1.3)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("cccRISE |loading| / max on the column's top pair")
    ax.set_ylabel("containment in cccRISE span")
    ax.legend(fontsize=style.FONT_SIZE_DENSE - 1, loc="upper left", ncol=2)
    ax.grid(False)


def plot(fig):
    table = columns()
    left, right = fig.subplots(1, 2, width_ratios=(0.8, 1.4))
    _effective(left, table)
    _containment(right, table)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 80 * style.MM))
