"""A comparator column names cell types; how many of those types' cells
carry the matched cccRISE program? (`run_lr_concordance` cell tables.)

Rows: every CP comparator column whose LR pairs agree with its best cccRISE
match (`lr_class_min` recovered or same LR, other who/where); columns:
senders and receivers. For each cell type the comparator column weights at
least `MIN_WEIGHT` x its largest (on its own annotation; one per side in all
but one column), a bar split into the type's cells active in the matched
cccRISE component (> 0.5 x the component's 99th-percentile loading; the
component's colour) and the rest (grey), with the type on the bar and the
share and number of its cells beside it."""

import numpy as np
import pandas as pd
import seaborn as sns

from .. import common, style
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS, SHORT, load

MIN_WEIGHT = 0.25
AGREE = ("recovered", "same LR, different who/where")
SIDES = ("sender", "receiver")
BAR = 0.62  # bar height, in rows
TRACK = "0.9"


def matches() -> pd.DataFrame:
    """CP columns whose LR pairs agree with their match, coarse fits first."""
    m = load("matches")
    m = m[m["lr_class_min"].isin(AGREE)]
    fits = [(a, me) for a in ANNOTATIONS for me in METHODS]
    order = [fits.index(k) for k in zip(m["annotation"], m["method"], strict=True)]
    return m.assign(order=order).sort_values(["order", "comparator_component"])


def _heavy_types(weights, match, side):
    w = (
        weights[
            (weights["annotation"] == match.annotation)
            & (weights["method"] == match.method)
            & (weights["component"] == match.comparator_component)
            & (weights["side"] == side)
        ]
        .set_index("celltype")["weight"]
        .abs()
    )
    w = (w / w.max()).sort_values(ascending=False)
    return list(w[w >= MIN_WEIGHT].index)


def bars() -> pd.DataFrame:
    """One row per (match, side, heavy type), with the share of the type's
    cells active in the matched component, the type's cell count, the match
    (`row`), the slot within it (`slot`) and the bar's position `y`."""
    weights, activity = load("cell_weights"), load("cell_activity")
    rows = []
    for i, m in enumerate(matches().itertuples()):
        act = activity[
            (activity["annotation"] == m.annotation)
            & (activity["component"] == m.cccrise_component)
        ].set_index(["side", "celltype"])
        for side in SIDES:
            for slot, celltype in enumerate(_heavy_types(weights, m, side)):
                rows.append({
                    "row": i, "slot": slot, "side": side, "celltype": celltype,
                    "component": m.cccrise_component,
                    "match": f"{SHORT[m.method]} {m.annotation} {m.comparator_component}"
                    f"  →  C{m.cccrise_component}",
                    "active": act.loc[(side, celltype), "active_frac"],
                    "n_cells": act.loc[(side, celltype), "n_cells"],
                })  # fmt: skip
    table = pd.DataFrame(rows)
    # each match as tall as its more crowded side
    height = table.groupby("row")["slot"].max() + 1
    top = np.concatenate([[0], np.cumsum(height.to_numpy() + 0.5)[:-1]])
    return table.assign(y=top[table["row"]] + table["slot"])


def _side(ax, table, palette):
    colors = [palette[c - 1] for c in table["component"]]
    ax.barh(table["y"], 1, height=BAR, color=TRACK)
    ax.barh(table["y"], table["active"], height=BAR, color=colors)
    for r in table.itertuples():
        ax.text(
            0.02, r.y, r.celltype, va="center", color="white",
            fontsize=style.FONT_SIZE_DENSE - 0.5, fontweight="bold",
        )  # fmt: skip
        ax.text(
            1.03, r.y, f"{r.active:.0%} of {r.n_cells:,}", va="center",
            fontsize=style.FONT_SIZE_DENSE - 0.5,
        )  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(False)


def plot(fig):
    table = bars()
    palette = sns.color_palette("Dark2", 8)
    axes = fig.subplots(1, len(SIDES), sharey=True)
    for ax, side in zip(axes, SIDES, strict=True):
        _side(ax, table[table["side"] == side], palette)
        ax.set_title(f"{side}s", fontsize=style.FONT_SIZE_TITLE)
        ax.set_xlabel("cells of the type active in cccRISE")
    labels = table.groupby("row").agg(y=("y", "mean"), match=("match", "first"))
    axes[0].set_yticks(labels["y"], labels["match"], fontsize=style.FONT_SIZE_DENSE)
    axes[0].set_ylim(table["y"].max() + 0.7, -0.7)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 75 * style.MM))
