"""Granularity (b): a comparator column names cell types; how many of those
types' cells carry the matched cccRISE program?

For every CP comparator column whose LR pairs agree with its best cccRISE
match (`lr_class_min` recovered or same LR, other who/where), and each side:
the cell types the column weights at least `MIN_WEIGHT` x its largest (on
its own annotation; in practice one per side), each as one bar split into
its cells active in the matched cccRISE component (> 0.5 x the component's
99th-percentile loading; component colour) and the rest (grey), with the
share and count of cells after the bar."""

import numpy as np
import pandas as pd
import seaborn as sns

from .. import common, style
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS, SHORT, load

MIN_WEIGHT = 0.25
AGREE = ("recovered", "same LR, different who/where")
SIDES = ("sender", "receiver")


def matches() -> pd.DataFrame:
    """CP columns whose LR pairs agree with their match, by fit."""
    m = load("matches")
    m = m[m["lr_class_min"].isin(AGREE)]
    order = {
        (a, me): i
        for i, (a, me) in enumerate((a, me) for a in ANNOTATIONS for me in METHODS)
    }
    return m.assign(
        order=[order[k] for k in zip(m["annotation"], m["method"], strict=True)]
    ).sort_values(["order", "comparator_component"])


def bars() -> pd.DataFrame:
    """One row per (match, side, heavy type): the share of the type's cells
    active in the matched component and the type's cell count."""
    weights, activity = load("cell_weights"), load("cell_activity")
    rows = []
    for m in matches().itertuples():
        for side in SIDES:
            w = (
                weights[
                    (weights["annotation"] == m.annotation)
                    & (weights["method"] == m.method)
                    & (weights["component"] == m.comparator_component)
                    & (weights["side"] == side)
                ]
                .set_index("celltype")["weight"]
                .abs()
            )
            act = activity[
                (activity["annotation"] == m.annotation)
                & (activity["component"] == m.cccrise_component)
                & (activity["side"] == side)
            ].set_index("celltype")
            for celltype, weight in (w / w.max()).sort_values(ascending=False).items():
                if weight >= MIN_WEIGHT:
                    rows.append(
                        {
                            "match": f"{SHORT[m.method]} {m.annotation} {m.comparator_component}"
                            f" → C{m.cccrise_component}",
                            "component": m.cccrise_component,
                            "label": f"{side}s: {celltype}",
                            "active": act.loc[celltype, "active_frac"],
                            "n_cells": act.loc[celltype, "n_cells"],
                        }
                    )
    table = pd.DataFrame(rows)
    group = table["match"].factorize()[0]
    return table.assign(y=np.arange(len(table)) + 1.2 * (group + 1))


def plot(fig):
    table = bars()
    palette = sns.color_palette("Dark2", 5)
    ax = fig.subplots()
    colors = [palette[c - 1] for c in table["component"]]
    ax.barh(table["y"], table["active"], color=colors, height=0.7)
    ax.barh(
        table["y"], 1 - table["active"], left=table["active"], color="0.88", height=0.7
    )
    for row in table.itertuples():
        ax.text(
            1.02, row.y, f"{row.active:.0%} of {row.n_cells:,} cells", va="center",
            fontsize=style.FONT_SIZE_DENSE - 0.5,
        )  # fmt: skip
    for match, sub in table.groupby("match", sort=False):
        ax.text(
            0, sub["y"].min() - 0.85, match, ha="left", va="center",
            fontsize=style.FONT_SIZE_DENSE, fontweight="bold",
        )  # fmt: skip
    ax.set_yticks(table["y"], table["label"], fontsize=style.FONT_SIZE_DENSE - 0.5)
    ax.set_ylim(table["y"].max() + 0.6, table["y"].min() - 1.3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.set_xlabel("cells of the type active in the matched cccRISE component")
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(
        "Comparator cell types, cell by cell in cccRISE",
        fontsize=style.FONT_SIZE_TITLE,
    )
    ax.grid(False)


def makeFigure():
    return common.standalone(plot, (140 * style.MM, 95 * style.MM))
