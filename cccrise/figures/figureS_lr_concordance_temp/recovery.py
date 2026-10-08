"""Which cccRISE components the comparators recover (draft panel A).

Per cccRISE component and comparator fit (own rank; coarse and fine
annotation), the fit's column sharing the most of the component's top-10 LR
pairs (ties: the larger comparator loading on them). Number: pairs shared,
starred above the abundance-weighted null's 95th percentile. Colour: the
share of that column's squared loading the shared pairs carry. Below the
number: that column's heaviest pair (grey when not starred)."""

import numpy as np
import pandas as pd

from .. import common, style
from ..figureS_lr_concordance._common import (
    ANNOTATIONS,
    METHODS,
    SHORT,
    load,
    top_pair,
)

K = 10
FITS = [(a, m) for a in ANNOTATIONS for m in METHODS]


def best(table: pd.DataFrame) -> pd.DataFrame:
    """Per (cccRISE component, annotation, method): the column with the most
    shared top-`K` pairs, then the largest comparator loading on them."""
    table = table[(table["k"] == K) & table["method"].isin(METHODS)]
    table = table.sort_values(
        ["shared", "mass_comparator"], ascending=False, kind="stable"
    )
    keys = ["cccrise_component", "annotation", "method"]
    return table.drop_duplicates(keys).set_index(keys).sort_index()


def _grid(rows, comps, name):
    return np.array([[rows.loc[(c, *fit), name] for fit in FITS] for c in comps])


def _annotate(ax, rows, comps, mass, pairs):
    for i, c in enumerate(comps):
        for j, (annotation, method) in enumerate(FITS):
            row = rows.loc[(c, annotation, method)]
            color = (
                "white"
                if mass[i, j] > 0.6
                else style.COLORS["ink"]
                if row["hit"]
                else style.COLORS["muted"]
            )
            ax.text(
                j, i - 0.12, f"{int(row['shared'])}{'*' if row['hit'] else ''}",
                ha="center", va="center", color=color,
                fontsize=style.FONT_SIZE_DENSE + 1, fontweight="bold",
            )  # fmt: skip
            key = (annotation, method, int(row["comparator_component"]))
            ax.text(
                j, i + 0.25,
                pairs.get(key, "").replace("^", "–"),
                ha="center", va="center", color=color,
                fontsize=style.FONT_SIZE_DENSE - 1.5,
            )  # fmt: skip


def plot(fig):
    rows = best(load("overlap"))
    comps = sorted(rows.index.get_level_values(0).unique())
    mass = _grid(rows, comps, "mass_comparator")
    ax = fig.subplots()
    image = ax.imshow(mass, cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto")
    _annotate(ax, rows, comps, mass, top_pair(load("columns")))
    ax.set_xticks(range(len(FITS)), [f"{SHORT[m]}\n{a}" for a, m in FITS])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(range(len(comps)), [f"C{c}" for c in comps])
    ax.axvline(len(METHODS) - 0.5, color="white", lw=3)
    ax.set_ylabel("cccRISE component")
    ax.grid(False)
    fig.colorbar(
        image, ax=ax, label="comparator loading\non shared pairs", shrink=0.8, pad=0.01
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 70 * style.MM))
