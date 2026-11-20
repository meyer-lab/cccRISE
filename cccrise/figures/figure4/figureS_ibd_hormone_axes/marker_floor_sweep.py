"""Mann-Whitney Benjamini-Hochberg q of each hormone state marker gene between
villus-like and crypt-like EE cells across magnitude floors.

Figure 4K's marker tests (`hormone_extreme_markers`: villus-like vs
crypt-like Mann-Whitney, BH q across its genes) redone at every magnitude
floor in `FLOORS` (`_common.hormone_extremes`), and the `N_WORST` genes whose
largest q over the sweep is highest drawn against the floor."""

import numpy as np
import pandas as pd

from ... import common, style
from .. import _common as S
from ..hormone_extreme_markers import EXTREMES, SECTIONS, extreme_genes

AXES = ("square",)
#: Up to 1, where 6 villus-like cells remain
FLOORS = np.round(np.arange(0, 1.001, 0.05), 2)
N_WORST = 6
MARKER_SCALE = 0.6  # of `style.MARKER_SIZE`: 21 floors per gene


def floor_sweep() -> pd.DataFrame:
    """Each `SECTIONS` gene's test at each of `FLOORS`, with the cells per
    class."""
    cells = S.programs("enteroendocrine_cells")
    section = {g: name for name, genes in SECTIONS.items() for g in genes}
    tables = []
    for floor in FLOORS:
        state = S.hormone_extremes(cells, floor)
        counts = {f"{name} cells": (state == name).sum() for name in EXTREMES}
        tests = extreme_genes(cells, state)
        tables.append(
            tests.assign(Floor=floor, Section=tests["Gene"].map(section), **counts)
        )
    sweep = pd.concat(tables, ignore_index=True)
    return sweep[["Floor", "Gene", "Section", *counts, "p", "q"]]


def plot(axs):
    (ax,) = axs
    sweep = floor_sweep()
    common.source_data(
        "hormone_extreme_markers_by_floor",
        sweep,
        "Each Figure 4K gene's Mann-Whitney p between villus-like and crypt-like "
        "EE cells, and its BH q across the genes, with the classes' magnitude "
        f"floor (Figure 4K: {S.EXTREME_FLOOR}) at each value from {FLOORS[0]} to "
        f"{FLOORS[-1]}, and the cells in each class at that floor.",
    )
    worst = sweep.groupby("Gene")["q"].max().nlargest(N_WORST).index
    for gene, color in zip(worst, style.CATEGORICAL, strict=False):
        rows = sweep[sweep["Gene"] == gene]
        ax.plot(
            rows["Floor"],
            rows["q"],
            marker="o",
            markersize=MARKER_SCALE * style.MARKER_SIZE,
            color=color,
            linewidth=style.LINE_WIDTH_THIN,
            label=gene,
        )
    line = {"color": style.COLORS["ink"], "linewidth": style.LINE_WIDTH_THIN}
    ax.axhline(
        common.SIG_ALPHA,
        linestyle=style.LINE_STYLE_REFERENCE,
        label=f"q = {common.SIG_ALPHA}",
        **line,
    )
    ax.axvline(S.EXTREME_FLOOR, linestyle=":", label="Fig. 4K floor", **line)
    ax.set_yscale("log")
    ax.set_xlabel("Magnitude floor", fontsize=style.FONT_SIZE_DENSE)
    ax.set_ylabel("q, villus- vs crypt-like")
    common.legend(ax, "right", fontsize=style.FONT_SIZE_DENSE, handlelength=1.5)


def makeFigure():
    return common.standalone_fixed(__name__)
