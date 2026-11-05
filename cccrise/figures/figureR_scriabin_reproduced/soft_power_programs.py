"""Added to Extended Data Fig. 8f: whether the same interaction programs
come back between random inDrop subsamples of one size (300 cells left,
1,000 right), against the R^2 threshold: the median (line) and interquartile
range (band) over every pair of subsamples, at upstream's soft power (1
where none reaches the threshold or it is above 3) and uncapped. Dotted:
the default threshold, 0.6. Grey: the Jaccard range the paper calls
"generally stable"."""

import pandas as pd

from .. import common, style
from . import _common as C

AXES = ("rect", "rect")
CELLS = (300, 1000)
#: Each power rule's label and color
RULES = {
    "default": ("Shipped default (power ≤ 3, else 1)", C.INK),
    "uncapped": ("Uncapped power", style.COLORS["emphasis"]),
}
DEFAULT_R2 = 0.6


def _source_data(table: pd.DataFrame) -> None:
    common.source_data(
        "program_agreement_by_subsample_pair",
        pd.DataFrame(
            {
                "Cells": table.Cells.astype("Int64"),
                "R2 threshold": table.R2,
                "Power rule": table.Rule.map({k: v[0] for k, v in RULES.items()}),
                "Subsample A": table.SubsampleA.astype("Int64"),
                "Subsample B": table.SubsampleB.astype("Int64"),
                "Program agreement": table.Agreement,
            }
        ),
        "For every pair of the 35 random inDrop subsamples of one size (as in D-K), at each "
        "R2 threshold: best-match Jaccard of their upstream InteractionPrograms programs, "
        "cut at the soft power each rule gives (shipped default: WGCNA's recommended power, "
        "1 where none reaches the threshold or it is above 3; uncapped: 1 only where none "
        "reaches it). Drawn: the median (line) and interquartile range (band) over subsample "
        f"pairs. Dotted: the default threshold, {DEFAULT_R2}. The paper's 'generally stable' "
        f"range drawn is its ED Fig. 8a mean Jaccard, {C.STABLE_BAND[0]}-{C.STABLE_BAND[1]}.",
    )


def _draw(ax, rows: pd.DataFrame) -> None:
    for rule, (label, color) in RULES.items():
        q = rows[rows.Rule == rule].groupby("R2").Agreement.quantile([0.25, 0.5, 0.75])
        q = q.unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.2, lw=0)
        ax.plot(q.index, q[0.5], color=color, lw=style.LINE_WIDTH, label=label)
    C.stable_band(ax, label_x=0.02)
    ax.axvline(DEFAULT_R2, ls=":", color=C.PAPER, lw=style.LINE_WIDTH_THIN)


def plot(axs):
    table = C.load("soft_power_programs")
    for ax, cells in zip(axs, CELLS, strict=True):
        _draw(ax, table[table.Cells == cells])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("Minimum scale-free topology fit (R²)")
        ax.set_title(
            f"Programs between subsamples of {cells:,} cells",
            fontsize=style.FONT_SIZE_LABEL,
        )
    axs[0].set_ylabel("Program agreement\n(best-match Jaccard)")
    axs[0].legend(loc="lower right", fontsize=style.FONT_SIZE_DENSE, borderaxespad=0.3)
    _source_data(table)


def makeFigure():
    return common.standalone_fixed(__name__)
