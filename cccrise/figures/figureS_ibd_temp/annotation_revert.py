"""IBD draft: does a component the comparators recover depend on particular
fine types having their own label? Targeted labellings
(`run_comparator_fits`, scored by `run_granularity_sampling`): each cccRISE
component's LR containment in each comparator's span (colour; dots at
>= 0.85), one heatmap per method.

A. Lineage labels, with the Secretory types that carry C4-C6 (EE, EC, Tuft)
   broken out in every combination; the first row is the plain lineage
   labelling (s0).
B. Every fine type except the Secretory lineage's, which stays pooled apart
   from the same combinations; the last row is the full fine labelling.
   Going down from the row with all three, each row reverts one or more of
   them to their lineage label.
C. The same for the single-type components: Mono SOD2+ (C1, C7), ILC (C2,
   C3, C8) and Mast (C3), broken out of lineage labels (top) or kept apart
   from an otherwise pooled lineage in the fine labelling (bottom).

The last column of each heatmap is the method's selected LR rank for that
labelling (STACCato: its LR-mode rank): misses at a low rank are the rank
selection's as much as the labels'."""

import numpy as np
import pandas as pd

from .. import common, style
from ..figureS_lr_concordance._common import METHODS, SHORT
from ._sensitivity import CONTAINED, describe, labels, load
from .granularity_sampling import _lr_rank

COMBOS = ("", "+ee", "+ec", "+tuft", "+ee+ec", "+ee+tuft", "+ec+tuft", "+ee+ec+tuft")
ROWS = {
    "lineage": ["_s0r0"] + [f"_lin{c}" for c in COMBOS[1:]],
    "fine": [f"_fine-sec{c}" for c in COMBOS[::-1]] + ["_fine"],
    "single": [
        "_s0r0", "_lin+mono", "_lin+ilc", "_lin+mast",
        "_fine-mye", "_fine-mye+mono", "_fine-mye+mast", "_fine-t", "_fine-t+ilc", "_fine",
    ],
}  # fmt: skip


def _table():
    targeted = load("targeted_components")
    sampled = load("sampling_components")
    reference = sampled[sampled["labelling"].isin(["_s0r0", "_fine"])]
    return pd.concat([targeted, reference], ignore_index=True)


def _name(labelling: str) -> str:
    fixed = {"_s0r0": "lineage", "_fine": "fine (all types)"}
    return fixed[labelling] if labelling in fixed else describe(labelling)


def _rank_column(ax, table, rows, column):
    """Each row's LR rank as text in a column right of the heatmap."""
    ranks = _lr_rank(table).set_index("labelling")["r"]
    for y, labelling in enumerate(rows):
        ax.text(column, y, str(ranks.get(labelling, "")), ha="center", va="center",
                fontsize=style.FONT_SIZE_DENSE, color=style.COLORS["ink"])  # fmt: skip
    ax.set_xlim(-0.5, column + 0.5)


def _heatmaps(axes, table, rows, names, title, show_x):
    components = sorted(table["cccrise_component"].unique())
    for ax, method in zip(axes, METHODS, strict=True):
        grid = (
            table[table["method"] == method]
            .pivot(index="labelling", columns="cccrise_component", values="lr_containment")
            .reindex(index=rows, columns=components)
        )
        image = ax.imshow(
            grid, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="auto"
        )
        yy, xx = np.nonzero(grid.to_numpy() >= CONTAINED)
        ax.scatter(xx, yy, s=3, color="white", lw=0)
        _rank_column(ax, table[table["method"] == method], rows, len(components))
        ticks = [names[k] for k in components] + ["LR rank"]
        ax.set_xticks(
            range(len(ticks)), ticks if show_x else [], rotation=90,
            fontsize=style.FONT_SIZE_DENSE,
        )  # fmt: skip
        ax.set_yticks(range(len(rows)), [_name(r) for r in rows],
                      fontsize=style.FONT_SIZE_DENSE)  # fmt: skip
        ax.set_title(f"{SHORT[method]}: {title}", fontsize=style.FONT_SIZE_DENSE + 1)
        ax.grid(False)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    common.colorbar(axes[-1], image, "LR containment")


def plot(fig):
    table, names = _table(), labels()
    titles = {
        "lineage": "lineage + Secretory types",
        "fine": "fine, Secretory reverted",
        "single": "single-type components",
    }
    heights = [len(r) + 1.5 for r in ROWS.values()]
    heights[-1] += 5  # the component labels
    subfigs = fig.subfigures(len(ROWS), 1, height_ratios=heights)
    for i, (sub, (key, rows)) in enumerate(zip(subfigs, ROWS.items(), strict=True)):
        _heatmaps(sub.subplots(1, len(METHODS)), table, rows, names, titles[key],
                  show_x=i == len(ROWS) - 1)  # fmt: skip


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 190 * style.MM))
