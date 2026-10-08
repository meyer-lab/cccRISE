"""Which cccRISE components the comparators recover, component-blind on the
comparator side: how much of each cccRISE LR column lies in the span of each
comparator fit's LR columns (``||P_X a_k||^2`` for unit `a_k`; 1 when some
combination of the fit's columns reproduces it). Rows: own-rank fits on the
coarse and fine annotation; columns: cccRISE components. Boxed: contained
(>= 0.85 and above the row-permutation null's 95th percentile;
`cccrise-comparator-agreement`)."""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ...runners.datasets import VALIDATION_DIR
from .. import common, style
from ._common import ANNOTATIONS, METHODS, SHORT

SUFFIXES = {"coarse": "", "fine": "_fine"}
FITS = [(m, a) for a in ANNOTATIONS for m in METHODS]


def load() -> pd.DataFrame:
    """LR-mode containment of every cccRISE component in every own-rank fit."""
    frames = [
        common.load_runner_output(
            f"comparator_containment_alad{SUFFIXES[a]}.parquet",
            VALIDATION_DIR,
            hint="    uv run cccrise-comparator-agreement --dataset alad",
        ).assign(annotation=a)
        for a in ANNOTATIONS
    ]
    table = pd.concat(frames, ignore_index=True)
    return table[
        (table["direction"] == "cccRISE in comparator")
        & (table["mode"] == "lr")
        & (table["rank_mode"] == "own")
    ].set_index(["method", "annotation", "component"])


def _grid(table, comps, column):
    return np.array([[table.loc[(*f, c), column] for c in comps] for f in FITS])


def _annotate(ax, values, contained):
    for (i, j), v in np.ndenumerate(values):
        ax.text(
            j, i, f"{v:.2f}", ha="center", va="center",
            fontsize=style.FONT_SIZE_DENSE,
            color="white" if v > 0.6 else style.COLORS["ink"],
        )  # fmt: skip
        if contained[i, j]:
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.2,
                    edgecolor=style.COLORS["ink"],
                )
            )  # fmt: skip


def plot(fig):
    table = load()
    comps = sorted(table.index.get_level_values("component").unique())
    values = _grid(table, comps, "containment")
    ax = fig.subplots()
    image = ax.imshow(
        values, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="auto"
    )
    _annotate(ax, values, _grid(table, comps, "contained").astype(bool))
    ax.set_yticks(range(len(FITS)), [f"{SHORT[m]} {a}" for m, a in FITS])
    ax.set_xticks(range(len(comps)), [f"C{c}" for c in comps])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    ax.axhline(len(METHODS) - 0.5, color="white", lw=3)
    ax.set_xlabel("cccRISE component")
    ax.set_title("Recovery of cccRISE components", fontsize=style.FONT_SIZE_TITLE)
    ax.grid(False)
    fig.colorbar(
        image,
        ax=ax,
        label="LR containment in comparator span",
        location="bottom",
        shrink=0.8,
        pad=0.02,
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE / 2, 85 * style.MM))
