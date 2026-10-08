"""Which cccRISE components the comparators recover, component-blind on the
comparator side: how much of each cccRISE LR column lies in the span of each
comparator fit's LR columns (``||P_X a_k||^2`` for unit `a_k`; 1 when some
combination of the fit's columns reproduces it). Rows: own-rank fits on the
coarse and fine annotation; columns: cccRISE components
(`cccrise-comparator-agreement`). `plot_reverse`: the other direction, each
comparator column's containment in the span of cccRISE's LR columns."""

import numpy as np
import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common, style
from ._common import ANNOTATIONS, METHODS, SHORT

SUFFIXES = {"coarse": "", "fine": "_fine"}
FITS = [(m, a) for a in ANNOTATIONS for m in METHODS]


def load(direction: str = "cccRISE in comparator") -> pd.DataFrame:
    """LR-mode containment of every cccRISE component in every own-rank fit
    (or, ``"comparator in cccRISE"``, of every fit's columns in cccRISE)."""
    frames = [
        common.load_runner_output(
            f"comparator_containment_alad{SUFFIXES[a]}.parquet",
            VALIDATION_DIR,
            hint="    uv run cccrise-comparator-agreement --dataset alad",
        ).assign(annotation=a)
        for a in ANNOTATIONS
    ]
    table = pd.concat(frames, ignore_index=True)
    return (
        table[
            (table["direction"] == direction)
            & (table["mode"] == "lr")
            & (table["rank_mode"] == "own")
        ]
        .set_index(["method", "annotation", "component"])
        .sort_index()
    )


def _grid(table, comps, column):
    return np.array([[table.loc[(*f, c), column] for c in comps] for f in FITS])


def _annotate(ax, values, size=style.FONT_SIZE_DENSE):
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=size,
                color="white" if v > 0.6 else style.COLORS["ink"],
            )  # fmt: skip


def _colorbar(fig, image, ax, label):
    fig.colorbar(
        image, ax=ax, label=label, location="bottom", shrink=0.8, pad=0.02
    )  # fmt: skip


def plot(fig):
    table = load()
    comps = sorted(table.index.get_level_values("component").unique())
    values = _grid(table, comps, "containment")
    ax = fig.subplots()
    image = ax.imshow(
        values, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="auto"
    )
    _annotate(ax, values)
    ax.set_yticks(range(len(FITS)), [f"{SHORT[m]} {a}" for m, a in FITS])
    ax.set_xticks(range(len(comps)), [f"C{c}" for c in comps])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    ax.axhline(len(METHODS) - 0.5, color="white", lw=3)
    ax.set_xlabel("cccRISE component")
    ax.set_title("Recovery of cccRISE components", fontsize=style.FONT_SIZE_TITLE)
    ax.grid(False)
    _colorbar(fig, image, ax, "LR containment in comparator span")


def _reverse_rows(table):
    """Each fit's columns in `FITS` order, a blank (NaN) row between fits:
    the column of values, the fit of every row (None for blanks) and the
    tick labels."""
    values, fits, labels = [], [], []
    for i, fit in enumerate(FITS):
        if i:
            values, fits, labels = [*values, np.nan], [*fits, None], [*labels, ""]
        sub = table.loc[fit, "containment"].sort_index()
        values += list(sub)
        fits += [fit] * len(sub)
        labels += [str(c) for c in sub.index]
    return np.array(values)[:, None], fits, labels


def plot_reverse(fig):
    values, fits, labels = _reverse_rows(load("comparator in cccRISE"))
    left, ax, right = fig.subplots(1, 3, width_ratios=(1, 0.45, 1))
    left.set_axis_off()
    right.set_axis_off()
    cmap = style.CMAPS["sequential"].with_extremes(bad="white")
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, aspect="auto"
    )
    _annotate(ax, values, size=style.FONT_SIZE_DENSE - 1.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=style.FONT_SIZE_DENSE - 1.5)
    for fit in FITS:
        rows = [i for i, f in enumerate(fits) if f == fit]
        ax.text(
            -0.25, np.mean(rows), f"{SHORT[fit[0]]}\n{fit[1]}", ha="right",
            va="center", transform=ax.get_yaxis_transform(),
            fontsize=style.FONT_SIZE_DENSE,
        )  # fmt: skip
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("Recovery of comparator columns", fontsize=style.FONT_SIZE_TITLE)
    ax.grid(False)
    fig.colorbar(
        image, ax=ax, label="LR containment\nin cccRISE span", location="bottom",
        ticks=[0, 0.5, 1], pad=0.02,
    )  # fmt: skip


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE / 2, 85 * style.MM))
