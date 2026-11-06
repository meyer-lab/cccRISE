"""How each comparator's coarse LR columns map onto its fine ones."""

import numpy as np
import pandas as pd

from .. import common, style
from ._common import COMPARATORS, METHOD_COLORS
from ._programs import lr_matrix
from ._recovery import panel_title

#: Columns are joined at this cos^2 or more.
FLOW_FROM = 0.09
BLOCK_STEP = 1.75
NAME_SIZE = style.FONT_SIZE_DENSE - 2.8


def _unit(dataset: str, method: str, labelling: str, pairs) -> pd.DataFrame:
    columns = lr_matrix(dataset, method, labelling).reindex(pairs).fillna(0)
    return columns / np.linalg.norm(columns.to_numpy(), axis=0, keepdims=True)


def _cos2(dataset: str, method: str) -> pd.DataFrame:
    """A comparator's coarse x fine LR cos^2, on cccRISE's LR pairs."""
    pairs = lr_matrix(dataset, "cccRISE").index
    coarse = _unit(dataset, method, "coarse", pairs)
    fine = _unit(dataset, method, "fine", pairs)
    return pd.DataFrame(
        (coarse.to_numpy().T @ fine.to_numpy()) ** 2,
        index=coarse.columns,
        columns=fine.columns,
    )


def _names(ax, x: float, ys, columns, method: str, side: int) -> None:
    color = METHOD_COLORS[method]
    ax.scatter([x] * len(ys), ys, s=3, color=color, lw=0, zorder=3)
    for y, column in zip(ys, columns, strict=True):
        ax.text(
            x + 0.07 * side,
            y,
            f"{method[0]}{column}",
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=NAME_SIZE,
        )


def _block(ax, method: str, cos2: pd.DataFrame, x: float) -> None:
    values = cos2.to_numpy()
    rows = np.linspace(1, 0, len(cos2)) if len(cos2) > 1 else np.array([0.5])
    columns = np.empty(cos2.shape[1])
    columns[np.lexsort((-values.max(axis=0), values.argmax(axis=0)))] = np.linspace(
        1, 0, cos2.shape[1]
    )
    for (i, j), value in np.ndenumerate(values):
        if value >= FLOW_FROM:
            ax.plot(
                [x, x + 1],
                [rows[i], columns[j]],
                color=METHOD_COLORS[method],
                lw=0.25 + 1.4 * value,
                alpha=0.75,
                solid_capstyle="butt",
                zorder=2,
            )
    _names(ax, x, rows, cos2.index, method, -1)
    _names(ax, x + 1, columns, cos2.columns, method, 1)
    ax.text(x + 0.5, 1.07, method, ha="center", fontsize=style.FONT_SIZE_DENSE - 0.5)
    for position, letter in ((x, "C"), (x + 1, "F")):
        ax.text(
            position, -0.11, letter, ha="center", fontsize=style.FONT_SIZE_DENSE - 1
        )


def match_label(ax) -> None:
    """Name what the panels' lines mean, under `ax`."""
    ax.text(
        0.5,
        -0.04,
        "LR cos$^2$ match",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=style.FONT_SIZE_LABEL,
    )


def coarse_to_fine(ax, dataset: str, title: str) -> None:
    """Each comparator's coarse LR columns (left) joined to its fine ones
    (right) where cos^2 >= `FLOW_FROM`, line width growing with cos^2."""
    tables = {m: _cos2(dataset, m) for m in COMPARATORS}
    common.source_data(
        "coarse_to_fine_lr_cos2",
        pd.concat(
            t.rename_axis(index="Coarse column", columns="Fine column")
            .stack()
            .rename("LR cos2")
            .reset_index()
            .assign(Method=m)
            for m, t in tables.items()
        ),
        f"cos^2 between each comparator's coarse and fine LR columns on {title}, on "
        f"CCC-RISE's LR pairs; a line is drawn at {FLOW_FROM} or more.",
    )
    for i, (method, cos2) in enumerate(tables.items()):
        _block(ax, method, cos2, BLOCK_STEP * i)
    ax.set(xlim=(-0.38, BLOCK_STEP * len(tables) - 0.37), ylim=(-0.15, 1.15))
    ax.set_axis_off()
    panel_title(ax, title, pad=2)
