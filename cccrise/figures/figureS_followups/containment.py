"""Containment of each cccRISE component in each comparator fit's span
(condition and LR modes), with the comparators fit on the raw, winsorised and
block-normed pseudobulk tensor, at their own rank rule and at rank 5. Stars:
contained (>= 0.85 and above the row-permutation null)."""

import numpy as np
import pandas as pd

from .. import common, style
from ._common import METHODS, SHORT, VARIANT_LABELS, load

MODES = {"condition": "Condition mode", "lr": "LR mode"}
KEYS = ["method", "rank_mode", "variant"]
SORT = {
    "method": {m: i for i, m in enumerate(METHODS)},
    "rank_mode": {"own": 0, "shared": 1},
    "variant": {v: i for i, v in enumerate(VARIANT_LABELS)},
}


def _order(table: pd.DataFrame) -> pd.MultiIndex:
    """(method, rank mode, variant) rows in display order."""
    fits = table[KEYS].drop_duplicates()
    fits = fits.sort_values(KEYS, key=lambda c: c.map(SORT[c.name]))
    return pd.MultiIndex.from_frame(fits)


def _annotate(ax, values: np.ndarray, calls: np.ndarray) -> None:
    for (i, j), v in np.ndenumerate(values):
        ax.text(
            j,
            i,
            f"{v:.2f}" + ("*" if calls[i, j] else ""),
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1.5,
            color="white" if v > 0.6 else style.COLORS["ink"],
        )


def plot(fig):
    table = load("comparator_containment")
    table = table[table["direction"] == "cccRISE in comparator"]
    order = _order(table)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (mode, title) in zip(axes, MODES.items(), strict=True):
        sub = table[table["mode"] == mode]
        values, calls = (
            sub.pivot_table(index=KEYS, columns="component", values=v)
            .reindex(order)
            .to_numpy(float)
            for v in ("containment", "contained")
        )
        image = ax.imshow(
            values, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="auto"
        )
        _annotate(ax, values, calls > 0.5)
        ax.set_xticks(
            range(values.shape[1]), [f"C{k + 1}" for k in range(values.shape[1])]
        )
        ax.set_title(title)
        ax.grid(False)
    labels = [
        f"{SHORT[m]} {'own' if r == 'own' else 'rank 5'}, {VARIANT_LABELS[v]}"
        for m, r, v in order
    ]
    axes[0].set_yticks(range(len(labels)), labels, fontsize=style.FONT_SIZE_DENSE - 1)
    fig.colorbar(image, ax=axes, label="containment", shrink=0.6)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
