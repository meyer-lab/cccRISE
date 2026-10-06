"""Exploratory: which cells carry each cccRISE component, and whether the same
cells send and receive. A cell is active when its loading (sender `sc_B`,
receiver `rc_C`; scaled by sqrt(n_cells) of its sample, so samples of
different sizes share one scale) exceeds half the component's 99th
percentile. Left, middle: the fraction of each cell type's cells active as
sender, as receiver. Right: the fraction active as sender, receiver or both.
Where the right value is close to the larger of the left two, the same cells
send and receive; where it is close to their sum, different cells do.
Components shared with the comparators (at their own ranks) are starred."""

import numpy as np
import pandas as pd

from ...comparators.agreement import shared_calls
from .. import common, style
from ._common import load


def _shared_components() -> set[int]:
    containment = load("containment")
    calls = shared_calls(
        containment[containment["rank_mode"] == "own"], ["condition", "lr"]
    )
    calls = calls[
        (calls["direction"] == "cccRISE in comparator") & calls["contained_all_modes"]
    ]
    return set(calls["component"])


def _heatmap(ax, values: pd.DataFrame, labels, title):
    image = ax.imshow(
        values.to_numpy(float),
        cmap=style.CMAPS["sequential"],
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for (i, j), v in np.ndenumerate(values.to_numpy(float)):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if v > 0.6 else "0.2",
        )
    ax.set_xticks(range(values.shape[1]), values.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.grid(False)
    return image


def plot(fig):
    overlap = load("cell_overlap")
    shared = _shared_components()
    types = sorted(overlap["celltype"].unique())
    components = sorted(overlap["component"].unique())
    labels = [f"C{k}{' *' if k in shared else ''}" for k in components]

    def grid(column):
        return overlap.pivot(
            index="component", columns="celltype", values=column
        ).reindex(index=components, columns=types)

    axes = fig.subplots(1, 3, sharey=True)
    _heatmap(axes[0], grid("sender_frac"), labels, "Sender")
    _heatmap(axes[1], grid("receiver_frac"), labels, "Receiver")
    image = _heatmap(axes[2], grid("either_frac"), labels, r"Sender $\cup$ Receiver")
    fig.colorbar(image, ax=axes, label="Active fraction", shrink=0.8)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
