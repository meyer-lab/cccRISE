"""Exploratory: which cells carry each cccRISE component. A cell is active
when its loading (sender `sc_B`, receiver `rc_C`; scaled by sqrt(n_cells) of
its sample, so samples of different sizes share one scale) exceeds half the
component's 99th percentile. Top: the fraction of each cell type's cells that
are active, over all samples. Bottom: the same inside the component's
dominant sample (largest condition loading; cell counts in brackets).
Components shared with the comparators are starred: the question is whether
those are carried by every cell of one type, which a cell-type mean sees
unchanged, while the others are carried by part of a type."""

import numpy as np
import pandas as pd

from ...comparators.agreement import shared_calls
from .. import common, style
from ._common import load


def _shared_components() -> set[int]:
    calls = shared_calls(load("containment"), ["condition", "lr"])
    calls = calls[
        (calls["direction"] == "cccRISE in comparator") & calls["contained_all_modes"]
    ]
    return set(calls["component"])


def _heatmap(ax, values: pd.DataFrame, counts: pd.DataFrame | None, labels, title):
    image = ax.imshow(
        values.to_numpy(float),
        cmap=style.CMAPS["sequential"],
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for (i, j), v in np.ndenumerate(values.to_numpy(float)):
        if not np.isfinite(v):
            continue
        text = f"{v:.2f}" if counts is None else f"{v:.2f}\n({counts.iat[i, j]:.0f})"
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if v > 0.6 else "0.2",
        )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.grid(False)
    return image


def plot(fig):
    overall = load("cell_fractions")
    dominant = load("sample_fractions")
    shared = _shared_components()
    types = sorted(overall["celltype"].unique())
    components = sorted(overall["component"].unique())
    samples = dominant.groupby("component")["sample"].first()
    labels = [f"C{k}{' *' if k in shared else ''}" for k in components]
    dom_labels = [
        f"{lab} {samples[k].replace('ALAD1-', '').replace('-E1', '')}"
        for lab, k in zip(labels, components, strict=True)
    ]
    axes = fig.subplots(2, 2, sharex=True)
    for col, side in enumerate(("sender", "receiver")):
        sub = overall[overall["side"] == side]
        values = sub.pivot(
            index="component", columns="celltype", values="active_frac_0.5"
        )
        _heatmap(
            axes[0, col],
            values[types],
            None,
            labels,
            f"{side.capitalize()}: active fraction, all samples",
        )
        dom = dominant[dominant["side"] == side]
        values = dom.pivot(index="component", columns="celltype", values="active_frac")
        counts = dom.pivot(index="component", columns="celltype", values="n_cells")
        image = _heatmap(
            axes[1, col],
            values.reindex(index=components, columns=types),
            counts.reindex(index=components, columns=types),
            dom_labels,
            f"{side.capitalize()}: inside the dominant sample",
        )
        axes[1, col].set_xticks(range(len(types)), types, rotation=40, ha="right")
    fig.colorbar(
        image, ax=axes, label="Active fraction of the type's cells", shrink=0.7
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 2 * style.PANEL_HEIGHT))
