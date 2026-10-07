"""Top-10 LR-pair overlap of every cccRISE component with every comparator
column (own rank; c = coarse, f = fine types; labelled by the column's
heaviest pair). Number: pairs shared, starred above the abundance-weighted
null's 95th percentile. Colour: the share of the comparator column's squared
loading those shared pairs carry. Last column: the 10 most abundant pairs."""

import numpy as np

from .. import common, style
from ._common import column_label, column_order, load, top_pair

K = 10


def _grids(table, keys, comps):
    """(shared count, comparator mass on shared, hit) per cccRISE component x
    comparator column, with the abundance reference as the last column."""
    index = table.set_index(
        ["annotation", "method", "comparator_component", "cccrise_component"]
    )
    ref = table[(table["method"] == "abundance") & (table["annotation"] == "coarse")]
    ref = ref.set_index("cccrise_component").loc[comps]
    extra = {"shared": ref["shared"], "mass_comparator": np.nan, "hit": ref["hit"]}
    return [
        np.column_stack(
            [
                np.array([[index.loc[(*k, c), name] for k in keys] for c in comps]),
                np.broadcast_to(np.asarray(extra[name], dtype=float), len(comps)),
            ]
        )
        for name in ("shared", "mass_comparator", "hit")
    ]


def _annotate(ax, shared, mass, hit):
    for (i, j), n in np.ndenumerate(shared):
        ax.text(
            j,
            i,
            f"{int(n)}{'*' if hit[i, j] else ''}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if mass[i, j] > 0.8 else style.COLORS["ink"],
        )


def plot(fig):
    table = load("overlap")
    table = table[table["k"] == K]
    keys = column_order(table)
    comps = sorted(table["cccrise_component"].unique())
    shared, mass, hit = _grids(table, keys, comps)
    ax = fig.subplots()
    image = ax.imshow(mass, cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto")
    _annotate(ax, shared, mass, hit.astype(bool))
    pairs = top_pair(load("columns"))
    labels = [column_label(k, pairs) for k in keys] + ["10 most abundant"]
    ax.set_xticks(
        range(len(labels)), labels, rotation=90, fontsize=style.FONT_SIZE_DENSE - 1
    )
    ax.set_yticks(range(len(comps)), [f"C{c}" for c in comps])
    boundaries = [i for i in range(1, len(keys)) if keys[i][:2] != keys[i - 1][:2]]
    for x in [*boundaries, len(keys)]:
        ax.axvline(x - 0.5, color="white", lw=1.5)
    ax.set_ylabel("cccRISE component")
    ax.grid(False)
    fig.colorbar(
        image, ax=ax, label="comparator loading\non shared pairs", shrink=0.8, pad=0.01
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
