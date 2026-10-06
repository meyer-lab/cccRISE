"""Per clinical association, each method's strongest component as a maxT
permutation p (10,000 patient-block shuffles; family-wise across the method's
components), on coarse (filled) and fine (open) cell types."""

import numpy as np

from .. import common, style
from ._common import METHODS, SHORT, load

MARKERS = {"coarse": "o", "fine": "D"}


def _draw(ax, null, granularity: str, order: list[str], offset: float) -> None:
    """One granularity's best-component maxT p per method and association."""
    colors = dict(zip(METHODS, style.CATEGORICAL, strict=False))
    for k, method in enumerate(METHODS):
        rows = null[null["Method"] == method].set_index("Association")
        if rows.empty or (granularity == "fine" and method == "cccRISE"):
            continue
        ax.scatter(
            -np.log10(rows.reindex(order)["PermP"]),
            np.arange(len(order)) + (k - 1.5) * 0.15 + offset,
            marker=MARKERS[granularity],
            s=14,
            facecolor=colors[method] if granularity == "coarse" else "none",
            edgecolor=colors[method],
            label=SHORT.get(method, method)
            + ("" if method == "cccRISE" else f" ({granularity})"),
        )


def plot(fig):
    ax = fig.subplots()
    tables = {g: load("anova_null", g) for g in ("coarse", "fine")}
    order = list(dict.fromkeys(tables["coarse"]["Association"]))
    for offset, (granularity, null) in zip((0, 0.05), tables.items(), strict=True):
        if null is not None:
            _draw(ax, null, granularity, order, offset)
    ax.axvline(-np.log10(0.05), color=style.COLORS["reference"], ls="--", lw=0.8)
    ax.set_yticks(range(len(order)), order, fontsize=style.FONT_SIZE_DENSE)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}$ maxT permutation p (best component)")
    ax.legend(
        fontsize=style.FONT_SIZE_DENSE - 1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
