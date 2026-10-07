"""Head against tail: for each comparator column (own rank, coarse and fine)
and its best-matching cccRISE component, the fraction of their top-k pairs
they share (left; dashed: abundance null 95th percentile) and the share of
each column's squared loading those shared pairs carry (right)."""

import numpy as np

from .. import common, style
from ._common import load

KS = (5, 10, 25)


def _strip(ax, values, x, color, offset=0.0):
    rng = np.random.default_rng(0)
    xs = x + offset + rng.uniform(-0.08, 0.08, len(values))
    ax.scatter(xs, values, s=style.POINT_SIZE_BACKGROUND, color=color, alpha=0.6, lw=0)
    ax.plot(
        [x + offset - 0.15, x + offset + 0.15],
        [np.median(values)] * 2,
        color=color,
        lw=style.LINE_WIDTH * 1.5,
    )


def plot(fig):
    matches = load("matches")
    null = load("null").set_index("k")
    left, right = fig.subplots(1, 2)
    for x, k in enumerate(KS):
        _strip(left, matches[f"shared_k{k}"] / k, x, style.COLORS["ink"])
        left.plot(
            [x - 0.3, x + 0.3],
            [null.loc[k, "null_q95"] / k] * 2,
            ls=style.LINE_STYLE_REFERENCE,
            color=style.COLORS["reference"],
            lw=style.LINE_WIDTH_THIN,
        )
        _strip(
            right, matches[f"mass_comparator_k{k}"], x, style.COLORS["receiver"], -0.17
        )
        _strip(right, matches[f"mass_cccrise_k{k}"], x, style.COLORS["sender"], 0.17)
    for ax in (left, right):
        ax.set_xticks(range(len(KS)), [f"top-{k}" for k in KS])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(False)
    left.set_ylabel("pairs shared (fraction of k)")
    right.set_ylabel("loading on shared pairs")
    right.scatter(
        [], [], color=style.COLORS["receiver"], s=10, label="comparator column"
    )
    right.scatter([], [], color=style.COLORS["sender"], s=10, label="cccRISE component")
    right.legend(fontsize=style.FONT_SIZE_DENSE, loc="lower right")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
