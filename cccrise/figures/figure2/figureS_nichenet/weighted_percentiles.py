"""The per-ligand values behind Figure 2's weight-averaged NicheNet target-score
percentiles: per CP component, every ligand's target-score percentile among
the component's ligands (``nichenet.score_percentiles``), each ligand counted
in proportion to its weight in the component -- box at the weighted
quartiles, whiskers at the weighted 5th and 95th percentiles, diamond at the
weighted mean (Figure 2's statistic). "Null": the same, pooled over every
component and every shuffled copy of the NicheNet prior
(``nichenet.shuffled_percentiles``)."""

import numpy as np

from ....validation.nichenet import null_weights, score_percentiles
from ... import common, style
from ..nichenet import BOX_STYLE, load_null, load_scores

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
NULL_LABEL = "Null"
AXES = ("rect",)


def weighted_box(pct: np.ndarray, w: np.ndarray) -> dict:
    """`Axes.bxp` stats of `pct` with each value counted in proportion to `w`
    (weighted quantiles at `QUANTILES`, weighted mean)."""
    order = np.argsort(pct)
    pct, w = pct[order], w[order]
    q = np.interp(QUANTILES, (np.cumsum(w) - w / 2) / w.sum(), pct)
    return dict(
        zip(("whislo", "q1", "med", "q3", "whishi"), q, strict=True),
        mean=w @ pct / w.sum(),
    )


def plot(axs):
    (ax,) = axs
    scores = load_scores()
    groups = {str(c): g for c, g in score_percentiles(scores).groupby("component")}
    groups[NULL_LABEL] = null_weights(load_null(), scores)
    stats = [
        weighted_box(g["score_pct"].to_numpy(), g["weight"].to_numpy())
        for g in groups.values()
    ]
    boxes = ax.bxp(
        stats,
        positions=range(len(stats)),
        showmeans=True,
        meanprops={
            "marker": "D",
            "markersize": style.MARKER_SIZE,
            "markerfacecolor": style.COLORS["emphasis"],
            "markeredgecolor": style.COLORS["emphasis"],
        },
        **BOX_STYLE,
    )
    boxes["boxes"][-1].set_facecolor(style.COLORS["muted"])
    common.reference_line(ax, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("CP component")
    ax.set_ylabel("Ligand target-score percentile")


def makeFigure():
    return common.standalone_fixed(__name__)
