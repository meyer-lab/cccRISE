"""Genes differing between the sender and receiver groups: the difference in
the fraction of cells detecting each gene against the Wilcoxon BH q
(`common.rank_genes`). The `N_LABELED` most sender- and receiver-enriched
significant genes are labeled, and the genes of the top LR pairs (`common.top_pair_genes`) outlined (and
labeled, bold, if significant)."""

import numpy as np
from adjustText import adjust_text

from .. import common, style

N_LABELED = 6
Q_FLOOR = 1e-30  # q below this drawn at it
HEADROOM = 1.5  # y limit over the highest point, room for labels
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    ranking = common.rank_genes(("sender",), "receiver")["sender"]
    detected, _ = common.group_expression(ranking.index.tolist())
    delta = detected["sender"] - detected["receiver"]
    y = -np.log10(np.clip(ranking["q"], Q_FLOOR, 1))
    significant = ranking.index[ranking["q"] < common.SIG_ALPHA]
    side = np.where(ranking["score"] > 0, "sender", "receiver")
    color = np.where(
        ranking["q"] < common.SIG_ALPHA,
        [style.COLORS[s] for s in side],
        style.COLORS["muted"],
    )
    ax.scatter(
        delta, y, c=color, s=style.POINT_SIZE_BACKGROUND, linewidths=0, rasterized=True
    )
    lr = common.top_pair_genes()
    ax.scatter(
        delta[lr],
        y[lr],
        facecolors="none",
        edgecolors=style.COLORS["ink"],
        s=style.POINT_SIZE_FOREGROUND,
        linewidths=style.LINE_WIDTH_THIN,
    )
    common.reference_line(ax, -np.log10(common.SIG_ALPHA))
    common.reference_line(ax, 0, horizontal=False)
    limit = np.abs(delta).max() * 1.25
    ax.set_xlim(-limit, limit)
    ax.set_ylim(0, y.max() * HEADROOM)
    lr_significant = [g for g in lr if g in significant]
    labeled = dict.fromkeys(
        [*significant[:N_LABELED], *significant[::-1][:N_LABELED], *lr_significant]
    )
    texts = [
        ax.text(
            delta[g],
            y[g],
            g,
            fontsize=common.FACTOR_LABEL_FONT_SIZE,
            fontweight="bold" if g in lr else "normal",
        )
        for g in labeled
    ]
    adjust_text(
        texts,
        x=delta.to_numpy(),
        y=y.to_numpy(),
        ax=ax,
        expand=(1.3, 1.6),
        force_text=(0.6, 1.0),
        force_pull=(0.002, 0.002),
        time_lim=10,
        arrowprops={
            "arrowstyle": "-",
            "lw": style.LINE_WIDTH_THIN / 2,
            "color": style.COLORS["reference"],
        },
    )
    ax.set(
        xlabel="Detected fraction,\nsender − receiver",
        ylabel="−log$_{10}$ q (Wilcoxon)",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
