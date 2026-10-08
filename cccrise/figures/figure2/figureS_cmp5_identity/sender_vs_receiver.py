"""Genes differing between the sender and receiver groups: the difference in
the fraction of cells detecting each gene against the Wilcoxon BH q
(`S.rank_genes`). The `N_LABELED` most sender- and receiver-enriched
significant genes are labeled, the genes of the top LR pairs (`S.top_pair_genes`) outlined (and
labeled, bold, if significant), and the genes both groups share against the
rest (`de_genes.gene_sets`) colored `SHARED_COLOR` and labeled."""

import numpy as np
from adjustText import adjust_text

from ... import common, style
from .. import _common as S
from .de_genes import gene_sets

N_LABELED = 6
Q_FLOOR = 1e-30  # q below this drawn at it
HEADROOM = 1.5  # y limit over the highest point, room for labels
SHARED_COLOR = style.COLORS["highlight"]
SHARED_LABEL_X = 0.75  # shared gene labels' x, as a fraction of the x limit
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    ranking = S.rank_genes(("sender",), "receiver")["sender"]
    detected, _ = S.group_expression(ranking.index.tolist())
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
    lr = S.top_pair_genes()
    ax.scatter(
        delta[lr],
        y[lr],
        facecolors="none",
        edgecolors=style.COLORS["ink"],
        s=style.POINT_SIZE_FOREGROUND,
        linewidths=style.LINE_WIDTH_THIN,
    )
    shared = gene_sets()["Shared"]
    ax.scatter(
        delta[shared],
        y[shared],
        c=SHARED_COLOR,
        s=style.POINT_SIZE,
        linewidths=0,
        zorder=3,
    )
    common.reference_line(ax, -np.log10(common.SIG_ALPHA))
    common.reference_line(ax, 0, horizontal=False)
    limit = np.abs(delta).max() * 1.25
    ax.set_xlim(-limit, limit)
    ax.set_ylim(0, y.max() * HEADROOM)
    lr_significant = [g for g in lr if g in significant]
    labeled = dict.fromkeys(
        [
            *significant[:N_LABELED],
            *significant[::-1][:N_LABELED],
            *lr_significant,
        ]
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
    arrowprops = {
        "arrowstyle": "-",
        "lw": style.LINE_WIDTH_THIN / 2,
        "color": style.COLORS["reference"],
    }
    # Shared genes sit in the non-significant cloud: labeled in the empty
    # lower left, which the other labels then avoid.
    shared_texts = [
        ax.annotate(
            g,
            (delta[g], y[g]),
            xytext=(
                SHARED_LABEL_X * -limit,
                (0.5 + i) * -np.log10(common.SIG_ALPHA) / 2,
            ),
            ha="center",
            va="center",
            fontsize=common.FACTOR_LABEL_FONT_SIZE,
            color=SHARED_COLOR,
            arrowprops=arrowprops,
        )
        for i, g in enumerate(sorted(shared, key=lambda g: y[g]))
    ]
    adjust_text(
        texts,
        objects=shared_texts,
        x=delta.to_numpy(),
        y=y.to_numpy(),
        ax=ax,
        expand=(1.3, 1.6),
        force_text=(0.6, 1.0),
        force_pull=(0.002, 0.002),
        time_lim=10,
        arrowprops=arrowprops,
    )
    ax.set(
        xlabel="Detected fraction,\nsender − receiver",
        ylabel="−log$_{10}$ q (Wilcoxon)",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
