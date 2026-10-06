"""Label-free broad cell type call per group: the AUROC with which the
group's Wilcoxon score against the rest of the cells (`S.rank_genes`) ranks
each cell type's canonical markers (`S.CELLTYPE_MARKERS`) above every other
gene. 0.5: the markers are no more enriched than a random gene set."""

import numpy as np
from sklearn.metrics import roc_auc_score

from ... import common, style
from .. import _common as S

AXES = ("short",)


def plot(axs):
    (ax,) = axs
    ranked = S.rank_genes(S.DE_GROUPS, "rest")
    auroc = np.array(
        [
            [
                roc_auc_score(ranked[g].index.isin(genes), ranked[g]["score"])
                for genes in S.CELLTYPE_MARKERS.values()
            ]
            for g in S.DE_GROUPS
        ]
    )
    groups = [g.capitalize() for g in S.DE_GROUPS]
    celltypes = list(S.CELLTYPE_MARKERS)
    image = ax.imshow(
        auroc, cmap=style.CMAPS["diverging"], vmin=0, vmax=1, aspect="auto"
    )
    for (i, j), value in np.ndenumerate(auroc):
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=common.FACTOR_LABEL_FONT_SIZE,
            color="white" if abs(value - 0.5) > 0.35 else style.COLORS["ink"],
        )
    ax.set_xticks(
        range(len(celltypes)),
        celltypes,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(groups)), groups)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    common.colorbar(ax, image, "Marker AUROC")


def makeFigure():
    return common.standalone_fixed(__name__)
