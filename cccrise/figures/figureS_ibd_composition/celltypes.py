"""Each fine cell type's share of all cells against inflamed vs healthy:
Spearman rho and BH q, colored by lineage; carriers of cccRISE's programs and
the strongest shift each way labeled."""

import numpy as np

from .. import common, style
from . import _common

AXES = ("rect",)
ALPHA = 0.05


def plot(axs):
    (ax,) = axs
    rows = _common.table("composition_celltypes")
    rows = rows[(rows["level"] == "fine") & (rows["share"] == "all cells")]
    colors = _common.lineage_colors(rows["lineage"])
    score = -np.log10(rows["q"])
    for lineage, color in colors.items():
        mask = rows["lineage"] == lineage
        ax.scatter(rows.loc[mask, "rho"], score[mask], s=style.POINT_SIZE, color=color, lw=0,
                   label=lineage)  # fmt: skip
    ax.axhline(-np.log10(ALPHA), color=style.COLORS["reference"], ls=style.LINE_STYLE_REFERENCE,
               lw=style.LINE_WIDTH_THIN)  # fmt: skip
    # the strongest shift each way, and the carriers of cccRISE's programs
    labeled = {rows.loc[rows["rho"].idxmin(), "celltype"], rows.loc[rows["rho"].idxmax(), "celltype"]}
    labeled |= set(_common.CARRIERS) & set(rows["celltype"])
    for row in rows[rows["celltype"].isin(labeled)].itertuples():
        ax.annotate(row.celltype, (row.rho, -np.log10(row.q)), xytext=(3, 2), textcoords="offset points",
                    fontsize=style.FONT_SIZE_DENSE - 1)  # fmt: skip
    ax.set_xlabel("Spearman $\\rho$, share of all cells\nvs inflamed (1) / healthy (0)")
    ax.set_ylabel("$-\\log_{10}$ q")
    common.legend(ax, "right", fontsize=style.FONT_SIZE_DENSE)


def makeFigure():
    return common.standalone_fixed(__name__)
