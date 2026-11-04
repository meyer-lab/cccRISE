"""Per-cell C3 and C4 sender weight of L-hormone+ EE, other EE and EC cells
(colored as `ee_state_map`, so without a legend), and each program's AUROC
for L-hormone+ against other EE cells: the weights track L-hormone
expression the annotation does not encode."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import common, style
from . import _common as S

AXES = ("square",)
EE = ["L-hormone+ EE", "Other EE"]


def auroc(cells: pd.DataFrame, k: int) -> float:
    ee = cells[cells["Group"].isin(EE)]
    return float(roc_auc_score(ee["Group"] == EE[0], ee[f"C{k} sender weight"]))


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells")
    cells = cells[cells["Group"].isin(list(S.EE_GROUPS))]
    common.source_data(
        "sender_weight_by_l_hormone_call",
        cells[["Group", "C3 sender weight", "C4 sender weight"]],
        "Unscaled per-cell C3 and C4 sender weight of every L-hormone+ EE (GCG, "
        "PYY or INSL5 detected), other EE and EC cell; AUROC for L-hormone+ vs "
        "other EE cells above each program.",
    )
    for i, k in enumerate((3, 4)):
        values = [
            cells.loc[cells["Group"] == g, f"C{k} sender weight"] for g in S.EE_GROUPS
        ]
        positions = i + 0.26 * (np.arange(len(values)) - 1)
        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=0.22,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": style.COLORS["ink"]},
        )
        for box, color in zip(boxes["boxes"], S.EE_GROUPS.values(), strict=True):
            box.set(facecolor=color, alpha=0.8, lw=style.LINE_WIDTH_THIN)
        ax.text(
            i,
            1,
            f"AUROC\n{auroc(cells, k):.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
        )
    common.reference_line(ax, 0)
    ax.set_xticks(
        [0, 1], ["C3\nPYY–\nNPY1R", "C4\nGCG–\nGLP1R"], fontsize=style.FONT_SIZE_DENSE
    )
    ax.set_ylabel("Sender weight")


def makeFigure():
    return common.standalone_fixed(__name__)
