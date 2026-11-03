"""How robustly the C3 and C4 sender weights separate L-hormone+ from other
EE cells (Figure 5H's AUROC): within tertiles of genes detected per cell,
beside detection depth alone (dashed), and within each inflammation status."""

import pandas as pd
from sklearn.metrics import roc_auc_score

from ... import common, style
from .. import _common as S

AXES = ("square", "square")
TERTILES = ["low", "mid", "high"]


def aurocs() -> tuple[pd.DataFrame, float]:
    cells = S.programs("enteroendocrine_cells")
    ee = cells[cells["Group"].isin(["L-hormone+ EE", "Other EE"])].copy()
    ee["L-hormone+"] = ee["Group"] == "L-hormone+ EE"
    ee["Depth tertile"] = pd.qcut(ee["Genes detected"], 3, labels=TERTILES)
    rows = [
        {
            "Stratum": column,
            "Level": level,
            "Component": k,
            "Cells": len(sub),
            "AUROC": roc_auc_score(sub["L-hormone+"], sub[f"C{k} sender weight"]),
        }
        for column, levels in (
            ("Depth tertile", TERTILES),
            ("Status", list(S.GROUP_LABELS)),
        )
        for level in levels
        for k in (3, 4)
        if len(sub := ee[ee[column].astype(str) == level])
    ]
    return pd.DataFrame(rows), float(
        roc_auc_score(ee["L-hormone+"], ee["Genes detected"])
    )


def plot(axs):
    table, depth = aurocs()
    common.source_data(
        "l_hormone_auroc_by_stratum",
        table,
        "AUROC of each program's sender weight for L-hormone+ vs other EE cells "
        f"within tertiles of genes detected and within each status; genes detected "
        f"alone give {depth:.2f} (dashed).",
    )
    for ax, (column, levels) in zip(
        axs,
        (("Depth tertile", TERTILES), ("Status", list(S.GROUP_LABELS))),
        strict=True,
    ):
        for k, color in S.HORMONE_COLORS.items():
            rows = table[(table["Stratum"] == column) & (table["Component"] == k)]
            ax.plot(
                range(len(levels)),
                rows["AUROC"],
                marker="o",
                color=color,
                label=f"C{k}",
            )
        if column == "Depth tertile":
            common.reference_line(ax, depth, label="Depth alone")
        ax.set_ylim(0.5, 1)
        ax.set_ylabel("AUROC, L-hormone+\nvs other EE")
        ax.legend(fontsize=style.FONT_SIZE_DENSE, loc="lower right")
    axs[0].set_xticks(range(3), TERTILES)
    axs[0].set_xlabel("Genes detected (tertile)")
    axs[1].set_xticks(
        range(3),
        list(S.GROUP_LABELS.values()),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
