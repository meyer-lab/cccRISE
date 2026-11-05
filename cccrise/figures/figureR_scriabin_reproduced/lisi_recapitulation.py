"""Fig. 2e (left), pancreas islets (C1): the percentage of ground-truth
cell-cell pairs recapitulated by a downsampled pair (LISI > 1) after
integrating each downsampled CCIM with the full-depth one, per random
300-cell subsample, colored by the ground-truth pairs' mean LISI. Grey: the
paper's boxes for the same arms, read off its Fig. 2e."""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import common, style
from . import _common as C

AXES = ("square",)
ARMS = {"raw": "Raw", "alra": "ALRA-\ndenoised"}
#: The paper's Fig. 2e pancreas-islet boxes (median, lowest and highest
#: point, %), read off the published figure by eye: approximate
PAPER_PERCENT = {"raw": (72, 67, 76), "alra": (91, 89, 94)}
#: Mean-LISI color scale, red (1) to green (2) through olive, as the paper's
LISI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "lisi", ["#8b1a00", "#808000", "#3fcf1a"]
)
LISI_NORM = mcolors.Normalize(1.1, 1.7)
OFFSET = 0.3  # the paper's box, right of ours


def _source_data(table: pd.DataFrame) -> None:
    ours = pd.DataFrame(
        {
            "Source": "This reproduction",
            "Arm": table.Arm.map(ARMS).str.replace("\n", ""),
            "Subsample": table.Subsample.astype("Int64"),
            "GT pairs with LISI > 1 (%)": 100 * table.FractionLisiAbove1,
            "Mean LISI of GT pairs": table.MeanLisi,
            "GT pairs integrated": table.Pairs.astype("Int64"),
        }
    )
    paper = pd.DataFrame(
        [
            {
                "Source": f"Paper, read off Fig. 2e ({stat})",
                "Arm": ARMS[arm].replace("\n", ""),
                "GT pairs with LISI > 1 (%)": value,
            }
            for arm, values in PAPER_PERCENT.items()
            for stat, value in zip(("median", "lowest", "highest"), values, strict=True)
        ]
    )
    common.source_data(
        "lisi_recapitulation_by_subsample",
        pd.concat([ours, paper], ignore_index=True),
        "Fig. 2e (left) of Wilk et al. 2024 rerun on its Fluidigm C1 pancreas islets (panc8): "
        "per random 300-cell subsample, the full-depth counts (GT) and the counts downsampled "
        "to inDrop coverage (Raw), or downsampled and ALRA-denoised, each through SCTransform and "
        "upstream GenerateCCIM (OmniPath); the downsampled CCIM integrated with the GT CCIM "
        "(Seurat anchors, 30 PCs) on the same GT pairs (at most 10,000 per subsample), and the "
        "LISI of each GT pair (1: only GT pairs nearby; 2: evenly mixed). Drawn: the percentage of "
        "GT pairs with LISI > 1, colored by their mean LISI. Paper rows: the median, lowest and "
        "highest point of the paper's pancreas-islet boxes, read off its published Fig. 2e by "
        "eye (approximate).",
    )


def _paper_boxes(ax) -> None:
    for i, arm in enumerate(ARMS):
        median, low, high = PAPER_PERCENT[arm]
        x = i + OFFSET
        ax.plot([x, x], [low, high], color=C.PAPER, lw=style.LINE_WIDTH)
        ax.plot(x, median, "D", mfc="white", mec=C.PAPER, ms=style.MARKER_SIZE + 1)


def plot(axs):
    ax = axs[0]
    table = C.load("downsampling")
    values = [100 * table[table.Arm == arm].FractionLisiAbove1 for arm in ARMS]
    C.boxplot(ax, values, width=0.4)
    for i, arm in enumerate(ARMS):
        rows = table[table.Arm == arm]
        points = ax.scatter(
            C.jitter(np.full(len(rows), i), 0.1, seed=i),
            100 * rows.FractionLisiAbove1,
            c=rows.MeanLisi,
            cmap=LISI_CMAP,
            norm=LISI_NORM,
            s=style.POINT_SIZE,
            lw=0,
            zorder=3,
        )
    _paper_boxes(ax)
    ax.set_xticks(range(len(ARMS)), list(ARMS.values()))
    ax.set_xlim(-0.5, len(ARMS) - 0.5 + OFFSET)
    ax.set_ylim(60, 101)
    ax.set_ylabel("GT cell–cell pairs\nwith LISI > 1 (%)")
    ax.set_title("Fig. 2e · pancreas islets (C1)", fontsize=style.FONT_SIZE_LABEL)
    common.colorbar(ax, points, "Mean LISI of\nGT pairs")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=C.INK,
                marker="o",
                ls="",
                ms=style.MARKER_SIZE,
                label="Ours",
            ),
            Line2D(
                [],
                [],
                color=C.PAPER,
                marker="D",
                mfc="white",
                ms=style.MARKER_SIZE + 1,
                label="Paper (≈)",
            ),
        ],
        loc="lower right",
        fontsize=style.FONT_SIZE_DENSE,
        handletextpad=0.2,
        borderaxespad=0.2,
    )
    _source_data(table)


def makeFigure():
    return common.standalone_fixed(__name__)
