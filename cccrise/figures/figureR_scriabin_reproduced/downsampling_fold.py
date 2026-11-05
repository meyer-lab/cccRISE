"""Fig. 2e (right), pancreas islets (C1): how many-fold the counts were
downsampled to reach inDrop's mean coverage. Grey: the paper's bar, read
off its Fig. 2e."""

import pandas as pd

from .. import common, style
from . import _common as C

AXES = ("narrow",)
#: The paper's pancreas-islet bar, read off its Fig. 2e by eye: approximate
PAPER_FOLD = 270


def plot(axs):
    ax = axs[0]
    table = C.load("downsampling")
    fold = float(table.FoldDownsampled.iloc[0])
    ax.bar([0], [fold], 0.6, color=C.INK)
    ax.plot(0, PAPER_FOLD, "D", mfc="white", mec=C.PAPER, ms=style.MARKER_SIZE + 1)
    ax.set_xticks([0], ["Pancreas\nislets (C1)"])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 300)
    ax.set_ylabel("Fold downsampled to\nmean inDrop coverage")
    common.source_data(
        "fold_downsampled",
        pd.DataFrame(
            {
                "Source": ["This reproduction", "Paper, read off Fig. 2e"],
                "Fold downsampled": [fold, PAPER_FOLD],
                "Downsampled mean UMIs per cell": [
                    float(table.DownsampledMeanUmis.mean()),
                    float("nan"),
                ],
            }
        ),
        "Fig. 2e (right) of Wilk et al. 2024: the C1 islets' mean counts per cell over inDrop's "
        "mean of 5,828 UMIs per cell (the paper's figure), the factor by which each subsample "
        "was downsampled (scuttle::downsampleMatrix), and the downsampled subsamples' mean "
        "counts per cell (mean over the 12 subsamples). Paper row: its bar, read off the "
        "published figure by eye (approximate; no downsampled counts given).",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
