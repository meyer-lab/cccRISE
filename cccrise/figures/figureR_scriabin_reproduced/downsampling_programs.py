"""Added to Fig. 2e: whether the same interaction programs come back on the
same C1 subsamples. Best-match Jaccard of upstream InteractionPrograms'
programs (its default power) between the full-depth ground truth and a
downsampled draw, raw or ALRA, and between two independent downsampling
draws of the same cells; one point per subsample. Grey: the Jaccard range
the paper calls "generally stable"."""

import numpy as np
import pandas as pd

from .. import common, style
from . import _common as C

AXES = ("rect",)
#: Each comparison's tick label (and, newlines dropped, its source-data name)
COMPARISONS = {
    "gt_vs_downsampled": "GT vs\nraw",
    "gt_vs_downsampled_alra": "GT vs\nALRA",
    "two_draws": "Draw vs\ndraw\n(raw)",
    "two_draws_alra": "Draw vs\ndraw\n(ALRA)",
}


def _source_data(table: pd.DataFrame) -> None:
    common.source_data(
        "program_agreement_by_subsample",
        pd.DataFrame(
            {
                "Comparison": table.Comparison.map(COMPARISONS).str.replace("\n", " "),
                "Subsample": table.Subsample.astype("Int64"),
                "Programs (first)": table.ProgramsA.astype("Int64"),
                "Programs (second)": table.ProgramsB.astype("Int64"),
                "Program agreement": table.Agreement,
            }
        ),
        "Per random 300-cell subsample of the Fluidigm C1 islets (as in A). GT vs raw / "
        "ALRA: full depth against downsampling draw 1, raw or ALRA-denoised; draw vs draw: "
        "draws 1 and 2 of the same cells, both raw or both ALRA. Upstream "
        "InteractionPrograms (cell_types = celltype; its default soft power, R2 0.6 with "
        "fallback to 1) on the full-depth counts (GT) and on two independent downsampling "
        "draws to inDrop coverage, each raw (SCTransform) or ALRA-denoised. Program "
        "agreement: best-match Jaccard of the two runs' programs (each program's best "
        "Jaccard in the other run, averaged over both runs' programs). Programs: the "
        "number in each run (first: GT or draw 1). The paper's 'generally stable' range "
        f"drawn is its ED Fig. 8a mean Jaccard, {C.STABLE_BAND[0]}-{C.STABLE_BAND[1]}.",
    )


def plot(axs):
    ax = axs[0]
    table = C.load("downsampling_programs")
    values = [table[table.Comparison == c].Agreement for c in COMPARISONS]
    C.boxplot(ax, values, width=0.45)
    for i, v in enumerate(values):
        ax.scatter(
            C.jitter(np.full(len(v), i), 0.12, seed=10 + i),
            v,
            s=style.POINT_SIZE,
            color=C.INK,
            alpha=0.7,
            lw=0,
            zorder=3,
        )
    C.stable_band(ax)
    ax.set_xticks(range(len(COMPARISONS)), list(COMPARISONS.values()))
    ax.set_xlabel("Downsampled to inDrop depth: raw or ALRA")
    ax.set_xlim(-0.5, len(COMPARISONS) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Program agreement\n(best-match Jaccard)")
    ax.set_title("Programs, same cells", fontsize=style.FONT_SIZE_LABEL)
    _source_data(table)


def makeFigure():
    return common.standalone_fixed(__name__)
