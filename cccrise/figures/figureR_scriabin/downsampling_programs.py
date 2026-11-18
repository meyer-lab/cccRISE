"""Added to Fig. 2e: whether the same interaction programs come back on the
same C1 subsamples. Best-match Jaccard of upstream InteractionPrograms'
significant programs (its defaults, then upstream's significance test as the
vignette applies it) between the full-depth ground truth and a downsampled
draw, raw or ALRA, and between two independent downsampling draws of the
same cells; one point per subsample. At 300 cells InteractionPrograms uses
the full CCIM, so two draws differ only through the downsampling's random
draw (and the test's random programs)."""

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
                "Significant programs (first)": table.ProgramsA.astype("Int64"),
                "Significant programs (second)": table.ProgramsB.astype("Int64"),
                "Program agreement": table.Agreement,
                "Programs before the test (first)": table.AllProgramsA.astype("Int64"),
                "Programs before the test (second)": table.AllProgramsB.astype("Int64"),
                "Program agreement before the test": table.AgreementAllPrograms,
            }
        ),
        "Per random 300-cell subsample of the Fluidigm C1 islets (as in A). GT vs raw / "
        "ALRA: full depth against downsampling draw 1, raw or ALRA-denoised; draw vs draw: "
        "draws 1 and 2 of the same cells, both raw or both ALRA. Upstream "
        "InteractionPrograms (cell_types = celltype; its default soft power, R2 0.6 with "
        "fallback to 1) on the full-depth counts (GT) and on two independent downsampling "
        "draws to inDrop coverage, each raw (SCTransform) or ALRA-denoised; then upstream "
        "InteractionProgramSignificance (500 random programs, the vignette's; seeded by the subsample) "
        "on each run's programs in its own similarity matrix, keeping those with p < 0.05 "
        "as the vignette does. Program agreement: best-match Jaccard of the two runs' "
        "significant programs (each program's best Jaccard in the other run, averaged over "
        "both runs' programs); before the test: of all their programs. Programs: the "
        "number in each run (first: GT or draw 1). At 300 cells InteractionPrograms uses "
        "the full CCIM (no cell subsampling), so draw 1 and draw 2 differ only through "
        "scuttle::downsampleMatrix's random draw (seeded 1000 x subsample + draw).",
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
    ax.set_xticks(range(len(COMPARISONS)), list(COMPARISONS.values()))
    ax.set_xlim(-0.5, len(COMPARISONS) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Program agreement\n(best-match Jaccard)")
    ax.set_title(
        "Interaction Program Consistency Between Draws", fontsize=style.FONT_SIZE_LABEL
    )
    _source_data(table)


def makeFigure():
    return common.standalone_fixed(__name__)
