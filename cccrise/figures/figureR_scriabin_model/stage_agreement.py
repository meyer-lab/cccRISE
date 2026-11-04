"""Seed-to-seed agreement at each step of Scriabin's program discovery,
every run of A: each point the median over seed pairs (of the median over
subsampled groups, for the per-group steps). The shaded step, the module
partition, is where agreement breaks; programs are the method's output."""

import pandas as pd
from matplotlib.lines import Line2D

from .. import common, style
from . import _common as C

AXES = ("tall_narrow",)
STAGES = {
    "similarity": "LR similarity",
    "tom": "TOM",
    "modules": "Modules",
    "programs": "Programs",
}
#: The step shaded as where agreement breaks
BREAK = "modules"


def _per_pair(rows: pd.DataFrame) -> pd.Series:
    """Each stage's per-pair values: the median over groups for the
    per-group stages."""
    return rows.groupby(["Stage", "SeedA", "SeedB"]).Value.median()


def _source_data(rows: pd.DataFrame) -> None:
    frame = pd.DataFrame(
        {
            "Dataset": [
                C.DATASET_LABELS[(d, g)]
                for d, g in zip(rows.Dataset, rows.Grouping, strict=True)
            ],
            "Preprocessing": rows.Preprocessing,
            "Step": rows.Stage.map(STAGES),
            "Group": rows.Group,
            "Seed A": rows.SeedA,
            "Seed B": rows.SeedB,
            "Agreement": rows.Value,
        }
    )
    common.source_data(
        "step_agreement_by_seed_pair",
        frame,
        "Agreement between two seeds' upstream R runs at each step of program discovery, for the "
        "subsampled samples or cell types (Group). LR similarity: Pearson r of the averaged LR-pair "
        "similarity over the pairs both runs keep. TOM: Pearson r of the signed topological overlap at "
        "each run's soft power. Modules: best-match Jaccard of the group's modules (dynamic tree cut). "
        "Programs: best-match Jaccard of the programs merged across groups (Jaccard 0.15; no Group). "
        "Point drawn: median over seed pairs of the per-pair value (median over groups where Group is "
        "given). Preprocessing: cccrise = cccRISE's with CellChatDB; sct_alra / sct = SCTransform + ALRA / "
        "SCTransform with OmniPath.",
    )


def plot(axs):
    ax = axs[0]
    table = C.seed_agreement()
    table = table[table.Method == "Scriabin"]
    x = range(len(STAGES))
    for run in C.SCRIABIN_RUNS:
        dataset, grouping, preprocessing = run
        medians = (
            _per_pair(C.run_rows(table, *run, method="Scriabin"))
            .groupby("Stage")
            .median()
        )
        marker, ls = C.DATASET_STYLES[C.DATASET_LABELS[(dataset, grouping)]]
        ax.plot(
            x,
            medians.reindex(list(STAGES)).to_numpy(),
            color=C.PREPROCESSING_COLORS[preprocessing],
            marker=marker,
            ms=style.MARKER_SIZE,
            ls=ls,
            lw=style.LINE_WIDTH * 1.1,
            alpha=0.95,
        )
    ax.set_xticks(
        x, list(STAGES.values()), rotation=35, ha="right", rotation_mode="anchor"
    )
    ax.get_xticklabels()[-1].set_fontweight("bold")
    b = list(STAGES).index(BREAK)
    ax.axvspan(
        b - 0.3, b + 0.3, color=style.COLORS["muted"], alpha=0.35, lw=0, zorder=0
    )
    ax.set_xlim(-0.25, len(STAGES) - 0.75)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Seed-to-seed agreement")
    ax.set_xlabel("Scriabin step")
    handles = [
        Line2D(
            [],
            [],
            color="0.35",
            marker=m,
            ms=style.MARKER_SIZE,
            ls=ls,
            lw=style.LINE_WIDTH * 1.1,
            label=label,
        )
        for label, (m, ls) in C.DATASET_STYLES.items()
    ]
    ax.legend(
        handles=handles,
        title="Dataset",
        loc="lower left",
        alignment="left",
        fontsize=style.FONT_SIZE_DENSE,
        title_fontsize=style.FONT_SIZE_DENSE,
        handlelength=2.0,
        labelspacing=0.15,
        borderaxespad=0.1,
        borderpad=0.1,
    )
    _source_data(
        pd.concat(
            [C.run_rows(table, *run, method="Scriabin") for run in C.SCRIABIN_RUNS]
        )
    )


def makeFigure():
    return common.standalone_fixed(__name__)
