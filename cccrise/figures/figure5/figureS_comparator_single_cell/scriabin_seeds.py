"""Seed-to-seed best-match Jaccard of Scriabin's significant interaction
programs and of CCC-RISE components' carrying LR pairs, ALAD cohort.

Why Scriabin is not a comparator (`figureS_comparator_single_cell` A). Seed-to-seed
agreement of Scriabin's significant interaction programs (upstream R on
CCC-RISE's inputs, then upstream's significance test as its vignette applies
it) beside that of CCC-RISE's components' carrying LR pairs, in ALAD: each
point a pair of seeds, each bar their median (``uv run cccrise-scriabin-seeds
--dataset alad``). IBD is not shown: its significance test is not run (a
seed's test on IBD's merged programs is ~70 CPU-hours at 500 random programs)."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ....runners.datasets import SCRIABIN_DIR, scores_filename
from ... import common, style
from .._common import DATASET_LABELS, METHOD_COLORS

AXES = ("seed_strip",)
#: Each method's color and offset within its cohort
METHODS = {
    "cccRISE": ("CCC-RISE", METHOD_COLORS["cccRISE"], -0.18),
    "Scriabin": ("Scriabin", style.COLORS["reference"], 0.18),
}
JITTER = 0.08
#: Each method's rows of the seed-agreement table
STAGES = {"cccRISE": "lr_sets", "Scriabin": "significant_programs"}
DATASETS = ("alad",)


def plot(axs):
    (ax,) = axs
    rng = np.random.default_rng(0)
    frames = []
    for x, dataset in enumerate(DATASETS):
        label = DATASET_LABELS[dataset]
        t = common.load_runner_output(
            scores_filename("seed_agreement", dataset),
            SCRIABIN_DIR,
            f"uv run cccrise-scriabin-seeds --dataset {dataset}",
        )
        t = t[t["Stage"] == t["Method"].map(STAGES)]
        frames.append(t.assign(Cohort=label))
        for method, (_, color, offset) in METHODS.items():
            values = t.loc[t["Method"] == method, "Value"].to_numpy()
            at = x + offset
            ax.scatter(
                at + rng.uniform(-JITTER, JITTER, len(values)),
                values,
                s=style.POINT_SIZE_BACKGROUND,
                color=color,
                alpha=0.7,
                lw=0,
                zorder=2,
            )
            ax.plot(
                [at - 0.13, at + 0.13],
                [np.median(values)] * 2,
                color=style.COLORS["ink"],
                lw=1.0,
                zorder=3,
            )
    t = pd.concat(frames, ignore_index=True)
    common.source_data(
        "seed_to_seed_lr_set_jaccard",
        t.rename(columns={"Value": "Jaccard", "SeedA": "Seed A", "SeedB": "Seed B"})[
            ["Cohort", "Method", "Stage", "Seed A", "Seed B", "Jaccard"]
        ],
        "Per pair of seeds: the best-match Jaccard (mean over both directions) of "
        "Scriabin's merged interaction programs that pass upstream's significance test "
        "(Stage significant_programs; upstream R runs on CCC-RISE's inputs, every sample "
        "of more cells than Scriabin subsamples, merged, then InteractionProgramSignificance "
        "with 500 random programs in each sample, kept if any sample gives p < 0.05) "
        "or of CCC-RISE's components' carrying LR pairs between seed refits (Stage "
        "lr_sets).",
    )
    ax.set_xticks(range(len(DATASETS)), [DATASET_LABELS[d] for d in DATASETS])
    ax.set(
        xlim=(-0.6, len(DATASETS) - 0.4),
        ylim=(0, 1.4),
        yticks=[0, 0.5, 1],
        ylabel="Seed-to-seed\nLR-set Jaccard",
    )
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=3.5, color=color, label=name)
            for name, color, _ in METHODS.values()
        ],
        loc="upper center",
        fontsize=style.FONT_SIZE_DENSE,
        handletextpad=0.2,
        frameon=False,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
