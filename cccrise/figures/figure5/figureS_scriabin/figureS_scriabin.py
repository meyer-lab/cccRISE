"""Supplementary figure: why Scriabin is not a comparator. Seed-to-seed
agreement of Scriabin's interaction programs (upstream R, on CCC-RISE's
inputs) beside that of CCC-RISE's components' carrying LR pairs, ALAD and
IBD: each point a pair of seeds, each bar their median (``uv run
cccrise-scriabin-seeds``)."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ....runners.datasets import SCRIABIN_DIR, scores_filename
from ... import common, style
from .._common import DATASET_LABELS, METHOD_COLORS

AXES = ("scriabin_seeds",)
#: Each method's color and offset within its cohort
METHODS = {
    "cccRISE": ("CCC-RISE", METHOD_COLORS["cccRISE"], -0.18),
    "Scriabin": ("Scriabin", style.CATEGORICAL[9], 0.18),
}
JITTER = 0.08


def plot(axs):
    (ax,) = axs
    rng = np.random.default_rng(0)
    frames = []
    for x, (dataset, label) in enumerate(DATASET_LABELS.items()):
        t = common.load_runner_output(
            scores_filename("seed_agreement", dataset),
            SCRIABIN_DIR,
            f"uv run cccrise-scriabin-seeds --dataset {dataset}",
        )
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
        "Scriabin's merged interaction programs (Stage programs; upstream R runs on "
        "CCC-RISE's inputs, every sample of more cells than Scriabin subsamples) or "
        "of CCC-RISE's components' carrying LR pairs between seed refits (Stage "
        "lr_sets).",
    )
    ax.set_xticks(range(len(DATASET_LABELS)), list(DATASET_LABELS.values()))
    ax.set(
        xlim=(-0.6, len(DATASET_LABELS) - 0.4),
        ylim=(0, 1.05),
        yticks=[0, 0.5, 1],
        ylabel="Seed-to-seed LR-set Jaccard",
    )
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=3.5, color=color, label=name)
            for name, color, _ in METHODS.values()
        ],
        loc="center right",
        fontsize=style.FONT_SIZE_DENSE,
        handletextpad=0.2,
        frameon=False,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
