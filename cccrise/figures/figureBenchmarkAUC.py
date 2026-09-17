"""
Figure BenchmarkAUC: ALAD-status AUC for every benchmarked method.

Panels (a) the forest plot of cross-validated AUC per method, with its
permutation null where one was computed, and (b) the AUC of each method under
each classifier, which is the sensitivity check behind the D1 choice of L1
logistic regression.

Reads `benchmark_auc_<dataset>.csv`; produce it with

    uv run cccrise-bench-auc --dataset alad --n-permutations 200
"""

import numpy as np
import seaborn as sns

from .common import getSetup, load_runner_csv, subplotLabel

DATASET = "alad"


def makeFigure():
    df = load_runner_csv(f"benchmark_auc_{DATASET}.csv")

    ax, f = getSetup((9, 4), (1, 2))
    subplotLabel(ax)

    primary = df[df["Classifier"] == df["Classifier"].iloc[0]].copy()
    primary = primary.sort_values("AUC", ascending=True)
    y = np.arange(len(primary))

    # (a) Forest plot: point, interval, chance line, and the permutation null.
    ax[0].errorbar(
        primary["AUC"],
        y,
        xerr=[
            (primary["AUC"] - primary["AUCLo"]).clip(lower=0),
            (primary["AUCHi"] - primary["AUC"]).clip(lower=0),
        ],
        fmt="o",
        color="black",
        capsize=3,
        markersize=5,
        linewidth=1.2,
    )
    if primary["NullAUC"].notna().any():
        ax[0].scatter(
            primary["NullAUC"],
            y,
            marker="|",
            s=80,
            color="grey",
            label="permuted-label null",
        )
        ax[0].legend(loc="lower right")
    ax[0].axvline(0.5, color="firebrick", linestyle=":", linewidth=1)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(
        [f"{lab} (r={int(r)})" for lab, r in zip(primary["Label"], primary["Rank"])]
    )
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_xlabel("ALAD-status AUC (cross-validated)")
    ax[0].set_title(
        f"{primary['Classifier'].iloc[0]}, {int(primary['NConditions'].iloc[0])} patients"
    )

    # (b) Same methods under every classifier the runner scored.
    sns.barplot(data=df, x="Label", y="AUC", hue="Classifier", ax=ax[1])
    ax[1].axhline(0.5, color="firebrick", linestyle=":", linewidth=1)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_xlabel("")
    ax[1].set_ylabel("ALAD-status AUC")
    ax[1].set_title("Classifier sensitivity")
    ax[1].tick_params(axis="x", rotation=30)

    return f
