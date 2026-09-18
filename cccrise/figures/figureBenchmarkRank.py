"""
Figure BenchmarkRank: ALAD-status AUC as a function of CP rank.

The quantitative answer to Reviewer #1's comment #4. Panel (a) is the AUC curve
per method with its bootstrap interval and its permuted-label null; panel (b) is
the gap between the two, which is the part of the AUC that is not explained by
simply having more components to fit.

The vertical marker is the rank the registry selected by CORCONDIA, which never
saw the phenotype. Each method's curve stops at its own cap: CP rank cannot
exceed the tensor's smallest mode, which is the cell-type count (8) for the
pseudobulk tensor and the RISE rank (19) for CCC-RISE.

Reads `benchmark_rank_<dataset>.csv`; produce it with

    uv run cccrise-bench-rank --dataset alad --ranks 1:20 --n-permutations 25
"""

import seaborn as sns

from .common import getSetup, load_runner_csv, subplotLabel

DATASET = "alad"


def makeFigure():
    df = load_runner_csv(f"benchmark_rank_{DATASET}.csv").sort_values(
        ["Method", "Rank"]
    )

    ax, f = getSetup((10, 4), (1, 2))
    subplotLabel(ax)

    palette = dict(
        zip(
            df["Label"].unique(), sns.color_palette("colorblind", df["Label"].nunique())
        )
    )

    # (a) AUC vs rank, with interval and null.
    for label, sub in df.groupby("Label"):
        color = palette[label]
        ax[0].plot(
            sub["Rank"], sub["AUC"], "-o", color=color, label=label, markersize=4
        )
        ax[0].fill_between(
            sub["Rank"],
            sub["AUCLo"],
            sub["AUCHi"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        if sub["NullAUC"].notna().any():
            ax[0].plot(
                sub["Rank"],
                sub["NullAUC"],
                "--",
                color=color,
                alpha=0.6,
                linewidth=1,
                label=f"{label} (permuted)",
            )
    ax[0].axhline(0.5, color="firebrick", linestyle=":", linewidth=1)

    selected = df[df["IsSelectedRank"]]
    if len(selected):
        rank = int(selected["Rank"].iloc[0])
        ax[0].axvline(rank, color="grey", linestyle="-.", linewidth=1)
        ax[0].annotate(
            f"selected CP rank {rank}\n(CORCONDIA, phenotype-blind)",
            xy=(rank, 0.04),
            xytext=(rank + 0.6, 0.06),
            fontsize=6,
            color="grey",
        )
    ax[0].set_xlabel("CP rank")
    ax[0].set_ylabel("ALAD-status AUC (cross-validated)")
    ax[0].set_ylim(0.0, 1.0)
    ax[0].legend(fontsize=6, loc="upper left")
    ax[0].set_title("AUC vs CP rank")

    # (b) AUC minus its own permuted null: the part rank alone does not buy.
    if df["NullAUC"].notna().any():
        excess = df.dropna(subset=["NullAUC"]).copy()
        excess["Excess"] = excess["AUC"] - excess["NullAUC"]
        for label, sub in excess.groupby("Label"):
            ax[1].plot(
                sub["Rank"],
                sub["Excess"],
                "-o",
                color=palette[label],
                label=label,
                markersize=4,
            )
        ax[1].axhline(0.0, color="firebrick", linestyle=":", linewidth=1)
        ax[1].set_xlabel("CP rank")
        ax[1].set_ylabel("AUC - permuted-label null")
        ax[1].legend(fontsize=6)
        ax[1].set_title("Excess over chance at the same rank")
    else:
        ax[1].axis("off")
        ax[1].text(
            0.5,
            0.5,
            "no permutation null in this CSV\n(re-run with --n-permutations)",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
        )

    return f
