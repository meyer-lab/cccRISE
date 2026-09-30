"""
Figure S1e_g: Cell type proportions across conditions.

TODO(IBD): VERIFY. Ported from the retired BALF cohort's severity ranking;
the IBD ordering Healthy < CD_uninflamed < CD_inflamed in `SEVERITY` is a
proposal, not a published ranking.
"""

import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .common import (
    IBD_SPEC,
    get_ibd_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotGeneral import compute_cell_type_proportions

SEVERITY = {"Healthy": 1, "CD_uninflamed": 2, "CD_inflamed": 3}


def makeFigure():
    ax, f = getSetup((4, 4), (2, 2))
    subplotLabel(ax)

    # Only cell metadata is needed, which the factors file stores.
    X = get_ibd_data(project_cells=False)
    df = compute_cell_type_proportions(
        X,
        celltype_col=IBD_SPEC.celltype_column,
        sample_col=IBD_SPEC.condition_column,
        condition_col=IBD_SPEC.group_column,
    ).rename(
        columns={
            IBD_SPEC.celltype_column: "celltype",
            IBD_SPEC.condition_column: "sample",
            IBD_SPEC.group_column: "condition",
        }
    )
    print(df)

    sns.boxplot(
        data=df, x="condition", y="proportion", hue="celltype", ax=ax[0], palette="Set3"
    )
    ax[0].set_yscale("log")
    ax[0].set_title("Cell Type Proportions by Condition")
    ax[0].set_ylim(0.0001, 1.5)

    df["severity"] = df["condition"].map(SEVERITY).astype(int)

    correlation_results = []
    for celltype in df["celltype"].unique():
        subset = df[df["celltype"] == celltype]
        if (
            len(subset["severity"].unique()) > 1
        ):  # Ensure there's variability in severity
            corr = spearmanr(subset["proportion"], subset["severity"])[0]
            pval = spearmanr(subset["proportion"], subset["severity"])[1]
            correlation_results.append((celltype, corr, pval))
    corr_df = pd.DataFrame(
        correlation_results, columns=pd.Index(["celltype", "correlation", "pvalue"])
    ).dropna()
    print(corr_df)
    sns.scatterplot(
        data=corr_df,
        x="correlation",
        y="pvalue",
        ax=ax[1],
        hue="celltype",
        palette="Set3",
    )
    ax[1].set_title("Correlation of Cell Type Proportion with Severity")
    ax[1].set_xlabel("Spearman Correlation")
    ax[1].set_ylabel("P-value")
    ax[1].axhline(0.05, color="red", linestyle="--")
    ax[1].set_yscale("log")
    ax[1].set_xlim(-1, 1)

    sample_totals = (
        df.groupby(["sample", "condition"], observed=False)["count"]
        .sum()
        .reset_index(name="total_cells")
    )
    pal = sns.color_palette("Set2")
    pal = [pal[0], pal[1], pal[2]]
    sns.scatterplot(
        data=sample_totals,
        x="sample",
        y="total_cells",
        hue="condition",
        ax=ax[3],
        palette=pal,
    )
    ax[3].set_title("Total Cells per Sample")
    ax[3].set_xticks([])  # hundreds of samples; their IDs aren't legible
    ax[3].set_ylabel("Total Cell Count")

    return f
