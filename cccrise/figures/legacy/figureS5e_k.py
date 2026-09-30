"""
Figure S1e_g: Cell type proportions in BALF ALAD data and ratios of DCs to CD4 T cells.
"""

import seaborn as sns
from scipy.stats import spearmanr

from ..common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotGeneral import compute_cell_type_proportions


def makeFigure():
    ax, f = getSetup((4, 4), (2, 2))
    subplotLabel(ax)

    X = get_alad_data(project_cells=True)
    sample_id = "dsco_id"
    condition_id = "ALADstatus"
    celltype_id = "broad_cell_type"

    df = compute_cell_type_proportions(
        X, celltype_col=celltype_id, sample_col=sample_id, condition_col=condition_id
    )
    print(df)

    sns.boxplot(
        data=df,
        x=condition_id,
        y="proportion",
        hue=celltype_id,
        ax=ax[0],
        palette="Set3",
    )
    ax[0].set_yscale("log")
    ax[0].set_title("Cell Type Proportions by Condition")
    ax[0].set_ylim(0.0001, 1.5)

    # Look at ratio of DCs to CD4 T cells across conditions
    df_dc_cd4 = df[df[celltype_id].isin(["Dendritic Cells", "CD4 T cells"])]
    df_dc_cd4_pivot = df_dc_cd4.pivot_table(
        index=[sample_id, condition_id],
        columns=celltype_id,
        values="count",
        fill_value=0,
        observed=False,
    ).reset_index()

    df_dc_cd4_pivot["DC_to_CD4_Ratio"] = (
        df_dc_cd4_pivot["Dendritic Cells"] / df_dc_cd4_pivot["CD4 T cells"]
    )
    # Combine recovered and decline into one condition for better visualization
    df_dc_cd4_pivot[condition_id] = (
        df_dc_cd4_pivot[condition_id]
        .astype(str)
        .replace({"recovered": "ALAD", "declined": "ALAD"})
    )
    sns.boxplot(
        data=df_dc_cd4_pivot,
        x=condition_id,
        y="DC_to_CD4_Ratio",
        ax=ax[1],
        hue=condition_id,
    )

    # Statistical test between Control and ALAD
    from scipy.stats import mannwhitneyu

    control_data = df_dc_cd4_pivot[df_dc_cd4_pivot[condition_id] == "control"][
        "DC_to_CD4_Ratio"
    ]
    alad_data = df_dc_cd4_pivot[df_dc_cd4_pivot[condition_id] == "ALAD"][
        "DC_to_CD4_Ratio"
    ]
    stat, p_value = mannwhitneyu(control_data, alad_data)
    print(f"Mann-Whitney U test between Control and ALAD: U={stat}, p-value={p_value}")
    # ax[1].

    # Plot scatter plot of DC proportion vs CD4 T cell proportion where each point is a sample
    df_scatter = df[df[celltype_id].isin(["Dendritic Cells", "CD4 T cells"])]
    df_scatter_pivot = df_scatter.pivot_table(
        index=[sample_id, condition_id],
        columns=celltype_id,
        values="proportion",
        fill_value=0,
        observed=False,
    ).reset_index()

    # Combine recovered and decline into one condition for better visualization
    df_scatter_pivot[condition_id] = (
        df_scatter_pivot[condition_id]
        .astype(str)
        .replace({"recovered": "ALAD", "declined": "ALAD"})
    )
    # Create scatter plot
    sns.scatterplot(
        data=df_scatter_pivot,
        x="CD4 T cells",
        y="Dendritic Cells",
        hue=condition_id,
        ax=ax[2],
    )
    ax[2].set_xlabel("CD4 T cell proportion")
    ax[2].set_ylabel("DC proportion")
    ax[2].set_yscale("log")
    ax[2].set_xscale("log")

    # Calculate and display correlation
    correlation, p_value = spearmanr(
        df_scatter_pivot["CD4 T cells"], df_scatter_pivot["Dendritic Cells"]
    )
    ax[2].set_title(
        f"DC vs CD4 T cell proportions: ρ={correlation:.2f}, p={p_value:.2g}"
    )

    return f
