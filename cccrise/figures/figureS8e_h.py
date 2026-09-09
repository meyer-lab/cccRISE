"""
Figure S3: cccRISE on BALF alad data. Showing weighted sender and receiver cell factors.
"""

import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind

from .common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotGeneral import (
    DEFAULT_PATIENT_INFO,
    get_alad_patient_factors_df,
)


def makeFigure():
    ax, f = getSetup((10, 4), (1, 2))
    subplotLabel(ax)

    X = get_alad_data(project_cells=True)
    target_components = [5, 13, 17]

    patient_continuous = [
        "age",
        "timeaftertx",
        "baselineFEV1",
        "FEV16monthsb4",
        "FEV16monthsb4p",
        "FEV1alad",
        "FEV1aladp",
        "FEV16monthsafter",
        "FEV16monthsafterp",
        "FEV11yearafter",
        "FEV1pyearafter",
        "d2b_index",
    ]

    _, factor_df = get_alad_patient_factors_df(X, target_components=target_components)

    # Pearson correlations for continuous variables
    correlation_results = []
    for comp_num in target_components:
        comp_name = f"Component_{comp_num}"
        comp_data = factor_df[factor_df["component"] == comp_name]
        for cont_var in patient_continuous:
            if cont_var not in comp_data.columns:
                continue
            numeric_var = pd.to_numeric(comp_data[cont_var], errors="coerce")
            valid_mask = comp_data["factor_value"].notna() & numeric_var.notna()
            valid_comp = comp_data.loc[valid_mask, "factor_value"]
            valid_var = numeric_var.loc[valid_mask]
            if len(valid_comp) > 3:
                try:
                    _, p_value = pearsonr(valid_comp, valid_var)
                    correlation_results.append(
                        {
                            "component": comp_name,
                            "variable": cont_var,
                            "p_value": p_value,
                        }
                    )
                except Exception as e:
                    print(f"Pearson failed for {comp_name}, {cont_var}: {e}")
    pearson_df = pd.DataFrame(correlation_results)

    # T-test p-values for categorical variables
    ttest_results = []
    for comp_num in target_components:
        comp_name = f"Component_{comp_num}"
        comp_data = factor_df[factor_df["component"] == comp_name]
        for cat_var in DEFAULT_PATIENT_INFO:
            if cat_var not in comp_data.columns:
                continue
            valid_mask = comp_data["factor_value"].notna() & comp_data[cat_var].notna()
            valid_comp = comp_data.loc[valid_mask, "factor_value"]
            valid_cat = comp_data.loc[valid_mask, cat_var]
            if valid_cat.nunique() == 2 and len(valid_comp) > 3:
                try:
                    groups = [valid_comp[valid_cat == g] for g in valid_cat.unique()]
                    if all(len(g) > 1 for g in groups):
                        _, p_value = ttest_ind(groups[0], groups[1], nan_policy="omit")
                        ttest_results.append(
                            {
                                "component": comp_name,
                                "variable": cat_var,
                                "p_value": p_value,
                            }
                        )
                except Exception as e:
                    print(f"T-test failed for {comp_name}, {cat_var}: {e}")
    ttest_df = pd.DataFrame(ttest_results)
    # Plot heatmaps
    if not pearson_df.empty:
        pearson_pval_matrix = pearson_df.pivot(
            index="component", columns="variable", values="p_value"
        )
        sns.heatmap(
            pearson_pval_matrix,
            ax=ax[0],
            cmap="viridis_r",
            cbar_kws={"label": "Pearson p-value"},
            vmin=0,
            vmax=1,
        )
        ax[0].set_xlabel("Continuous Variable")
        ax[0].set_ylabel("Component")
        ax[0].tick_params(axis="x", rotation=90)
    else:
        ax[0].text(
            0.5,
            0.5,
            "No Pearson p-values",
            ha="center",
            va="center",
            transform=ax[0].transAxes,
        )
        ax[0].set_title("Pearson p-value Heatmap (Continuous)")
    if not ttest_df.empty:
        ttest_pval_matrix = ttest_df.pivot(
            index="component", columns="variable", values="p_value"
        )
        sns.heatmap(
            ttest_pval_matrix,
            ax=ax[1],
            cmap="mako_r",
            cbar_kws={"label": "T-test p-value"},
            vmin=0,
            vmax=1,
        )
        ax[1].set_xlabel("Categorical Variable")
        ax[1].set_ylabel("Component")

        ax[1].tick_params(axis="x", rotation=45)
    else:
        ax[1].text(
            0.5,
            0.5,
            "No T-test p-values",
            ha="center",
            va="center",
            transform=ax[1].transAxes,
        )
        ax[1].set_title("T-test p-value Heatmap (Categorical)")
    # Hide unused axes
    for i in range(2, len(ax)):
        ax[i].axis("off")
    return f
