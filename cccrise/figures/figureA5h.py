"""
Figure A5h: Boxplots of associations between cccRISE components and patient clinical variables in BALF ALAD data.
"""

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind

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
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)

    X = get_alad_data(project_cells=True)
    target_components = [4, 12, 16]  # 1-indexed component numbers
    target_indices = [comp - 1 for comp in target_components]  # Convert to 0-indexed
    patient_info = DEFAULT_PATIENT_INFO

    _patient_info_df, factor_df = get_alad_patient_factors_df(
        X, target_components=target_components
    )

    # Perform t-tests for specific components only (12, 16, and 4)
    results = []

    for comp_idx in target_indices:
        comp_data = factor_df[factor_df["component_idx"] == comp_idx]

        for var in patient_info:
            # Get all categories for this variable
            var_categories = comp_data[var].dropna().unique()
            if len(var_categories) < 2:
                continue

            # Perform pairwise t-tests between all categories
            from itertools import combinations

            for cat1, cat2 in combinations(var_categories, 2):
                # Get factor values for each category
                group1_data = comp_data[comp_data[var] == cat1]["factor_value"].dropna()
                group2_data = comp_data[comp_data[var] == cat2]["factor_value"].dropna()

                # Perform t-test if both groups have sufficient data
                if len(group1_data) > 1 and len(group2_data) > 1:
                    try:
                        t_stat, p_value = ttest_ind(group1_data, group2_data)
                        results.append(
                            {
                                "component": f"Component_{comp_idx + 1}",
                                "variable": var,
                                "category1": cat1,
                                "category2": cat2,
                                "t_statistic": t_stat,
                                "p_value": p_value,
                                "n_group1": len(group1_data),
                                "n_group2": len(group2_data),
                                "group1_name": cat1,
                                "group2_name": cat2,
                            }
                        )
                    except Exception as e:
                        print(
                            f"T-test failed for Component {comp_idx + 1}, {var}, {cat1} vs {cat2}: {e}"
                        )

    # Create results dataframe and filter for significant results (p < 0.05)
    results_df = pd.DataFrame(results)
    print(results_df)

    # Perform ANOVA for 3-category outcome variables (6monthcondition and 1yearcondition)
    anova_results = []
    outcome_vars = ["6monthcondition", "1yearcondition"]

    for comp_idx in target_indices:
        comp_data = factor_df[factor_df["component_idx"] == comp_idx]

        for var in outcome_vars:
            # Get data for each of the 3 categories
            var_data = comp_data.dropna(subset=[var])
            categories = var_data[var].unique()

            if len(categories) == 3:  # Only proceed if we have all 3 categories
                groups = []
                group_names = []
                group_sizes = []

                for cat in categories:
                    group_data = var_data[var_data[var] == cat]["factor_value"].dropna()
                    if len(group_data) >= 2:  # Need at least 2 samples per group
                        groups.append(group_data)
                        group_names.append(cat)
                        group_sizes.append(len(group_data))

                if (
                    len(groups) == 3
                ):  # Proceed with ANOVA if all 3 groups have sufficient data
                    try:
                        f_stat, p_value = f_oneway(*groups)
                        anova_results.append(
                            {
                                "component": f"Component_{comp_idx + 1}",
                                "variable": var,
                                "f_statistic": f_stat,
                                "p_value": p_value,
                                "groups": group_names,
                                "group_sizes": group_sizes,
                                "total_n": sum(group_sizes),
                            }
                        )
                        print(
                            f"ANOVA - Component {comp_idx + 1}, {var}: F={f_stat:.3f}, p={p_value:.4f}, groups={group_names}, n={group_sizes}"
                        )
                    except Exception as e:
                        print(f"ANOVA failed for Component {comp_idx + 1}, {var}: {e}")

    # Print ANOVA results
    anova_df = pd.DataFrame(anova_results)
    if not anova_df.empty:
        significant_anova = anova_df[anova_df["p_value"] < 0.05]
        print(f"\nFound {len(significant_anova)} significant ANOVA results (p < 0.05):")
        for _, row in significant_anova.iterrows():
            print(
                f"{row['component']} - {row['variable']}: F={row['f_statistic']:.3f}, p={row['p_value']:.4f}, n={row['total_n']}"
            )

    # Combine t-test and ANOVA results for plotting
    all_significant_results = []

    # Add significant t-test results
    if not results_df.empty:
        significant_results = results_df[results_df["p_value"] < 0.05].copy()
        significant_results = significant_results.sort_values("p_value")

        print(
            f"Found {len(significant_results)} significant t-test results (p < 0.05):"
        )
        print(
            significant_results[
                ["component", "variable", "category1", "category2", "p_value"]
            ]
        )

        # Add t-test results to plotting list
        for _, row in significant_results.iterrows():
            all_significant_results.append(
                {"type": "t-test", "data": row, "plot_type": "boxplot_pairwise"}
            )

    # Add significant ANOVA results (or top results if none are significant)
    if not anova_df.empty:
        significant_anova = anova_df[anova_df["p_value"] < 0.05]
        if len(significant_anova) == 0:
            # If no significant ANOVA results, take the top 3 with lowest p-values
            print("No significant ANOVA results, showing top 3 with lowest p-values")
            significant_anova = anova_df.nsmallest(3, "p_value")

        for _, row in significant_anova.iterrows():
            all_significant_results.append(
                {"type": "anova", "data": row, "plot_type": "boxplot_all_groups"}
            )

    if all_significant_results:
        # Plot up to the number of significant results found, or the number of available axes
        n_plots = min(len(all_significant_results), 9)  # Maximum 9 plots for 3x3 grid

        for i in range(n_plots):
            result = all_significant_results[i]

            if result["type"] == "t-test":
                # Handle t-test plotting (existing logic)
                row = cast(pd.Series, result["data"])

                # Get data for this specific test
                comp_idx = int(row["component"].split("_")[1]) - 1
                var = row["variable"]
                cat1 = row["category1"]
                cat2 = row["category2"]

                # Filter data for the specific component and include both categories
                plot_data = factor_df[
                    (factor_df["component_idx"] == comp_idx)
                    & (factor_df[var].isin([cat1, cat2]))
                ]

                # Create boxplot comparing the two categories
                sns.boxplot(data=plot_data, x=var, y="factor_value", ax=ax[i])

                # Calculate sample sizes for each category
                n_cat1 = len(plot_data[plot_data[var] == cat1])
                n_cat2 = len(plot_data[plot_data[var] == cat2])
                n_total = len(plot_data)

                # Format title with p-value and sample size
                p_val = row["p_value"]
                if p_val < 0.001:
                    p_str = "p < 0.001"
                elif p_val < 0.01:
                    p_str = "p < 0.01"
                else:
                    p_str = f"p = {p_val:.3f}"

                title = f"{row['component']}\n{var}: {cat1} vs {cat2}\n{p_str} (n={n_cat1}+{n_cat2}={n_total})"
                ax[i].set_title(title, fontsize=10)
                ax[i].set_xlabel(var)
                ax[i].set_ylabel("Factor Value")

            elif result["type"] == "anova":
                # Handle ANOVA plotting (all 3 groups)
                row = cast(pd.Series, result["data"])

                # Get data for this ANOVA test
                comp_idx = int(row["component"].split("_")[1]) - 1
                var = row["variable"]

                # Filter data for the specific component
                plot_data = factor_df[
                    (factor_df["component_idx"] == comp_idx) & (factor_df[var].notna())
                ]

                # Create boxplot for all categories
                sns.boxplot(data=plot_data, x=var, y="factor_value", ax=ax[i])

                # Format title with F-statistic and p-value
                p_val = row["p_value"]
                f_stat = row["f_statistic"]
                if p_val < 0.001:
                    p_str = "p < 0.001"
                elif p_val < 0.01:
                    p_str = "p < 0.01"
                else:
                    p_str = f"p = {p_val:.3f}"

                title = f"{row['component']}\n{var} (ANOVA)\nF={f_stat:.2f}, {p_str} (n={row['total_n']})"
                ax[i].set_title(title, fontsize=10)
                ax[i].set_xlabel(var)
                ax[i].set_ylabel("Factor Value")

            # Rotate x-axis labels if needed
            plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right")

        # Hide unused axes
        for i in range(n_plots, len(ax)):
            ax[i].set_visible(False)

    else:
        print("No significant results found (p < 0.05)")
        # Add empty plots with informative titles for first few axes
        for i in range(min(4, len(ax))):
            ax[i].text(
                0.5,
                0.5,
                "No significant\nresults found",
                ha="center",
                va="center",
                transform=ax[i].transAxes,
            )
            ax[i].set_title("Component Analysis")

        # Hide remaining axes
        for i in range(min(4, len(ax)), len(ax)):
            ax[i].set_visible(False)

    return f
