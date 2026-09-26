"""
Figure A5h: Boxplots of associations between cccRISE components and patient clinical variables in BALF ALAD data.
"""

from itertools import combinations
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotGeneral import (
    DEFAULT_PATIENT_INFO,
    format_p_value,
    get_alad_patient_factors_df,
    run_group_anova,
    run_pairwise_ttest,
)

OUTCOME_VARS = ["6monthcondition", "1yearcondition"]


def _collect_ttest_results(
    factor_df: pd.DataFrame, target_indices: list[int], patient_info: list[str]
) -> pd.DataFrame:
    """Pairwise t-tests between every pair of categories, per component and variable."""
    results = []
    for comp_idx in target_indices:
        comp_data = factor_df[factor_df["component_idx"] == comp_idx]

        for var in patient_info:
            var_categories = comp_data[var].dropna().unique()
            if len(var_categories) < 2:
                continue

            for cat1, cat2 in combinations(var_categories, 2):
                pair_data = comp_data[comp_data[var].isin([cat1, cat2])]
                result = run_pairwise_ttest(pair_data, var, min_group_size=2)
                if result is not None:
                    results.append(
                        {
                            "component": f"Component_{comp_idx + 1}",
                            "variable": var,
                            "category1": result["group1"],
                            "category2": result["group2"],
                            "t_statistic": result["t_stat"],
                            "p_value": result["p_value"],
                            "n_group1": result["n1"],
                            "n_group2": result["n2"],
                        }
                    )

    return pd.DataFrame(results)


def _collect_anova_results(
    factor_df: pd.DataFrame, target_indices: list[int]
) -> pd.DataFrame:
    """Three-group ANOVAs for the 6-month/1-year outcome variables, per component."""
    results = []
    for comp_idx in target_indices:
        comp_data = factor_df[factor_df["component_idx"] == comp_idx]

        for var in OUTCOME_VARS:
            var_data = comp_data.dropna(subset=[var])
            if var_data[var].nunique() != 3:  # only proceed with all 3 categories
                continue

            result = run_group_anova(var_data, var, min_group_size=2)
            if result is not None and len(result["groups"]) == 3:
                results.append(
                    {
                        "component": f"Component_{comp_idx + 1}",
                        "variable": var,
                        "f_statistic": result["f_stat"],
                        "p_value": result["p_value"],
                        "groups": result["groups"],
                        "group_sizes": result["group_sizes"],
                        "total_n": result["total_n"],
                    }
                )
                print(
                    f"ANOVA - Component_{comp_idx + 1}, {var}: "
                    f"F={result['f_stat']:.3f}, p={result['p_value']:.4f}, "
                    f"groups={result['groups']}, n={result['group_sizes']}"
                )

    return pd.DataFrame(results)


def _plot_ttest_result(ax, factor_df: pd.DataFrame, row: pd.Series) -> None:
    comp_idx = int(row["component"].split("_")[1]) - 1
    var = row["variable"]
    cat1, cat2 = row["category1"], row["category2"]

    plot_data = factor_df[
        (factor_df["component_idx"] == comp_idx) & (factor_df[var].isin([cat1, cat2]))
    ]
    sns.boxplot(data=plot_data, x=var, y="factor_value", ax=ax)

    n_total = len(plot_data)
    p_str = format_p_value(row["p_value"])
    title = (
        f"{row['component']}\n{var}: {cat1} vs {cat2}\n"
        f"{p_str} (n={row['n_group1']}+{row['n_group2']}={n_total})"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(var)
    ax.set_ylabel("Factor Value")


def _plot_anova_result(ax, factor_df: pd.DataFrame, row: pd.Series) -> None:
    comp_idx = int(row["component"].split("_")[1]) - 1
    var = row["variable"]

    plot_data = factor_df[
        (factor_df["component_idx"] == comp_idx) & (factor_df[var].notna())
    ]
    sns.boxplot(data=plot_data, x=var, y="factor_value", ax=ax)

    p_str = format_p_value(row["p_value"])
    title = (
        f"{row['component']}\n{var} (ANOVA)\n"
        f"F={row['f_statistic']:.2f}, {p_str} (n={row['total_n']})"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(var)
    ax.set_ylabel("Factor Value")


def _log_significant_anova(anova_df: pd.DataFrame) -> None:
    significant_anova = anova_df[anova_df["p_value"] < 0.05]
    print(f"\nFound {len(significant_anova)} significant ANOVA results (p < 0.05):")
    for _, row in significant_anova.iterrows():
        print(
            f"{row['component']} - {row['variable']}: "
            f"F={row['f_statistic']:.3f}, p={row['p_value']:.4f}, n={row['total_n']}"
        )


def _queue_significant_ttests(results_df: pd.DataFrame) -> list[dict]:
    if results_df.empty:
        return []

    significant_results = results_df[results_df["p_value"] < 0.05].copy()
    significant_results = significant_results.sort_values("p_value")

    print(f"Found {len(significant_results)} significant t-test results (p < 0.05):")
    print(
        significant_results[
            ["component", "variable", "category1", "category2", "p_value"]
        ]
    )

    return [
        {"type": "t-test", "data": row} for _, row in significant_results.iterrows()
    ]


def _queue_significant_anovas(anova_df: pd.DataFrame) -> list[dict]:
    if anova_df.empty:
        return []

    significant_anova = anova_df[anova_df["p_value"] < 0.05]
    if len(significant_anova) == 0:
        print("No significant ANOVA results, showing top 3 with lowest p-values")
        significant_anova = anova_df.nsmallest(3, "p_value")

    return [{"type": "anova", "data": row} for _, row in significant_anova.iterrows()]


def _render_no_results_placeholder(ax) -> None:
    print("No significant results found (p < 0.05)")
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

    for i in range(min(4, len(ax)), len(ax)):
        ax[i].set_visible(False)


def makeFigure():
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)

    X = get_alad_data(project_cells=True)
    target_components = [4, 12, 16]  # 1-indexed component numbers
    target_indices = [comp - 1 for comp in target_components]
    patient_info = DEFAULT_PATIENT_INFO

    _patient_info_df, factor_df = get_alad_patient_factors_df(
        X, target_components=target_components
    )

    results_df = _collect_ttest_results(factor_df, target_indices, patient_info)
    print(results_df)

    anova_df = _collect_anova_results(factor_df, target_indices)
    if not anova_df.empty:
        _log_significant_anova(anova_df)

    plot_queue = _queue_significant_ttests(results_df) + _queue_significant_anovas(
        anova_df
    )

    if not plot_queue:
        _render_no_results_placeholder(ax)
        return f

    n_plots = min(len(plot_queue), 9)  # Maximum 9 plots for 3x3 grid
    for i in range(n_plots):
        result = plot_queue[i]
        row = cast(pd.Series, result["data"])

        if result["type"] == "t-test":
            _plot_ttest_result(ax[i], factor_df, row)
        else:
            _plot_anova_result(ax[i], factor_df, row)

        plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right")

    for i in range(n_plots, len(ax)):
        ax[i].set_visible(False)

    return f
