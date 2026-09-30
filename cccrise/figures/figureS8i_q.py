"""
Figure S8i-q: Box plots of condition and ALAD status comparisons for specific components in cccRISE on BALF alad data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotGeneral import (
    annotate_boxplot_n,
    format_p_value,
    get_alad_patient_factors_df,
    run_group_anova,
    run_pairwise_ttest,
)

CONDITION_COLUMNS = {
    "6monthcondition": "6-Month Condition",
    "1yearcondition": "1-Year Condition",
}


def _plot_condition_boxplot(
    ax, comp_data: pd.DataFrame, comp_num: int, group_col: str, label: str
) -> None:
    """Boxplot factor_value by an outcome condition, annotated with ANOVA results."""
    plot_data = comp_data.dropna(subset=[group_col])

    if plot_data.empty:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"Component {comp_num} - {label}")
        return

    sns.boxplot(data=plot_data, x=group_col, y="factor_value", ax=ax)

    title = f"Component {comp_num}\n{label}"
    anova_result = run_group_anova(plot_data, group_col)
    if anova_result is not None:
        f_stat = anova_result["f_stat"]
        p_str = format_p_value(anova_result["p_value"])
        title += f"\nF = {f_stat:.2f}, {p_str}"

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(label)
    ax.set_ylabel("Factor Value")
    annotate_boxplot_n(ax, len(plot_data))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _plot_alad_status_boxplot(ax, comp_data: pd.DataFrame, comp_num: int) -> None:
    """Boxplot factor_value by ALAD status, annotated with a t-test result."""
    sns.boxplot(data=comp_data, x="alad_status", y="factor_value", ax=ax)

    title = f"Component {comp_num}\nALAD Status"
    ttest_result = run_pairwise_ttest(comp_data, "alad_status")
    if ttest_result is not None:
        t_stat = ttest_result["t_stat"]
        p_str = format_p_value(ttest_result["p_value"])
        title += f"\nt = {t_stat:.2f}, {p_str}"

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("ALAD Status")
    ax.set_ylabel("Factor Value")
    annotate_boxplot_n(ax, len(comp_data))


def _log_anova_result(comp_num: int, label: str, result: dict) -> None:
    print(f"Component {comp_num} - {label}:")
    print(f"  F = {result['f_stat']:.3f}, {format_p_value(result['p_value'])}")
    print(f"  Groups: {result['groups']}")
    print(f"  Group sizes: {result['group_sizes']}")


def makeFigure():
    ax, f = getSetup((6, 6), (3, 3))  # 3 rows x 3 columns for 9 plots total
    subplotLabel(ax)

    X = get_alad_data(project_cells=True)
    target_components = [5, 13, 17]  # Specific components of interest
    _, factor_df = get_alad_patient_factors_df(X, target_components=target_components)

    print("ANOVA Results:")
    print("=" * 50)
    for comp_num in target_components:
        comp_data = factor_df[factor_df["component"] == f"Component_{comp_num}"]
        for group_col, label in CONDITION_COLUMNS.items():
            result = run_group_anova(comp_data.dropna(subset=[group_col]), group_col)
            if result is not None:
                _log_anova_result(comp_num, label, result)
        print()

    print("T-Test Results for ALAD Status:")
    print("=" * 40)
    for comp_num in target_components:
        comp_data = factor_df[factor_df["component"] == f"Component_{comp_num}"]
        result = run_pairwise_ttest(comp_data, "alad_status")
        if result is not None:
            print(f"Component {comp_num} - ALAD Status:")
            print(f"  t = {result['t_stat']:.3f}, {format_p_value(result['p_value'])}")
            print(
                f"  Groups: {result['group1']} (n={result['n1']}) vs "
                f"{result['group2']} (n={result['n2']})"
            )
        print()

    # Row 1: 6-month condition (positions 0, 1, 2)
    # Row 2: 1-year condition (positions 3, 4, 5)
    # Row 3: ALAD status (positions 6, 7, 8)
    for i, comp_num in enumerate(target_components):
        comp_data = factor_df[factor_df["component"] == f"Component_{comp_num}"]

        _plot_condition_boxplot(
            ax[i], comp_data, comp_num, "6monthcondition", "6-Month Condition"
        )
        _plot_condition_boxplot(
            ax[i + 3], comp_data, comp_num, "1yearcondition", "1-Year Condition"
        )
        _plot_alad_status_boxplot(ax[i + 6], comp_data, comp_num)

    print("Summary of condition categories:")
    for comp_num in target_components:
        comp_data = factor_df[factor_df["component"] == f"Component_{comp_num}"]
        print(f"\nComponent {comp_num}:")
        for group_col in CONDITION_COLUMNS:
            data = comp_data.dropna(subset=[group_col])
            if not data.empty:
                print(
                    f"  {group_col} counts: {data[group_col].value_counts().to_dict()}"
                )

    return f
