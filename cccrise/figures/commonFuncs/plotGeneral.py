from typing import cast

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import f_oneway, ttest_ind


def get_sample_to_group(
    X: anndata.AnnData, condition_column: str, group_col: str
) -> pd.Series:
    """Map each unique sample to its group label, ordered to match
    np.unique(X.obs[condition_column])."""
    sample_to_group = (
        cast(pd.DataFrame, X.obs)
        .drop_duplicates(subset=[condition_column, group_col])
        .set_index(condition_column)[group_col]
    )
    return sample_to_group.loc[np.unique(X.obs[condition_column], return_index=True)[0]]


def binarize_group(
    sample_to_group: pd.Series, match_value: str, match_name: str, other_name: str
) -> pd.Series:
    """Collapse a group Series into two named categories and return integer codes."""
    binary = sample_to_group.apply(
        lambda x: match_name if x == match_value else other_name
    )
    return binary.astype("category").cat.codes


def symmetrize_axes(ax: Axes):
    """Set both x and y limits to +/- the largest absolute limit on either axis."""
    lim = max(np.max(np.abs(ax.get_xlim())), np.max(np.abs(ax.get_ylim())))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)


DEFAULT_PATIENT_INFO = [
    "diagnosisgroup",
    "sex",
    "transplanttype",
    "cmvstatus",
    "cmv_status",
    "cmvstatus2",
    "ethnicity",
    "6monthcondition",
    "1yearcondition",
]

DEFAULT_PATIENT_CONTINUOUS = [
    "percent.mt",
    "percent.ribo",
    "isotype_ctl_max",
    "doubletFinderScore",
    "S.Score",
    "G2M.Score",
    "age",
    "timeaftertx",
    "Cells (M)",
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


def plot_fms_r2x_stability(
    df: pd.DataFrame,
    ax: list[Axes] | tuple[Axes, Axes] | np.ndarray,
    title_prefix: str = "",
    x_col: str = "Component",
    fms_col: str = "FMS",
    r2x_col: str = "R2X",
    fms_label: str = "FMS",
    r2x_label: str = "R2X",
    invert_xaxis: bool = False,
    x_label: str | None = None,
    y_label_fms: str | None = None,
    y_label_r2x: str | None = None,
):
    """Plots standard 2-panel FMS and R2X stability curves."""
    sns.lineplot(data=df, x=x_col, y=fms_col, ax=ax[0], label=fms_label)
    ax[0].set_ylim(0, 1)
    if title_prefix:
        ax[0].set_title(f"{title_prefix}: {fms_label}".strip())
    if x_label:
        ax[0].set_xlabel(x_label)
    if y_label_fms:
        ax[0].set_ylabel(y_label_fms)
    if invert_xaxis:
        ax[0].invert_xaxis()

    sns.lineplot(data=df, x=x_col, y=r2x_col, ax=ax[1], color="orange", label=r2x_label)
    ax[1].set_ylim(0, float(np.max(df[r2x_col])) + 0.02)
    if title_prefix:
        ax[1].set_title(f"{title_prefix}: {r2x_label}".strip())
    if x_label:
        ax[1].set_xlabel(x_label)
    if y_label_r2x:
        ax[1].set_ylabel(y_label_r2x)
    if invert_xaxis:
        ax[1].invert_xaxis()


def compute_cell_type_proportions(
    X: anndata.AnnData,
    celltype_col: str = "celltype",
    sample_col: str = "sample",
    condition_col: str | None = "condition",
) -> pd.DataFrame:
    """Compute cell type counts and proportions per sample, merging condition metadata if present."""
    obs = cast(pd.DataFrame, X.obs)
    counts = cast(
        pd.Series, obs.groupby([celltype_col, sample_col], observed=False).size()
    )
    df = counts.reset_index(name="count")

    if condition_col is not None and condition_col in obs.columns:
        sample_condition_map = obs[[sample_col, condition_col]].drop_duplicates()
        df = df.merge(sample_condition_map, on=sample_col, how="left")
        df[condition_col] = pd.Categorical(
            df[condition_col], categories=np.unique(df[condition_col]), ordered=True
        )

    df[celltype_col] = pd.Categorical(
        df[celltype_col], categories=np.unique(df[celltype_col]), ordered=True
    )
    df[sample_col] = pd.Categorical(
        df[sample_col], categories=np.unique(df[sample_col]), ordered=True
    )

    sort_cols = [
        c for c in [condition_col, sample_col, celltype_col] if c and c in df.columns
    ]
    df = df.sort_values(sort_cols)
    df["proportion"] = df.groupby(sample_col, observed=False)["count"].transform(
        lambda x: x / x.sum()
    )
    return df


def categorize_outcome(condition) -> str | None:
    """Categorize 6-month and 1-year conditions into standard groups."""
    condition_str = str(condition)
    if condition_str == "Died":
        return "Died"
    if condition_str in ["CLAD", "CLAD/Stable", "CLAD/Declined"]:
        return "CLAD"
    if condition_str in [
        "No CLAD/Stable",
        "No CLAD/Recovered",
        "No CLAD/Declined",
        "Recovered",
    ]:
        return "No CLAD"
    return None


def prepare_alad_clinical_metadata(X: anndata.AnnData) -> pd.DataFrame:
    """Preprocess ALAD clinical metadata on AnnData obs and return unique patient DataFrame."""
    X.obs["alad_status"] = (
        X.obs["ALADstatus"]
        .astype(str)
        .replace({"recovered": "ALAD", "declined": "ALAD"})
    )
    if "d2b_index" in X.obs.columns:
        X.obs["d2b_index"] = pd.to_numeric(X.obs["d2b_index"], errors="coerce")

    all_cols = ["dsco_id", "alad_status"] + [
        c
        for c in DEFAULT_PATIENT_INFO + DEFAULT_PATIENT_CONTINUOUS
        if c in X.obs.columns
    ]
    obs = cast(pd.DataFrame, X.obs)
    patient_df = (
        obs[all_cols].drop_duplicates(subset=["dsco_id"]).reset_index(drop=True)
    )

    for col in ["6monthcondition", "1yearcondition"]:
        if col in patient_df.columns:
            patient_df[col] = patient_df[col].apply(categorize_outcome)

    return patient_df


def get_alad_patient_factors_df(
    X: anndata.AnnData,
    target_components: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return patient metadata dataframe and tidy factors-metadata dataframe."""
    patient_info_df = prepare_alad_clinical_metadata(X)
    factor_A = X.uns["A"]
    if target_components is None:
        target_indices = list(range(factor_A.shape[1]))
    else:
        target_indices = [comp - 1 for comp in target_components]

    sample_to_idx = {
        sample: idx for idx, sample in enumerate(X.obs["dsco_id"].cat.categories)
    }

    factor_df_list = []
    for _, sample_row in patient_info_df.iterrows():
        sample = sample_row["dsco_id"]
        if sample in sample_to_idx:
            sample_idx = sample_to_idx[sample]
            for comp_idx in target_indices:
                row_data = {
                    "sample": sample,
                    "alad_status": sample_row["alad_status"],
                    "component": f"Component_{comp_idx + 1}",
                    "factor_value": factor_A[sample_idx, comp_idx],
                    "component_idx": comp_idx,
                }
                for col in patient_info_df.columns:
                    if col not in ["dsco_id", "alad_status"]:
                        row_data[col] = sample_row[col]
                factor_df_list.append(row_data)

    factor_df = pd.DataFrame(factor_df_list)
    return patient_info_df, factor_df


def format_p_value(p_value: float) -> str:
    """Format a p-value for display, flagging p < 0.05 with a trailing asterisk."""
    if p_value < 0.001:
        return "p < 0.001"
    if p_value < 0.01:
        return "p < 0.01"
    if p_value < 0.05:
        return f"p = {p_value:.3f}*"
    return f"p = {p_value:.3f}"


def run_group_anova(
    data: pd.DataFrame,
    group_col: str,
    value_col: str = "factor_value",
    min_group_size: int = 2,
) -> dict | None:
    """Run a one-way ANOVA of value_col across the categories of group_col.

    Returns None if fewer than two categories have at least min_group_size
    non-null observations.
    """
    subset = data.dropna(subset=[group_col])
    if subset.empty:
        return None

    groups, group_names, group_sizes = [], [], []
    for cat in subset[group_col].unique():
        group_data = subset.loc[subset[group_col] == cat, value_col].dropna()
        if len(group_data) >= min_group_size:
            groups.append(group_data)
            group_names.append(cat)
            group_sizes.append(len(group_data))

    if len(groups) < 2:
        return None

    f_stat, p_value = f_oneway(*groups)
    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "groups": group_names,
        "group_sizes": group_sizes,
        "total_n": sum(group_sizes),
    }


def run_pairwise_ttest(
    data: pd.DataFrame,
    group_col: str,
    value_col: str = "factor_value",
    min_group_size: int = 2,
) -> dict | None:
    """Run a two-sample t-test between exactly two categories of group_col.

    Returns None unless group_col has exactly two categories with at least
    min_group_size non-null observations each.
    """
    categories = data[group_col].dropna().unique()
    if len(categories) != 2:
        return None

    group1_data = data.loc[data[group_col] == categories[0], value_col].dropna()
    group2_data = data.loc[data[group_col] == categories[1], value_col].dropna()
    if len(group1_data) < min_group_size or len(group2_data) < min_group_size:
        return None

    t_stat, p_value = ttest_ind(group1_data, group2_data)
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "group1": categories[0],
        "group2": categories[1],
        "n1": len(group1_data),
        "n2": len(group2_data),
    }


def annotate_boxplot_n(ax: Axes, n: int, fontsize: int = 8) -> None:
    """Add a sample-size annotation to the top-left corner of a boxplot axis."""
    ax.text(
        0.02,
        0.98,
        f"n = {n}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=fontsize,
    )
