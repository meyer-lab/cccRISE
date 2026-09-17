import math
from typing import cast

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..common import getSetup


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


def collapse_group(
    sample_to_group: pd.Series, match_value: str, match_name: str, other_name: str
) -> pd.Series:
    """Collapse a group Series into two named categories, keeping the names.

    The names, rather than codes, are what a test that reports per-group
    statistics needs -- an ANOVA has to be able to say which group loaded
    higher. `binarize_group` is this with the names traded for codes.
    """
    return sample_to_group.map(lambda x: match_name if x == match_value else other_name)


def binarize_group(
    sample_to_group: pd.Series, match_value: str, match_name: str, other_name: str
) -> pd.Series:
    """Collapse a group Series into two named categories and return integer codes."""
    collapsed = collapse_group(sample_to_group, match_value, match_name, other_name)
    return collapsed.astype("category").cat.codes


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


FACTOR_KEYS = {"sender": "sc_B", "receiver": "rc_C"}


def plot_wc_histogram(
    X: anndata.AnnData,
    cmp: int,
    ax: Axes,
    celltype: str,
    celltype_col: str = "celltype",
    factor_types: tuple[str, ...] = ("sender", "receiver"),
    bins: int = 50,
    top_perc: float = 10,
    log_y: bool = True,
) -> None:
    """Histogram of one cell type's weights for one component, sender and receiver
    overlaid.

    Complements the cell type x component summaries
    (`celltype_component_association`): a mean says nothing about the shape it came
    from, and a bimodal population whose halves cancel to a mean of zero is
    indistinguishable there from one that is genuinely unassociated. Bin edges,
    axis limits, and the top-percentile threshold are all derived from every cell in
    `X` rather than from the plotted subset, so panels drawn for different cell
    types are directly comparable.

    Parameters
    ----------
    X : anndata.AnnData
        AnnData carrying per-cell loadings in `obsm["sc_B"]` / `obsm["rc_C"]` and
        cell type labels in `.obs`.
    cmp : int
        Component to plot (1-indexed).
    celltype : str
        The cell type whose cells are histogrammed.
    factor_types : tuple of {"sender", "receiver"}
        Which loadings to overlay.
    top_perc : float, default=10
        A dashed line marks this top percentile of all cells, and the legend reports
        the share of this cell type falling above it (the `"pct_top"` statistic of
        `celltype_component_association`).
    log_y : bool, default=True
        Log-scale the y axis, which keeps a small high-weight tail visible against a
        peak near zero.
    """
    labels = np.asarray(cast(pd.DataFrame, X.obs)[celltype_col]).astype(str)
    mask = labels == celltype

    all_values = [
        np.asarray(X.obsm[FACTOR_KEYS[factor_type]], dtype=float)[:, cmp - 1]
        for factor_type in factor_types
    ]
    lo = min(float(values.min()) for values in all_values)
    hi = max(float(values.max()) for values in all_values)
    # A list rather than an array: matplotlib's stubs type `bins` as Sequence[float]
    edges = np.linspace(lo, hi, bins + 1).tolist()
    colors = sns.color_palette("colorblind")

    ax.axvline(0.0, color="0.6", lw=0.6, zorder=0)
    for i, (factor_type, values) in enumerate(
        zip(factor_types, all_values, strict=True)
    ):
        subset = values[mask]
        color = colors[[0, 3][i % 2]]
        threshold = float(np.percentile(values, 100 - top_perc))
        pct_top = 100 * float(np.mean(subset > threshold)) if subset.size else np.nan

        ax.hist(
            subset,
            bins=edges,
            weights=np.full(subset.size, 1 / max(subset.size, 1)),
            histtype="stepfilled",
            alpha=0.45,
            color=color,
            edgecolor=color,
            lw=0.8,
            label=(
                f"{factor_type.capitalize()}: "
                f"$\\mu$={subset.mean():+.3f}, {pct_top:.0f}% in top {top_perc:g}%"
                if subset.size
                else f"{factor_type.capitalize()}: no cells"
            ),
        )
        ax.axvline(threshold, color=color, ls="--", lw=0.8)

    if log_y:
        ax.set_yscale("log")
    # Headroom so the legend clears the tallest bin
    ax.set_ylim(top=ax.get_ylim()[1] * (8 if log_y else 1.6))
    ax.set_xlim(lo, hi)
    ax.set(xlabel=f"Cmp. {cmp} Weight", ylabel="Fraction of Cells")
    ax.set_title(f"{celltype} (n={int(mask.sum())})")
    ax.legend(loc="upper right")


def celltype_grid_shape(
    X: anndata.AnnData, celltype_col: str = "celltype", ncols: int = 4
) -> tuple[int, int]:
    """Rows and columns needed to give every cell type its own panel."""
    n_celltypes = len(np.unique(np.asarray(cast(pd.DataFrame, X.obs)[celltype_col])))
    ncols = min(ncols, n_celltypes)
    return math.ceil(n_celltypes / ncols), ncols


def plot_wc_histogram_grid(
    X: anndata.AnnData,
    cmp: int,
    axes: list[Axes] | np.ndarray,
    celltype_col: str = "celltype",
    factor_types: tuple[str, ...] = ("sender", "receiver"),
    **kwargs,
) -> None:
    """Fill `axes` with one `plot_wc_histogram` panel per cell type, hiding any
    leftover axes. Remaining keyword arguments go to `plot_wc_histogram`."""
    celltypes = np.unique(
        np.asarray(cast(pd.DataFrame, X.obs)[celltype_col]).astype(str)
    )
    axes = np.asarray(axes).ravel()

    for axis, celltype in zip(axes, celltypes, strict=False):
        plot_wc_histogram(
            X,
            cmp,
            axis,
            celltype,
            celltype_col=celltype_col,
            factor_types=factor_types,
            **kwargs,
        )

    for axis in axes[len(celltypes) :]:
        axis.axis("off")


def make_wc_histogram_figure(
    X: anndata.AnnData,
    cmp: int,
    celltype_col: str = "celltype",
    factor_types: tuple[str, ...] = ("sender", "receiver"),
    ncols: int = 4,
    panel_size: tuple[float, float] = (4.5, 3.2),
    **kwargs,
) -> Figure:
    """Build a standalone grid of `plot_wc_histogram` panels, one per cell type, for
    one component."""
    nrows, ncols = celltype_grid_shape(X, celltype_col, ncols)

    ax, f = getSetup(
        (panel_size[0] * ncols, panel_size[1] * nrows),
        (nrows, ncols),
    )
    plot_wc_histogram_grid(
        X,
        cmp,
        ax,
        celltype_col=celltype_col,
        factor_types=factor_types,
        **kwargs,
    )
    f.suptitle(f"Component {cmp}: Cell Weight Distributions by Cell Type")

    return f


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
