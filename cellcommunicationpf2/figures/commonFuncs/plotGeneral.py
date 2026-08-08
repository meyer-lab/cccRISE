from typing import cast

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ...tensor import calculate_fms_rise, rise_store_r2x
from ...utils import resample


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


def plot_fms_r2x_diff_ranks(
    X: anndata.AnnData,
    condition_name: str,
    ax1: Axes,
    ax2: Axes,
    ranksList: list[int],
    runs: int,
):
    """Plots FMS when using different RISE components"""
    rows = []
    for j in range(runs):
        for i in ranksList:
            print(f"Run {j + 1}, Rank {i}")
            dataX, r2x = rise_store_r2x(X, rank=i, n_iter_max=1000, tolerance=1e-9)
            sampledX, _ = rise_store_r2x(
                resample(X, condition_name=condition_name),
                rank=i,
                n_iter_max=1000,
                tolerance=1e-9,
            )
            fmsScore = calculate_fms_rise(dataX, sampledX)
            # R2X is for X only (not resampled)
            rows.append({"Run": j, "Component": i, "FMS": fmsScore, "R2X": r2x})

    df = pd.DataFrame(rows)

    # Plot FMS
    sns.lineplot(data=df, x="Component", y="FMS", ax=ax1, label="FMS")
    ax1.set_ylim(0, 1)

    # Plot R2X on a secondary y-axis
    sns.lineplot(data=df, x="Component", y="R2X", ax=ax2, color="orange", label="R2X")
    ax2.set_ylim(0, np.max(df["R2X"]) + 0.02)
