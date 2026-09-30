"""
Figure for CPD rank selection.

Plots the sweep produced by ``uv run cccrise-cpdrank``: full-tensor fit R2X,
a component-collinearity/degeneracy screen, and the CORCONDIA core-consistency
diagnostic, one panel each. Run that first.
"""

import matplotlib.pyplot as plt
import pandas as pd

from ...runners.datasets import RANK_SELECTION_DIR
from ..common import getSetup, load_runner_output, subplotLabel

# Change this to switch dataset. The RISErank the sweep was conditioned on is recorded in the sweep's sidecar.
SELECTED_DATASET = "alad"
SWEEP_FILE = f"cpd_rank_{SELECTED_DATASET}.parquet"

PANELS_PER_ROW = 3


def plot_fit_r2x(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    ax.plot(df["Rank"], df["FitR2X"], marker="s")
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Fit $R^2X$")
    ax.set_title(title)


def plot_collinearity(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    colors: tuple[str, str, str] = ("0.5", "tab:blue", "tab:red"),
) -> None:
    """`colors` is (no collinear pair, near-duplicate pair, canceling pair)."""
    none_color, duplicate_color, canceling_color = colors
    ax.plot(df["Rank"], df["MaxModesCollinear"], color="0.7", zorder=1)

    # A canceling pair, a near-duplicate pair, or no collinear pair
    # at all (NaN). `fillna` keeps the NaN rows out of both of the other masks.
    flag = df["WorstPairCanceling"]
    none_found = flag.isna()
    canceling = flag.fillna(False).astype(bool) & ~none_found
    duplicate = ~canceling & ~none_found

    ax.scatter(
        df.loc[none_found, "Rank"],
        df.loc[none_found, "MaxModesCollinear"],
        color=none_color,
        marker="o",
        label="no collinear pair",
        zorder=2,
    )
    ax.scatter(
        df.loc[duplicate, "Rank"],
        df.loc[duplicate, "MaxModesCollinear"],
        color=duplicate_color,
        marker="o",
        label="near-duplicate pair",
        zorder=2,
    )
    ax.scatter(
        df.loc[canceling, "Rank"],
        df.loc[canceling, "MaxModesCollinear"],
        color=canceling_color,
        marker="D",
        label="canceling pair",
        zorder=3,
    )

    ax.axhline(2, color="0.8", linestyle="--", linewidth=0.8)
    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Max modes worst pair is collinear in")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper left")


def plot_core_consistency(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    threshold_colors: tuple[str, str] = ("tab:green", "tab:orange"),
) -> None:
    """`threshold_colors` colors the 90% and 50% reference lines."""
    ax.plot(df["Rank"], df["CoreConsistency"], marker="o")
    ax.axhline(
        90, color=threshold_colors[0], linestyle="--", linewidth=0.8, label="90%"
    )
    ax.axhline(
        50, color=threshold_colors[1], linestyle="--", linewidth=0.8, label="50%"
    )
    # Core consistency collapses by orders of magnitude past the valid-model
    # rank. symlog keeps the 0-100% transition readable while still showing
    # the full collapse instead of one axis extreme crushing the rest flat.
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Core consistency (%)")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper right")


def makeFigure():
    ax, f = getSetup((4 * PANELS_PER_ROW, 3.5), (1, PANELS_PER_ROW))
    subplotLabel(ax)

    df = load_runner_output(SWEEP_FILE, RANK_SELECTION_DIR)
    label = SELECTED_DATASET.upper()

    plot_fit_r2x(df, ax[0], f"CPD Fit $R^2X$ ({label})")
    plot_collinearity(df, ax[1], f"Component Collinearity Screen ({label})")
    plot_core_consistency(df, ax[2], f"Core Consistency (CORCONDIA) ({label})")

    return f
