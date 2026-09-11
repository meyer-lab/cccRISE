"""
Figure for the patient-resampled bootstrap of the CCC-RISE logistic
regression weights.

Plots the replicates produced by ``uv run cccrise-logregboot``; run that
first.

The pipeline is re-run end to end -- RISE, the interaction tensor, CP at the
same fixed ranks, then the logistic regression -- on bootstrap resamples of
the patient roster, and each replicate's components are matched back onto the
full-cohort fit before its weights are recorded (see
`stability.logreg_stability.bootstrap_logreg_weights`). Every component's weight therefore
comes with a distribution over rosters, which is what separates a weight that
would survive a different cohort from one that describes the patients who
happened to be enrolled.

Panels, in order:

    a. Bootstrap weight distributions -- one violin per CP component over the
       replicates, with the full-cohort weight overlaid. Components whose
       central interval excludes zero are drawn in the emphasis color; the
       rest are muted, since a weight whose interval spans zero is not
       distinguishable from no association at all.
    b. Interval forest -- the same intervals ranked by median magnitude and
       labeled with each component's dominant ligand-receptor pair, so the
       components worth interpreting can be read off directly. A hollow
       marker means the interval covers zero.
    c. Stability of selection and sign -- how often the L1 penalty kept the
       component at all, and how often its weight agreed in sign with the
       full-cohort fit. A component can have a tight interval and still fail
       here, by being dropped entirely for some rosters.
    d. Component identifiability -- the match score every replicate's
       component achieved against the reference fit. This panel is the
       control for panel a: where scores are low, a wide weight distribution
       says the CP component could not be re-identified across rosters, not
       that the regression was unstable.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from ..factors import DATA_DIR, load_factors_to_anndata
from ..runners.datasets import DATASETS
from ..runners.provenance import sidecar_path
from ..stability.logreg_stability import summarize_bootstrap_weights, top_lr_pairs
from .common import getSetup, load_runner_csv, subplotLabel

# Change this to switch dataset; must match the runner's --dataset. The ranks
# the bootstrap was conditioned on are recorded in its sidecar.
SELECTED_DATASET = "alad"
BOOTSTRAP_WEIGHTS_CSV = f"logreg_bootstrap_weights_{SELECTED_DATASET}.csv"

# Match score below which a replicate's component is not the reference
# component in any useful sense.
MATCH_SCORE_FLOOR = 0.7

# Polarity and status colors, held to one job each: the sign of a weight, and
# whether its interval clears zero.
NEGATIVE_COLOR = "tab:red"
POSITIVE_COLOR = "tab:blue"
EMPHASIS_COLOR = "#3a6b9c"
MUTED_COLOR = "0.72"
INK = "0.25"

PANELS_PER_ROW = 4


def _component_order(summary: pd.DataFrame) -> list[int]:
    """Components in numeric order -- the numbering the runner aligned every
    replicate onto."""
    return summary["Component"].astype(int).tolist()


# =============================================================================
# Panel a: bootstrap weight distributions
# =============================================================================


def plot_weight_distributions(
    df: pd.DataFrame, summary: pd.DataFrame, ax: plt.Axes, title: str
) -> None:
    """Violin per component over the bootstrap replicates, full-cohort weight
    overlaid."""
    order = _component_order(summary)
    excludes_zero = dict(
        zip(summary["Component"].astype(int), summary["CIExcludesZero"], strict=True)
    )
    palette = {
        component: EMPHASIS_COLOR if excludes_zero[component] else MUTED_COLOR
        for component in order
    }

    sns.violinplot(
        data=df,
        x="Component",
        y="Weight",
        hue="Component",
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        inner="quartile",
        linewidth=0.6,
        linecolor=INK,
        cut=0,
        density_norm="width",
        ax=ax,
    )

    positions = np.arange(len(order))
    reference = summary.set_index("Component").loc[order, "ReferenceWeight"]
    ax.scatter(
        positions,
        reference.to_numpy(),
        marker="D",
        s=16,
        color="black",
        zorder=5,
    )

    ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    # Headroom so the legend sits over empty space rather than over a violin.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.12 * (high - low))
    ax.set_xlabel("CP Component")
    ax.set_ylabel("Logistic regression weight")
    ax.set_title(title)
    handles = [
        plt.Line2D([0], [0], color=EMPHASIS_COLOR, lw=6, label="Interval excludes 0"),
        plt.Line2D([0], [0], color=MUTED_COLOR, lw=6, label="Interval covers 0"),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            lw=0,
            markersize=4,
            label="Full-cohort weight",
        ),
    ]
    ax.legend(handles=handles, fontsize=5, ncol=3, loc="upper center")


# =============================================================================
# Panel b: interval forest
# =============================================================================


def plot_weight_intervals(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    """Bootstrap interval per component, ranked by median magnitude and
    labeled with its dominant LR pair."""
    ranked = df.reindex(
        df["MedianWeight"].abs().sort_values(ascending=True).index
    ).reset_index(drop=True)

    positions = np.arange(len(ranked))
    medians = ranked["MedianWeight"].to_numpy()
    lows = ranked["CILow"].to_numpy()
    highs = ranked["CIHigh"].to_numpy()
    clears_zero = ranked["CIExcludesZero"].to_numpy(dtype=bool)

    for pos in positions:
        color = NEGATIVE_COLOR if medians[pos] < 0 else POSITIVE_COLOR
        ax.plot(
            [lows[pos], highs[pos]],
            [pos, pos],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            medians[pos],
            pos,
            s=26,
            facecolors=color if clears_zero[pos] else "white",
            edgecolors=color,
            linewidths=0.8,
            marker="o" if clears_zero[pos] else "s",
            zorder=3,
        )

    labels = [
        f"C{component}  {pair}"
        for component, pair in zip(
            ranked["Component"].to_numpy(dtype=int),
            ranked["TopLRPair"],
            strict=True,
        )
    ]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=5)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.set_xlabel("Logistic regression weight")
    ax.set_title(title)
    handles = [
        plt.Line2D([0], [0], color=POSITIVE_COLOR, lw=3, label="Positive median"),
        plt.Line2D([0], [0], color=NEGATIVE_COLOR, lw=3, label="Negative median"),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            lw=0,
            markersize=4,
            markerfacecolor="white",
            markeredgecolor="0.4",
            label="Interval covers 0",
        ),
    ]
    ax.legend(handles=handles, fontsize=5, loc="lower right")


# =============================================================================
# Panel c: selection and sign stability
# =============================================================================


def plot_selection_and_sign_stability(
    df: pd.DataFrame, ax: plt.Axes, title: str
) -> None:
    """Fraction of replicates keeping the component, and agreeing with the
    full-cohort sign."""
    positions = np.arange(len(df))
    width = 0.38

    ax.bar(
        positions - width / 2,
        df["FracNonzero"],
        width=width,
        color=EMPHASIS_COLOR,
        label="Kept by the L1 penalty",
    )
    ax.bar(
        positions + width / 2,
        df["FracSameSignAsReference"],
        width=width,
        color="#c98b3a",
        label="Same sign as full cohort",
    )

    ax.axhline(0.5, color="0.85", linewidth=0.8, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(df["Component"].astype(int), fontsize=6)
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("CP Component")
    ax.set_ylabel("Fraction of replicates")
    ax.set_title(title)
    ax.legend(fontsize=5, ncol=2, loc="upper center")


# =============================================================================
# Panel d: component identifiability
# =============================================================================


def plot_match_scores(
    df: pd.DataFrame, summary: pd.DataFrame, ax: plt.Axes, title: str
) -> None:
    """Match score each replicate's component achieved against the reference
    fit, per component."""
    order = _component_order(summary)
    sns.boxplot(
        data=df,
        x="Component",
        y="MatchScore",
        order=order,
        color=MUTED_COLOR,
        linewidth=0.6,
        fliersize=1.5,
        ax=ax,
    )
    ax.axhline(
        MATCH_SCORE_FLOOR,
        color=NEGATIVE_COLOR,
        linewidth=0.9,
        linestyle="--",
        label=f"Weak-match floor ({MATCH_SCORE_FLOOR})",
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("CP Component")
    ax.set_ylabel("Match score vs. full-cohort component")
    ax.set_title(title)
    ax.legend(fontsize=5, loc="lower right")


# =============================================================================
# Figure assembly
# =============================================================================


def makeFigure():
    bootstrap_weights = load_runner_csv(BOOTSTRAP_WEIGHTS_CSV)

    sidecar = sidecar_path(DATA_DIR / BOOTSTRAP_WEIGHTS_CSV)
    with open(sidecar) as handle:
        metadata = yaml.safe_load(handle)
    reference_weights = np.asarray(metadata["reference_logreg_weights"], dtype=float)

    bootstrap_summary = summarize_bootstrap_weights(
        bootstrap_weights, reference_weights
    )

    factors_path = DATASETS[SELECTED_DATASET].factors_path()
    factors_ds = load_factors_to_anndata(factors_path)
    labels = top_lr_pairs(
        factors_ds.uns["D"], factors_ds.uns["lr_pairs"], top_n=1
    )
    bootstrap_summary = bootstrap_summary.merge(
        labels[["Component", "LRPair", "DLoading"]].rename(
            columns={"LRPair": "TopLRPair", "DLoading": "TopLRLoading"}
        ),
        on="Component",
        how="left",
    )

    n_components = len(bootstrap_summary)
    panels = [
        (
            "Bootstrap Weight Distributions",
            lambda axis: plot_weight_distributions(
                bootstrap_weights,
                bootstrap_summary,
                axis,
                f"Bootstrap Weight Distributions ({SELECTED_DATASET.upper()})",
            ),
        ),
        (
            "Weight Intervals",
            lambda axis: plot_weight_intervals(
                bootstrap_summary,
                axis,
                f"Weight Intervals ({SELECTED_DATASET.upper()})",
            ),
        ),
        (
            "Selection and Sign Stability",
            lambda axis: plot_selection_and_sign_stability(
                bootstrap_summary,
                axis,
                f"Selection and Sign Stability ({SELECTED_DATASET.upper()})",
            ),
        ),
        (
            "Component Identifiability",
            lambda axis: plot_match_scores(
                bootstrap_weights,
                bootstrap_summary,
                axis,
                f"Component Identifiability ({SELECTED_DATASET.upper()})",
            ),
        ),
    ]

    ncols = min(PANELS_PER_ROW, len(panels))
    nrows = -(-len(panels) // ncols)  # ceil
    panel_width = max(4.5, 0.28 * n_components + 2.5)
    ax, f = getSetup((panel_width * ncols, 3.8 * nrows), (nrows, ncols))
    subplotLabel(ax)

    for axis, (_label, plot) in zip(ax, panels, strict=False):
        plot(axis)

    for axis in ax[len(panels) :]:
        axis.set_visible(False)

    print(bootstrap_summary.to_string(index=False))
    return f
