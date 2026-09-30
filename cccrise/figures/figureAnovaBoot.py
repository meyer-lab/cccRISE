"""
Figure for the patient-resampled bootstrap of the CCC-RISE ANOVA effect sizes.

Plots the replicates produced by ``uv run cccrise-anovaboot``; run that first.

The pipeline is re-run end to end -- RISE, the interaction tensor, CP at the
same fixed ranks, then the ANOVA -- on bootstrap resamples of the patient
roster, and each replicate's components are matched back onto the full-cohort
fit before its statistics are recorded (see
`validation.anova_stability.bootstrap_anova`). Every component's effect size
therefore comes with a distribution over rosters, which is what separates an
association that would survive a different cohort from one that describes the
patients who happened to be enrolled.

Panels, in order:

    a. Bootstrap effect size distributions -- one violin per CP component over
       the replicates, with the full-cohort eta-squared overlaid. Components
       whose full-cohort p-value is below alpha are drawn in the emphasis
       color; the rest are muted.
    b. Interval forest -- the same eta-squared intervals ranked by median
       magnitude and labeled with each component's dominant ligand-receptor pair,
       so the components worth interpreting can be read off directly. A hollow
       marker means the reference p-value is not significant.
    c. Stability of significance and sign -- how often the association reached
       p < alpha across rosters, and how often its group-mean difference agreed
       in direction with the full-cohort fit.
    d. Component identifiability -- the match score every replicate's
       component achieved against the reference fit. This panel is the
       control for panel a: where scores are low, a wide effect-size
       distribution says the CP component could not be re-identified across
       rosters, not that the phenotype association was unstable.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from ..factors import DATA_DIR
from ..runners.provenance import sidecar_path
from ..validation.anova_stability import summarize_bootstrap_anova
from .common import getSetup, load_runner_output, subplotLabel

# Change this to switch dataset; must match the runner's --dataset. The ranks
# the bootstrap was conditioned on are recorded in its sidecar.
SELECTED_DATASET = "alad"
BOOTSTRAP_FILE = f"anova_bootstrap_{SELECTED_DATASET}.parquet"

# Match score below which a replicate's component is not the reference
# component in any useful sense.
MATCH_SCORE_FLOOR = 0.7

# Polarity and status colors.
EMPHASIS_COLOR = "#3a6b9c"
MUTED_COLOR = "0.72"
INK = "0.25"

PANELS_PER_ROW = 4


def _component_order(summary: pd.DataFrame) -> list[int]:
    """Components in numeric order -- the numbering the runner aligned every
    replicate onto."""
    return summary["Component"].astype(int).tolist()


# =============================================================================
# Panel a: bootstrap effect size distributions
# =============================================================================


def plot_effect_size_distributions(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    alpha: float = 0.05,
) -> None:
    """Violin per component over the bootstrap replicates, full-cohort eta-squared
    overlaid."""
    order = _component_order(summary)
    is_sig = dict(
        zip(
            summary["Component"].astype(int), summary["ReferenceP"] < alpha, strict=True
        )
    )
    palette = {
        component: EMPHASIS_COLOR if is_sig[component] else MUTED_COLOR
        for component in order
    }

    sns.violinplot(
        data=df,
        x="Component",
        y="EtaSquared",
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
    reference = summary.set_index("Component").loc[order, "ReferenceEtaSquared"]
    ax.scatter(
        positions,
        reference.to_numpy(),
        marker="D",
        s=16,
        color="black",
        zorder=5,
    )

    # Headroom so the legend sits over empty space rather than over a violin.
    low, high = ax.get_ylim()
    ax.set_ylim(bottom=0, top=high + 0.12 * (high - low) if high > 0 else 1.0)
    ax.set_xlabel("CP Component")
    ax.set_ylabel(r"Effect size ($\eta^2$)")
    ax.set_title(title)
    handles = [
        plt.Line2D(
            [0], [0], color=EMPHASIS_COLOR, lw=6, label=f"Reference p < {alpha:g}"
        ),
        plt.Line2D(
            [0], [0], color=MUTED_COLOR, lw=6, label=f"Reference p \u2265 {alpha:g}"
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            lw=0,
            markersize=4,
            label=r"Full-cohort $\eta^2$",
        ),
    ]
    ax.legend(handles=handles, fontsize=5, ncol=3, loc="upper center")


# =============================================================================
# Panel b: interval forest
# =============================================================================


def plot_effect_size_intervals(
    df: pd.DataFrame, ax: plt.Axes, title: str, alpha: float = 0.05
) -> None:
    """Bootstrap interval per component, ranked by median magnitude and
    labeled with its dominant LR pair."""
    ranked = df.reindex(
        df["MedianEtaSquared"].sort_values(ascending=True).index
    ).reset_index(drop=True)

    positions = np.arange(len(ranked))
    medians = ranked["MedianEtaSquared"].to_numpy()
    lows = ranked["CILow"].to_numpy()
    highs = ranked["CIHigh"].to_numpy()
    is_sig = (ranked["ReferenceP"] < alpha).to_numpy(dtype=bool)

    for pos in positions:
        color = EMPHASIS_COLOR if is_sig[pos] else MUTED_COLOR
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
            facecolors=color if is_sig[pos] else "white",
            edgecolors=color,
            linewidths=0.8,
            marker="o" if is_sig[pos] else "s",
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
    ax.set_xlabel(r"Effect size ($\eta^2$)")
    ax.set_title(title)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=EMPHASIS_COLOR,
            lw=2,
            markersize=4,
            label=f"Reference p < {alpha:g}",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color=MUTED_COLOR,
            lw=2,
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=MUTED_COLOR,
            label=f"Reference p \u2265 {alpha:g}",
        ),
    ]
    ax.legend(handles=handles, fontsize=5, loc="lower right")


# =============================================================================
# Panel c: significance and sign stability
# =============================================================================


def plot_significance_and_sign_stability(
    df: pd.DataFrame, ax: plt.Axes, title: str, alpha: float = 0.05
) -> None:
    """Fraction of replicates with p < alpha, and agreeing with the
    full-cohort group mean difference direction."""
    positions = np.arange(len(df))
    width = 0.38

    ax.bar(
        positions - width / 2,
        df["FracSignificant"],
        width=width,
        color=EMPHASIS_COLOR,
        label=f"p < {alpha:g} across rosters",
    )
    ax.bar(
        positions + width / 2,
        df["FracSameSignAsReference"],
        width=width,
        color="#c98b3a",
        label="Same direction as full cohort",
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
        color="tab:red",
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
    bootstrap_anova_df = load_runner_output(BOOTSTRAP_FILE)

    sidecar = sidecar_path(DATA_DIR / BOOTSTRAP_FILE)
    with open(sidecar) as handle:
        metadata = yaml.safe_load(handle)

    n_comp = len(metadata["reference_eta_squared"])
    alpha = float(metadata.get("alpha", 0.05))
    ci_level = float(metadata.get("ci_level", 95.0))
    ref_f = metadata.get("reference_f", [np.nan] * n_comp)
    ref_diff = metadata.get("reference_group_mean_diff", [np.nan] * n_comp)

    reference_anova = pd.DataFrame(
        {
            "Component": np.arange(1, n_comp + 1),
            "EtaSquared": np.asarray(metadata["reference_eta_squared"], dtype=float),
            "F": np.asarray(ref_f, dtype=float),
            "P": np.asarray(metadata["reference_p"], dtype=float),
            "GroupMeanDiff": np.asarray(ref_diff, dtype=float),
        }
    )

    bootstrap_summary = summarize_bootstrap_anova(
        bootstrap_anova_df, reference_anova, ci_level=ci_level, alpha=alpha
    )

    # Labels come from the bootstrap's own reference fit, whose component
    # numbering every replicate was aligned onto.
    bootstrap_summary["TopLRPair"] = metadata["reference_top_lr_pair"]
    bootstrap_summary["TopLRLoading"] = metadata["reference_top_lr_loading"]

    n_components = len(bootstrap_summary)
    panels = [
        (
            "Bootstrap Effect Size Distributions",
            lambda axis: plot_effect_size_distributions(
                bootstrap_anova_df,
                bootstrap_summary,
                axis,
                f"Bootstrap Effect Size Distributions ({SELECTED_DATASET.upper()})",
                alpha=alpha,
            ),
        ),
        (
            "Effect Size Intervals",
            lambda axis: plot_effect_size_intervals(
                bootstrap_summary,
                axis,
                f"Effect Size Intervals ({SELECTED_DATASET.upper()})",
                alpha=alpha,
            ),
        ),
        (
            "Significance and Direction Stability",
            lambda axis: plot_significance_and_sign_stability(
                bootstrap_summary,
                axis,
                f"Significance and Direction Stability ({SELECTED_DATASET.upper()})",
                alpha=alpha,
            ),
        ),
        (
            "Component Identifiability",
            lambda axis: plot_match_scores(
                bootstrap_anova_df,
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
