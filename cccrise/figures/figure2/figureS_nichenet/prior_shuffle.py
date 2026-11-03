"""NicheNet downstream-target validation of CP components against a
shuffled prior (formerly Figure 2's NicheNet panel): do the ligands a
component weights most have the best-supported downstream targets in its
receiver cells?

Per component, the weight-averaged target-score percentile of its ligands
(``nichenet.weighted_percentile``; diamond) against the same average with
every ligand re-scored on copies of the NicheNet prior with its ligand-target
links shuffled at random (``nichenet.shuffled_percentiles``; box: quartiles,
whiskers at the 2.5th and 97.5th percentiles of the shuffles), and a star
where the one-sided permutation p, Benjamini-Hochberg adjusted across
components, is below `SIG_ALPHA` (``nichenet.weighted_percentile_tests``),
else n.s."""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ....validation.nichenet import weighted_percentile_tests
from ... import common, style
from ..nichenet import BOX_FACE, BOX_STYLE, load_null, load_scores, significance_marks

NULL_WHISKERS = (2.5, 97.5)
AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    tests, null = weighted_percentile_tests(load_scores(), load_null())
    common.source_data(
        "nichenet_weighted_percentile_tests",
        tests,
        "Per CP component: its top ligands, the weight-averaged NicheNet "
        "target-score percentile of its ligands (weighted_pct), the mean of "
        "that average over shuffled NicheNet priors (null_mean), and the "
        "one-sided permutation p, Benjamini-Hochberg adjusted across "
        "components (q).",
    )
    common.source_data(
        "nichenet_shuffled_prior_percentiles",
        null.rename("weighted_pct"),
        "The weight-averaged target-score percentile per CP component on each "
        "shuffled NicheNet prior (replicate): the null distribution boxed.",
    )
    tests = tests.set_index("component")
    components = list(tests.index)
    positions = range(len(components))
    ax.boxplot(
        [null.loc[c] for c in components],
        positions=positions,
        whis=NULL_WHISKERS,
        zorder=1,
        **BOX_STYLE,
    )
    ax.scatter(
        positions,
        tests["weighted_pct"],
        marker="D",
        s=style.POINT_SIZE_FOREGROUND,
        color=style.COLORS["emphasis"],
        zorder=3,
    )
    significance_marks(
        ax,
        positions,
        [
            max(tests.loc[c, "weighted_pct"], null.loc[c].quantile(0.975))
            for c in components
        ],
        tests["q"].loc[components],
    )
    common.reference_line(ax, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(positions, components)
    ax.set_xlim(-0.6, len(components) - 0.4)
    ax.set_xlabel("CP component")
    ax.set_ylabel("Weighted NicheNet\ntarget-score percentile")
    ax.legend(
        loc="lower center",
        ncols=2,
        handles=[
            Line2D(
                [],
                [],
                marker="D",
                ls="",
                color=style.COLORS["emphasis"],
                markersize=style.MARKER_SIZE,
                label="Observed",
            ),
            Patch(
                facecolor=BOX_FACE,
                edgecolor=style.COLORS["reference"],
                lw=style.LINE_WIDTH_THIN,
                label="Shuffled NicheNet",
            ),
        ],
    )


def makeFigure():
    return common.standalone_fixed(__name__)
