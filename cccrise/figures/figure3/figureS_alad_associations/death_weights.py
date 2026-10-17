"""Condition loadings of the CP components significantly associated with
death, per group, as Fig. 3 shows the 6-month outcome association."""

from ... import common
from ..outcome_association import plot_outcome_weights

#: (association, CP component) pairs significant after FDR correction
# TODO: 1-Year Died vs. Alive, comp 5 is no longer significant (PAdj 0.75)
# with read-depth-corrected loadings; decide whether to keep this panel.
TESTS = (
    ("6-Month Died vs. Alive", 4),
    ("1-Year Died vs. Alive", 4),
    ("1-Year Died vs. Alive", 5),
)
#: Fraction of the data range added above it, clearing the ANOVA statistics
STATS_HEADROOM = 0.5
AXES = ("square",) * len(TESTS)


def plot(axs):
    for ax, (association, component) in zip(axs, TESTS, strict=True):
        plot_outcome_weights(ax, association, component)
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + STATS_HEADROOM * (high - low))


def makeFigure():
    return common.standalone_fixed(__name__)
