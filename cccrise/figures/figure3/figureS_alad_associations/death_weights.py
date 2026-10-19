"""Condition loadings of the CP components significantly associated with
death, per group, as Fig. 3 shows the 6-month outcome association."""

from ... import common
from ..outcome_association import plot_outcome_weights

#: (association, CP component) pairs significant after FDR correction
TESTS = (
    ("6-Month Died vs. Alive", 4),
    ("1-Year Died vs. Alive", 4),
)
AXES = ("square",) * len(TESTS)


def plot(axs):
    for ax, (association, component) in zip(axs, TESTS, strict=True):
        plot_outcome_weights(ax, association, component)


def makeFigure():
    return common.standalone_fixed(__name__)
