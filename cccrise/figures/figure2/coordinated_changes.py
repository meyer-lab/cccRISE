"""CPD component associations against raw communication scores, showing
that components capture coordinated changes no single LR pair shows alone.

Placeholder until the comparison is defined. One candidate: each
outcome-associated component's ANOVA effect size against the distribution of
single-LR-pair effect sizes computed from the raw communication scores.
"""

from .. import common, style

DATASET = "alad"


def plot(fig):
    common.placeholder(
        fig, "CPD associations vs. raw\ncommunication scores\n(analysis TBD)"
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
