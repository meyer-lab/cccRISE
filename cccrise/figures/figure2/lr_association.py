"""The outcome-associated CP component's association against each LR pair's
own association in the raw communication scores.

Every LR pair is scored per condition in the component's own sender/receiver
context (`_common.component_lr_scores`: the raw tensor slice for that pair,
weighted by the component's sender and receiver eigenstate loadings and
summed over eigenstates). Each pair's scores, and the component's condition
loadings, then get Figure 3's component ANOVA against the 6-month outcome
(`_common.column_associations`: one-way ANOVA over the outcome groups, eta^2
as the effect size, Benjamini-Hochberg FDR across the 513 LR pairs). The
histogram is every LR pair's eta^2, with the whole component's eta^2 and its
`N_TOP` most heavily weighted LR pairs' marked.
"""

import numpy as np

from .. import common, style
from . import _common as C


def plot(fig):
    ax = fig.subplots()
    result = C.coordinated_analysis()
    plot_lr_association_distribution(
        result["lr_eta"], result["component_eta"], result["top"], ax
    )
    ax.set_title("Component vs. single LR pairs")


def plot_lr_association_distribution(
    lr_eta: np.ndarray, component_eta: float, top: np.ndarray, ax
) -> None:
    ax.hist(lr_eta, bins=40, color=style.COLORS["muted"], label="All LR pairs")
    for i, eta in enumerate(lr_eta[top]):
        ax.axvline(
            eta,
            ymax=0.08,
            color=style.COLORS["highlight"],
            linewidth=style.LINE_WIDTH,
            label=f"Cmp. {C.ASSOCIATED_COMPONENT} top {len(top)}" if i == 0 else None,
        )
    ax.axvline(
        component_eta,
        color=style.COLORS["emphasis"],
        linewidth=style.LINE_WIDTH,
        label=f"Cmp. {C.ASSOCIATED_COMPONENT}",
    )
    ax.set(xlabel=r"Outcome association ($\eta^2$)", ylabel="LR pairs")
    ax.legend(**C.OUTSIDE_LEGEND)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
