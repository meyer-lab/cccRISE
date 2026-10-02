"""Co-variation across conditions of the outcome-associated CP component's top
LR pairs, against random, expression-matched, and other associated LR pairs.

LR pairs are scored per condition in the component's own sender/receiver
context (`_common.component_lr_scores`). Co-variation of a set of LR pairs is
their mean pairwise Spearman correlation across conditions
(`_common.mean_pairwise_rank_correlation`): each pair's scores are ranked
across conditions and the correlation of those ranks is averaged over every
pair in the set. Ranks rather than raw scores keep a few conditions in which
many LR pairs spike together from dominating the correlation.

The component's `N_TOP` most heavily weighted LR pairs give the observed
value (the line). Each histogram is `COVARIATION_DRAWS` draws of as many LR
pairs from outside that set (`_common.covariation_baselines`):

- Random: drawn uniformly.
- Expression-matched: one pair per top pair, from that pair's
  `N_EXPRESSION_NEIGHBORS` nearest pairs in log mean score, so a draw has the
  top pairs' expression levels.
- Other associated: drawn from the LR pairs significantly associated with the
  outcome on their own (FDR < 0.05, see `lr_association`).
"""

import numpy as np

from .. import common, style
from . import _common as C

BASELINES = {  # label, color
    "random": ("Random", style.COLORS["muted"]),
    "expression_matched": ("Expression-matched", style.COLORS["receiver"]),
    "associated": ("Other associated", style.COLORS["up"]),
}


def plot(fig):
    ax = fig.subplots()
    result = C.coordinated_analysis()
    plot_lr_covariation(result["observed"], result["baselines"], ax)
    ax.set_title("Co-variation of LR pairs")


def plot_lr_covariation(observed: float, baselines: dict[str, np.ndarray], ax) -> None:
    lowest = min(b.min() for b in baselines.values())
    highest = max(observed, *(b.max() for b in baselines.values()))
    bins = np.linspace(lowest, highest, 40)
    for name, draws in baselines.items():
        label, color = BASELINES[name]
        ax.hist(draws, bins=bins, color=color, alpha=0.7, label=label)
    ax.axvline(
        observed,
        color=style.COLORS["emphasis"],
        linewidth=style.LINE_WIDTH,
        label=f"Cmp. {C.ASSOCIATED_COMPONENT} top {C.N_TOP}",
    )
    ax.set(xlabel="Mean pairwise rank\ncorrelation across conditions", ylabel="Draws")
    ax.legend(**C.OUTSIDE_LEGEND)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
