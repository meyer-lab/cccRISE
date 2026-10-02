"""Interpretation of the outcome-associated components: their LR pairs,
sender and receiver cell types, and gene programs.

Placeholder until the panels are chosen (#213 Fig 3.3-3.7).
"""

from .. import common, style


def plot(fig):
    common.placeholder(
        fig,
        "Interpretation of outcome-associated components\n"
        "(LR pairs, sender/receiver cell types, gene programs; TBD)",
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
