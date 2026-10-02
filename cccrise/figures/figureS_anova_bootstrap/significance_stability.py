"""Per CP component, the fraction of bootstrap rosters reaching p < alpha,
and -- for a two-level outcome, where a group-mean direction exists --
agreeing with the full cohort on which group loads higher."""

import numpy as np

from .. import common, style
from . import _shared as S


def plot(fig):
    ax = fig.subplots()
    table = S.summary()
    series = [("FracSignificant", style.COLORS["emphasis"], f"p < {S.alpha():g}")]
    if table["FracSameSignAsReference"].notna().any():
        series.append(
            ("FracSameSignAsReference", style.COLORS["highlight"], "Same direction")
        )
    x = np.arange(len(table))
    width = 0.8 / len(series)
    for i, (column, color, label) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, table[column], width, color=color, label=label)
    common.reference_line(ax, 0.5)
    ax.set_xticks(x, table["Component"].astype(int))
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("CP component")
    ax.set_ylabel("Fraction of replicates")
    ax.set_title("Significance stability")
    ax.legend(ncols=len(series), loc="upper center")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
