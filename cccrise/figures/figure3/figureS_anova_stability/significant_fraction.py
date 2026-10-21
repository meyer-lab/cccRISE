"""Fraction of leave-one-patient-out replicates in which each CP component's
outcome association stays significant (q < alpha, Benjamini-Hochberg
corrected across the replicate's components)."""

from ....validation.stability import JACKKNIFE_AXIS
from ... import common, style
from . import _common as S


def plot(fig):
    ax = fig.subplots()
    significant = (S.replicates()["PAdj"] < S.alpha()).groupby(
        S.replicates()["Component"]
    )
    fraction, count, total = significant.mean(), significant.sum(), significant.size()
    components = S.components()
    ax.bar(
        range(len(components)),
        fraction[components],
        color=style.STABILITY_AXIS_COLORS[JACKKNIFE_AXIS],
        edgecolor=style.COLORS["ink"],
        linewidth=style.LINE_WIDTH_THIN,
    )
    for x, k in enumerate(components):
        ax.text(
            x,
            fraction[k] + 0.02,
            f"{count[k]}/{total[k]}",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
        )
    ax.set_xticks(range(len(components)), components)
    ax.set_ylim(0, 1.15)  # count headroom
    ax.set_xlabel("CP component")
    ax.set_ylabel(f"Fraction of replicates\nwith q < {S.alpha():g}")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
