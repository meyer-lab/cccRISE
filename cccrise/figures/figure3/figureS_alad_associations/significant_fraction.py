"""Fraction of leave-one-patient-out replicates in which each CPD component's
association with the 6-month outcome has Benjamini-Hochberg q < 0.05 across the
replicate's components."""

from ....validation.stability import JACKKNIFE_AXIS
from ... import common, style
from . import _anova_stability as S

AXES = ("square",)


def plot(axs):
    (ax,) = axs
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
    ax.set_xlabel("CPD component")
    ax.set_ylabel(f"Fraction of replicates\nwith q < {S.alpha():g}")
    S.save_source_data()


def makeFigure():
    return common.standalone_fixed(__name__)
