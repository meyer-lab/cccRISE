"""Supplementary figure: per-component and aggregate RISE and CP stability
under patient jackknifing and seed refits, and the collinearity of the CP
modes Figure 2 leaves out."""

from ... import common, style
from . import cp_collinearity, cpd_fms, rise_fms

PANELS = {"rise": rise_fms, "cpd": cpd_fms, "collinearity": cp_collinearity}
# RISE:CP widths match their box counts (components + gap + 2 aggregates),
# so boxes share a width.
LAYOUT = [["rise", "cpd"], ["collinearity"] * 2]
WIDTH_RATIOS = [2.55, 1]


def makeFigure():
    fig = common.compose(
        (style.FIGURE_WIDTH_DOUBLE, 120 * style.MM),
        LAYOUT,
        PANELS,
        height_ratios=[1.1, 0.9],
        width_ratios=WIDTH_RATIOS,
    )
    # The CP panel shares the RISE panel's FMS axis, labeled once
    rise_ax, cpd_ax = (subfig.axes[0] for subfig in fig.subfigs[:2])
    cpd_ax.sharey(rise_ax)
    cpd_ax.set_ylabel("")
    cpd_ax.tick_params(axis="y", labelleft=False)
    return fig
