"""Supplementary figure: per-component RISE and CP stability under patient
jackknifing, with the FMS distribution of every bimodal component."""

from ... import common, style
from . import cpd_fms, fms_histograms, rise_fms

PANELS = {"rise": rise_fms, "cpd": cpd_fms, "histograms": fms_histograms}
# 3:1 matches the RISE:CP component counts, so boxes share a width.
LAYOUT = [["rise"] * 3 + ["cpd"], ["histograms"] * 4]


def makeFigure():
    fig = common.compose(
        (style.FIGURE_WIDTH_DOUBLE, 130 * style.MM),
        LAYOUT,
        PANELS,
        height_ratios=[1, 0.6],
    )
    # The CP panel shares the RISE panel's FMS axis, labeled once
    rise_ax, cpd_ax = (subfig.axes[0] for subfig in fig.subfigs[:2])
    cpd_ax.sharey(rise_ax)
    cpd_ax.set_ylabel("")
    cpd_ax.tick_params(axis="y", labelleft=False)
    return fig
