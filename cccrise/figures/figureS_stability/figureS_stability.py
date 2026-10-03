"""Supplementary figure: per-component RISE and CP stability under
resampling."""

from .. import common, style
from . import _common as S
from . import cpd_fms, rise_fms

PANELS = {"rise": rise_fms, "cpd": cpd_fms}
# 3:1 matches the RISE:CP component counts, so boxes share a width.
LAYOUT = [["rise"] * 3 + ["cpd"]]


def makeFigure():
    fig = common.compose((style.FIGURE_WIDTH_DOUBLE, 75 * style.MM), LAYOUT, PANELS)
    # The CP panel shares the RISE panel's FMS axis, labeled once
    rise_ax, cpd_ax = (subfig.axes[0] for subfig in fig.subfigs)
    cpd_ax.sharey(rise_ax)
    cpd_ax.set_ylabel("")
    cpd_ax.tick_params(axis="y", labelleft=False)
    S.legend(fig)
    return fig
