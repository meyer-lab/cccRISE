"""Supplementary figure: aggregate RISE and CP stability under seed refits and
patient jackknifing, per-component stability under the jackknife, and the FMS
distribution of every bimodal component."""

from ... import common, style
from . import aggregate_fms, cpd_fms, fms_histograms, rise_fms

PANELS = {
    "aggregate": aggregate_fms,
    "rise": rise_fms,
    "cpd": cpd_fms,
    "histograms": fms_histograms,
}
# RISE:CP widths 3:1 match their component counts, so boxes share a width.
LAYOUT = [["aggregate", "rise", "cpd"], ["histograms"] * 3]
WIDTH_RATIOS = [2.4, 5, 1.67]


def makeFigure():
    fig = common.compose(
        (style.FIGURE_WIDTH_DOUBLE, 130 * style.MM),
        LAYOUT,
        PANELS,
        height_ratios=[1, 0.6],
        width_ratios=WIDTH_RATIOS,
    )
    # The CP panel shares the RISE panel's FMS axis, labeled once
    rise_ax, cpd_ax = (subfig.axes[0] for subfig in fig.subfigs[1:3])
    cpd_ax.sharey(rise_ax)
    cpd_ax.set_ylabel("")
    cpd_ax.tick_params(axis="y", labelleft=False)
    _align_floor(fig.subfigs[0].axes[0], rise_ax)
    return fig


def _align_floor(ax, reference) -> None:
    """Raise `ax`'s x axis to `reference`'s (whose x label sits lower), once
    constrained layout has placed both; the layout is then frozen. Both are
    in the same row of subfigures, so their y positions compare."""
    ax.figure.canvas.draw()
    ax.figure.get_figure(root=True).set_layout_engine("none")
    box, floor = ax.get_position(), reference.get_position().y0
    ax.set_position([box.x0, floor, box.width, box.y1 - floor])
