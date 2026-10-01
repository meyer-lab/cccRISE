"""Per-component RISE stability under resampling of cells and conditions,
against the seed-only reproducibility ceiling, from ``uv run cccrise-stability``."""

from ...validation.component_stability import STAGE_RISE
from .. import common, style

DATASET = "alad"


def plot(fig):
    ax = fig.subplots()
    common.plot_per_component_fms(common.load_stability(DATASET), ax, STAGE_RISE)
    ax.set_title("RISE component stability")


def makeFigure():
    return common.standalone(
        plot, style.PANEL_SIZE["full"], legend=common.stability_legend
    )
