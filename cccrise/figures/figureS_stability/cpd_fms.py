"""Per-component CPD stability under resampling of cells, conditions, and LR
pairs, against the seed-only reproducibility ceiling, from
``uv run cccrise-stability``."""

from ...validation.component_stability import STAGE_CPD
from .. import common, style

DATASET = "alad"


def plot(fig):
    ax = fig.subplots()
    common.plot_per_component_fms(common.load_stability(DATASET), ax, STAGE_CPD)
    ax.set_title("CPD component stability")


def makeFigure():
    return common.standalone(
        plot, style.PANEL_SIZE["half"], legend=common.stability_legend
    )
