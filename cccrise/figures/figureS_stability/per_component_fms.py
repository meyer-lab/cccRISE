"""Per-component RISE and CPD stability under resampling of cells,
conditions, and LR pairs, against the seed-only reproducibility ceiling, from
``uv run cccrise-stability``."""

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...validation.component_stability import STAGE_CPD, STAGE_RISE
from .. import common, style

DATASET = "alad"


def load_stability():
    return common.load_runner_output(
        DATASETS[DATASET].stability_filename(), VALIDATION_DIR
    )


def plot(fig):
    stability = load_stability()
    axs = fig.subplots(1, 2, width_ratios=[3, 1])
    for ax, stage in zip(axs, (STAGE_RISE, STAGE_CPD), strict=True):
        common.plot_per_component_fms(stability, ax, stage)
        ax.set_title(f"{stage} component stability")
    axs[0].get_legend().remove()


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
