"""CPD rank selection: fit R2X, component collinearity, and core consistency,
from ``uv run cccrise-cpdrank`` (conditioned on the canonical RISE rank)."""

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from .. import common, style

DATASET = "alad"


def plot(fig):
    axs = fig.subplots(1, 3)
    sweep = common.load_runner_output(
        DATASETS[DATASET].cpd_rank_filename(), RANK_SELECTION_DIR
    )
    rank = common.selected_ranks(DATASET)[1]
    common.plot_cpd_fit_r2x(sweep, axs[0], rank)
    common.plot_collinearity(sweep, axs[1], rank)
    common.plot_core_consistency(sweep, axs[2], rank)
    for ax, title in zip(
        axs, ("CPD fit", "Component collinearity", "Core consistency"), strict=True
    ):
        ax.set_title(title)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
