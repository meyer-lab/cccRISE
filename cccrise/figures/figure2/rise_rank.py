"""RISE rank selection by bi-cross-validation, from ``uv run cccrise-riserank``."""

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from .. import common, style

DATASET = "alad"


def plot(fig):
    ax = fig.subplots()
    sweep = common.load_runner_output(
        DATASETS[DATASET].rise_rank_filename(), RANK_SELECTION_DIR
    )
    common.plot_rise_rank(sweep, ax, rank=common.selected_ranks(DATASET)[0])
    ax.set_title("RISE rank selection")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
