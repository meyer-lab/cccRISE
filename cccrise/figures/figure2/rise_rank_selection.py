"""RISE rank selection by bi-cross-validation, from ``uv run cccrise-riserank``."""

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from .. import common

DATASET = "alad"
AXES = ("rect",)


def plot(axs, dataset: str = DATASET):
    (ax,) = axs
    sweep = common.load_runner_output(
        DATASETS[dataset].rise_rank_filename(), RANK_SELECTION_DIR
    )
    common.plot_rise_rank(sweep, ax, rank=common.selected_ranks(dataset)[0])


def makeFigure():
    return common.standalone_fixed(__name__)
