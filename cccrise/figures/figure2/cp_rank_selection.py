"""CP rank selection: fit R2X and component collinearity, from
``uv run cccrise-cpdrank`` (conditioned on the canonical RISE rank)."""

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from .. import common

DATASET = "alad"
AXES = ("square", "square")


def plot(axs, dataset: str = DATASET):
    sweep = common.load_runner_output(
        DATASETS[dataset].cpd_rank_filename(), RANK_SELECTION_DIR
    )
    rank = common.selected_ranks(dataset)[1]
    common.plot_cpd_fit_r2x(sweep, axs[0], rank)
    common.plot_collinearity(sweep, axs[1], rank)


def makeFigure():
    return common.standalone_fixed(__name__)
