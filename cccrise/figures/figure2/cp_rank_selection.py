"""CP rank selection: fit and imputation R2X, and component collinearity, from
``uv run cccrise-cpdrank`` (conditioned on the canonical RISE rank)."""

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from .. import common, style

DATASET = "alad"
AXES = ("square", "square")


def plot(axs):
    sweep = common.load_runner_output(
        DATASETS[DATASET].cpd_rank_filename(), RANK_SELECTION_DIR
    )
    rank = common.selected_ranks(DATASET)[1]
    common.plot_cpd_fit_r2x(
        sweep, axs[0], rank, color=style.COLORS["fit_full"], marker=None
    )
    common.plot_cpd_imputation_r2x(sweep, axs[0])
    common.plot_collinearity(sweep, axs[1], rank)


def makeFigure():
    return common.standalone_fixed(__name__)
