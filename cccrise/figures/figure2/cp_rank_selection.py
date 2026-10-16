"""CP rank selection: fit R2X and component collinearity, from
``uv run cccrise-cpdrank`` (conditioned on the canonical RISE rank)."""

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from .. import common, style

DATASET = "alad"
AXES = ("square", "square")
# TODO: placeholder masked-imputation R2X until the CP imputation runner lands
PLACEHOLDER_IMPUTATION_SCALE = 0.5
PLACEHOLDER_IMPUTATION_SD = 0.1  # fraction of the curve


def plot(axs):
    sweep = common.load_runner_output(
        DATASETS[DATASET].cpd_rank_filename(), RANK_SELECTION_DIR
    )
    rank = common.selected_ranks(DATASET)[1]
    common.plot_cpd_fit_r2x(sweep, axs[0], rank, color=style.COLORS["fit_full"])
    imputation = PLACEHOLDER_IMPUTATION_SCALE * sweep["FitR2X"]
    common.plot_mean_sd(
        axs[0],
        sweep["Rank"],
        imputation,
        PLACEHOLDER_IMPUTATION_SD * imputation,
        style.COLORS["heldout"],
        "Masked imputation",
        marker="o",
    )
    axs[0].set_ylabel("$R^2X$")
    axs[0].set_ylim(bottom=0)  # room for the legend
    axs[0].legend(loc="lower right", fontsize=style.FONT_SIZE_DENSE, handlelength=1.5)
    common.plot_collinearity(sweep, axs[1], rank)


def makeFigure():
    return common.standalone_fixed(__name__)
