"""CP rank selection: fit and imputation R2X, and component collinearity, from
``uv run cccrise-cpdrank`` (on the canonical RISE fit's tensor)."""

from .. import common, style

DATASET = "alad"
AXES = ("square", "square")


def plot(axs):
    sweep = common.per_rank(common.load_cpd_rank(DATASET))
    rank = common.selected_ranks(DATASET)[1]
    common.plot_cpd_fit_r2x(
        sweep, axs[0], rank, color=style.COLORS["fit_full"], marker=None
    )
    common.plot_cpd_imputation_r2x(common.load_cpd_imputation(DATASET), axs[0])
    common.plot_collinearity(sweep, axs[1], rank)


def makeFigure():
    return common.standalone_fixed(__name__)
