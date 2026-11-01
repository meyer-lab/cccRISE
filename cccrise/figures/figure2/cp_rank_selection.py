"""CP rank selection: fit and imputation R2X from ``uv run cccrise-cpdrank``
(on the canonical RISE fit's tensor), and each mode's collinearity across ten
seeded SVD starts per rank."""

from .. import common, cp_rank, style

DATASET = "alad"
AXES = ("square", "square")


def plot(axs):
    sweep = common.production_fits(common.load_cpd_rank(DATASET))
    repeats = common.load_cpd_imputation(DATASET)
    rank = common.selected_ranks(DATASET)[1]
    common.plot_cpd_fit_r2x(
        sweep, axs[0], rank, color=style.COLORS["fit_full"], marker=None
    )
    common.plot_cpd_imputation_r2x(repeats, axs[0])
    common.source_data(
        "cp_fit_r2x_by_rank",
        sweep[["Rank", "FitR2X"]],
        "R2X of the full-tensor CP fit (the production start) at each CP rank.",
        part=0,
    )
    common.source_data(
        "cp_imputation_r2x_by_rank",
        repeats,
        "K-fold imputation R2X of each CP rank, one row per repeat of the "
        "fold split (with its seed); plotted as mean +/- SD over repeats.",
        part=0,
    )
    cp_rank.plot_mode_collinearity(axs[1], common.load_cpd_rank(DATASET), rank, part=1)


def makeFigure():
    return common.standalone_fixed(__name__)
