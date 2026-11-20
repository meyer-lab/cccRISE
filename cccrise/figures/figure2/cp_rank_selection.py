"""CPD rank selection, ALAD cohort: full-tensor fit and k-fold imputation R2X
at each rank; the largest absolute cosine between the LR pair loadings of two
components, per rank and seeded SVD start; and the number of seeded SVD starts
reaching each distinct minimum at each rank."""

from .. import common, style

DATASET = "alad"
AXES = ("square", "square", "square")


def plot(axs):
    sweep = common.production_fits(common.load_cpd_rank(DATASET))
    repeats = common.load_cpd_imputation(DATASET)
    rank = common.selected_ranks(DATASET)[1]
    common.plot_cpd_fit_r2x(
        sweep, axs[0], rank, color=style.COLORS["fit_full"], marker=None
    )
    common.plot_cpd_imputation_r2x(repeats, axs[0])
    common.source_data(
        "cpd_fit_r2x_by_rank",
        sweep[["Rank", "FitR2X"]],
        "R2X of the full-tensor CPD fit (the production start) at each CPD rank.",
        part=0,
    )
    common.source_data(
        "cpd_imputation_r2x_by_rank",
        repeats,
        "K-fold imputation R2X of each CPD rank, one row per repeat of the "
        "fold split (with its seed); plotted as mean +/- SD over repeats.",
        part=0,
    )
    # A narrow key fits right of the low-collinearity ranks
    common.plot_collinearity_boxes(
        axs[1], DATASET, part=1, wrap_key=True, key_loc="center right"
    )
    common.mark_rank(axs[1], rank)
    common.plot_minima(axs[2], DATASET, rank, part=2)


def makeFigure():
    return common.standalone_fixed(__name__)
