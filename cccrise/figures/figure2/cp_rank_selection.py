"""CP rank selection: fit and imputation R2X, and component collinearity, from
``uv run cccrise-cpdrank`` (on the canonical RISE fit's tensor)."""

from .. import common, style

DATASET = "alad"
AXES = ("square", "square")
COLLINEARITY_COLUMNS = [
    "MaxModesCollinear",
    "WorstPairCanceling",
    "WorstPairI",
    "WorstPairJ",
]


def plot(axs):
    sweep = common.per_rank(common.load_cpd_rank(DATASET))
    repeats = common.load_cpd_imputation(DATASET)
    rank = common.selected_ranks(DATASET)[1]
    common.plot_cpd_fit_r2x(
        sweep, axs[0], rank, color=style.COLORS["fit_full"], marker=None
    )
    common.plot_cpd_imputation_r2x(repeats, axs[0])
    common.plot_collinearity(sweep, axs[1], rank)
    common.source_data(
        "cp_fit_r2x_by_rank",
        sweep[["Rank", "FitR2X"]],
        "R2X of the full-tensor CP fit at each CP rank.",
        part=0,
    )
    common.source_data(
        "cp_imputation_r2x_by_rank",
        repeats,
        "K-fold imputation R2X of each CP rank, one row per repeat of the "
        "fold split (with its seed); plotted as mean +/- SD over repeats.",
        part=0,
    )
    common.source_data(
        "cp_collinearity_by_rank",
        sweep[["Rank", *COLLINEARITY_COLUMNS]],
        "Per CP rank, the number of modes the most collinear component pair "
        "(components WorstPairI and WorstPairJ; -1 if none) is collinear in, "
        "and whether that pair cancels (opposing signs) rather than duplicates "
        "itself (empty when no pair is collinear).",
        part=1,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
