"""R2X of the full-data RISE fit and of the bi-cross-validation training and
held-out blocks at each RISE rank, ALAD cohort."""

from matplotlib.lines import Line2D

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from .. import common, style

DATASET = "alad"
AXES = ("square",)
KEY_HEADROOM = 0.28  # fraction of the axes above the curves, for the key


def plot(axs):
    draw(axs[0], DATASET)


def draw(ax, dataset: str) -> None:
    """`dataset`'s BiCV sweep from R2X 0, its selected RISE rank marked, keyed
    in two columns above the curves."""
    sweep = common.load_runner_output(
        DATASETS[dataset].rise_rank_filename(), RANK_SELECTION_DIR
    )
    lines = common.plot_rise_rank(sweep, ax)
    # The rank line stops below the key
    common.mark_rank(ax, common.selected_ranks(dataset)[0], top=1 - KEY_HEADROOM)
    save_sweep(sweep)
    ax.set_ylim(0, ax.dataLim.y1 / (1 - KEY_HEADROOM))
    # The dashed rank line's key, shared by panels A-C
    rank_key = Line2D([], [], label="Selected rank", **common.RANK_LINE)
    ax.legend(
        handles=[*lines, rank_key],
        loc="upper left",
        ncols=2,
        fontsize=style.FONT_SIZE_DENSE,
        columnspacing=0.8,
    )


def save_sweep(sweep) -> None:
    common.source_data(
        "bicv_r2x_by_rise_rank",
        sweep,
        "RISE R2X per rank: the full-data fit ('Fit R2X' rows), and each "
        "bi-cross-validation repeat ('BiCV R2X' rows; R2X of the held-out block, "
        "and of the training block in 'Train Block R2X', with the cells and "
        "genes in each and the repeat's seed). Plotted as mean +/- SD over repeats.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
