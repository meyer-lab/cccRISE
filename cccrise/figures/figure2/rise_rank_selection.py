"""RISE rank selection by bi-cross-validation, from ``uv run cccrise-riserank``."""

from matplotlib.lines import Line2D

from ...runners.datasets import DATASETS, RANK_SELECTION_DIR
from .. import common

DATASET = "alad"
AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    sweep = common.load_runner_output(
        DATASETS[DATASET].rise_rank_filename(), RANK_SELECTION_DIR
    )
    common.plot_rise_rank(sweep, ax, rank=common.selected_ranks(DATASET)[0])
    common.source_data(
        "bicv_r2x_by_rise_rank",
        sweep,
        "RISE R2X per rank: the full-data fit ('Fit R2X' rows), and each "
        "bi-cross-validation repeat ('BiCV R2X' rows; R2X of the held-out block, "
        "and of the training block in 'Train Block R2X', with the cells and "
        "genes in each and the repeat's seed). Plotted as mean +/- SD over repeats.",
    )
    # The dashed rank line's key, shared by panels A-C
    ax.add_artist(ax.get_legend())
    ax.legend(
        handles=[Line2D([], [], label="Selected rank", **common.RANK_LINE)],
        loc="lower right",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
