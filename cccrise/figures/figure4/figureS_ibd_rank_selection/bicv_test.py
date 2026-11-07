"""IBD BiCV test R2X alone, on an axis zoomed to it: on the shared axis of
the previous panel it runs flat below the fit and BiCV train, from
``uv run cccrise-riserank``."""

from ....runners.datasets import DATASETS, RANK_SELECTION_DIR
from ... import common, style

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    ax = axs[0]
    sweep = common.load_runner_output(
        DATASETS[DATASET].rise_rank_filename(), RANK_SELECTION_DIR
    )
    bicv = sweep[sweep["Metric"] == "BiCV R2X"]
    common.source_data(
        "bicv_test_r2x_by_rise_rank",
        bicv,
        "Held-out (BiCV test) R2X of each bi-cross-validation repeat per RISE "
        "rank, with the cells and genes in each block and the repeat's seed. "
        "Plotted as mean +/- SD over repeats.",
    )
    by_rank = bicv.groupby("Rank")["R2X"]
    mean = by_rank.mean()
    common.plot_mean_sd(
        ax,
        mean.index,
        mean,
        by_rank.std().fillna(0),
        style.COLORS["heldout"],
        "BiCV test",
    )
    common.mark_rank(ax, common.selected_ranks(DATASET)[0])
    ax.set_xlabel("RISE rank")
    ax.set_ylabel("BiCV test R²X")


def makeFigure():
    return common.standalone_fixed(__name__)
