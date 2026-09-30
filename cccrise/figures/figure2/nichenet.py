"""NicheNet downstream-target validation of CPD components, from
``uv run cccrise-nichenet``: per-component summaries under both receiver
definitions, and the per-ligand AUROCs of the best-supported component."""

from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common, style

DATASET = "alad"
#: Receiver definition the per-ligand panel shows.
LIGAND_MODE = "within_celltype"


def plot(fig):
    scores = common.load_runner_output(
        DATASETS[DATASET].nichenet_scores_filename(), VALIDATION_DIR
    )
    summary = common.nichenet_summary(scores)

    axs = fig.subplots(1, 3, width_ratios=[1, 1, 1.2])
    common.plot_nichenet_summary(summary, axs[0], "mean_auroc", "Mean ligand AUROC")
    common.plot_nichenet_summary(
        summary, axs[1], "n_significant", f"Ligands with p < {common.SIG_ALPHA}"
    )
    axs[1].get_legend().remove()

    mode_scores = scores[scores["receiver_mode"] == LIGAND_MODE]
    best = summary.loc[LIGAND_MODE, "n_significant"].idxmax()
    common.plot_nichenet_ligands(mode_scores, best, axs[2])


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["two_thirds"])
