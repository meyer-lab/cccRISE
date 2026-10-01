"""NicheNet downstream-target validation of CPD components, from
``uv run cccrise-nichenet``: per-component summaries under both receiver
definitions, and each component's top ligands' AUROCs."""

from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common, style

DATASET = "alad"
#: Receiver definition the ligand heatmap shows.
LIGAND_MODE = "within_celltype"


def plot(fig):
    scores = common.load_runner_output(
        DATASETS[DATASET].nichenet_scores_filename(), VALIDATION_DIR
    )
    summary = common.nichenet_summary(scores)

    axs = fig.subplots(1, 3, width_ratios=[1, 1, 2.5])
    common.plot_nichenet_summary(summary, axs[0], "mean_auroc", "Mean ligand AUROC")
    common.plot_nichenet_summary(
        summary, axs[1], "n_significant", f"Ligands with p < {common.SIG_ALPHA}"
    )
    axs[1].get_legend().remove()
    common.plot_nichenet_heatmap(scores[scores["receiver_mode"] == LIGAND_MODE], axs[2])
    axs[2].set_title(
        f"Top ligands ({common.NICHENET_MODES[LIGAND_MODE].lower()}; "
        f"* p < {common.SIG_ALPHA})\ngrey: not in NicheNet"
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
