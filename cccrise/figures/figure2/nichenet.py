"""NicheNet downstream-target validation of CPD components, from
``uv run cccrise-nichenet``: per-component summaries under both receiver
definitions, and the AUROC of each component's top ligands."""

import numpy as np
from matplotlib.patches import Rectangle

from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common, style

DATASET = "alad"
RECEIVER_MODES = {"within_celltype": "Within cell type", "vs_rest": "Vs. rest"}
#: Receiver definition the ligand heatmap shows.
LIGAND_MODE = "within_celltype"
N_TOP_LIGANDS = 10


def plot(fig):
    scores = common.load_runner_output(
        DATASETS[DATASET].nichenet_scores_filename(), VALIDATION_DIR
    )
    summary = scores.groupby(["receiver_mode", "component"]).agg(
        mean_auroc=("auroc", "mean"),
        n_significant=("p_value", lambda p: int((p < common.SIG_ALPHA).sum())),
    )

    axs = fig.subplots(1, 3, width_ratios=[1, 1, 2.5])
    _summary_bars(summary["mean_auroc"], axs[0], "Mean ligand AUROC")
    common.reference_line(axs[0], 0.5)
    axs[0].set_ylim(0, 1)
    axs[0].legend(title="Receivers")
    _summary_bars(
        summary["n_significant"], axs[1], f"Ligands with p < {common.SIG_ALPHA}"
    )
    _ligand_heatmap(scores[scores["receiver_mode"] == LIGAND_MODE], axs[2])


def _summary_bars(values, ax, ylabel):
    """One bar per component and receiver mode."""
    table = values.unstack("receiver_mode")[list(RECEIVER_MODES)]
    width = 0.8 / len(RECEIVER_MODES)
    x = np.arange(len(table))
    for i, (mode, label) in enumerate(RECEIVER_MODES.items()):
        offset = (i - (len(RECEIVER_MODES) - 1) / 2) * width
        ax.bar(x + offset, table[mode], width, color=style.COLORS[mode], label=label)
    ax.set_xticks(x, table.index)
    ax.set_xlabel("CP component")
    ax.set_ylabel(ylabel)


def _ligand_heatmap(scores, ax):
    """AUROC of each component's top-loading ligands (columns: components,
    rows: loading rank). A ligand's AUROC does not depend on its receptor, so
    each ligand appears once. Significant ligands are bold with a star;
    ligands absent from the NicheNet database (NaN AUROC) are grey."""
    top = scores.sort_values("loading", key=np.abs, ascending=False)
    top = top.drop_duplicates(["component", "ligand"])
    top = top.groupby("component").head(N_TOP_LIGANDS).copy()
    top["rank"] = top.groupby("component").cumcount()
    auroc = top.pivot(index="rank", columns="component", values="auroc")

    limit = np.nanmax(np.abs(auroc.to_numpy() - 0.5))
    image = ax.imshow(
        auroc.to_numpy(),
        cmap=style.CMAPS["diverging"],
        vmin=0.5 - limit,
        vmax=0.5 + limit,
        aspect="auto",
    )
    columns = {component: i for i, component in enumerate(auroc.columns)}
    for component, rank, ligand, value, p_value in zip(
        top["component"],
        top["rank"],
        top["ligand"],
        top["auroc"],
        top["p_value"],
        strict=True,
    ):
        x = columns[component]
        if np.isnan(value):
            ax.add_patch(
                Rectangle(
                    (x - 0.5, rank - 0.5), 1, 1, color=style.COLORS["not_significant"]
                )
            )
        significant = p_value < common.SIG_ALPHA
        ax.text(
            x,
            rank,
            ligand + ("*" if significant else ""),
            ha="center",
            va="center",
            # Complex ligands (e.g. ITGA4&ITGB7) would overflow their cell.
            fontsize=style.FONT_SIZE_DENSE - (len(ligand) > 9),
            fontweight="bold" if significant else "normal",
        )
    ax.set_xticks(range(auroc.shape[1]), auroc.columns)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("CP component")
    ax.set_ylabel(f"Top {N_TOP_LIGANDS} ligands by loading")
    ax.set_title(
        f"Top ligands ({RECEIVER_MODES[LIGAND_MODE].lower()}; "
        f"* p < {common.SIG_ALPHA})\ngrey: not in NicheNet"
    )
    ax.figure.colorbar(image, ax=ax, label="Ligand AUROC", shrink=0.8)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
