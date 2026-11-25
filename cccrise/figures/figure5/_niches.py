"""Figure 5's NICHES supplement (`figureS_comparator_single_cell`): CCC-RISE's
sender and receiver populations against upstream NICHES, from the
`cccrise-niches` tables. A population's fraction NICHES-positive is its share
of cells with a non-zero NICHES program score; what cell type predicts is
each population cell at the NICHES-positive rate of its fine type in its own
sample (`comparators.niches_programs`)."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ...comparators.niches_programs import SIDES, in_population
from ...runners.datasets import NICHES_DIR, scores_filename
from .. import common, style
from ..figure2._common import GROUP_TAU, plot_weight_scatter
from ._common import DATASET_LABELS

PROGRAM_COLORS = {
    1: style.CATEGORICAL[4],
    2: style.CATEGORICAL[2],
    3: style.COLORS["up"],
    4: style.COLORS["emphasis"],
    5: style.CATEGORICAL[5],
    6: style.CATEGORICAL[9],
}
SIDE_STYLES = {"sender": "-", "receiver": ":"}
#: `tau_sweep` draws a population only while it has this many cells
MIN_TAU_CELLS = 10
#: `weight_scatters`' ticks, on relative weights
SCATTER_TICKS = (0, 0.5, 1)
SCATTER_PAD = 0.05
#: `weight_scatters`' populations: a cell past both cutoffs is in both
POPULATION_COLORS = {
    "Sender": style.COLORS["sender"],
    "Receiver": style.COLORS["receiver"],
    "Both": style.COLORS["highlight"],
}
PREDICTED = "Cell type predicted"


def load(name: str, dataset: str) -> pd.DataFrame:
    return common.load_runner_output(
        scores_filename(name, dataset),
        NICHES_DIR,
        f"uv run cccrise-niches --dataset {dataset}",
    )


def tau_sweep(ax, dataset: str) -> None:
    """Each population's fraction NICHES-positive minus what cell type predicts
    across tau, `GROUP_TAU` marked; a population is drawn while it has
    `MIN_TAU_CELLS` cells."""
    table = load("tau", dataset).assign(
        Difference=lambda t: t["Observed"] - t[PREDICTED]
    )
    common.source_data(
        f"niches_positive_fraction_by_tau_{dataset}",
        table.rename(columns={"Observed": "Fraction NICHES+"}),
        "Each population at each tau (its cutoff as a fraction of that weight's "
        "largest value): its cells and samples, fraction NICHES-positive, what cell "
        "type predicts (each cell at its fine type's rate in its own sample) and the "
        f"difference; {GROUP_TAU} is the figure's tau. Points of fewer than "
        f"{MIN_TAU_CELLS} cells are not drawn.",
    )
    ax.axhline(
        0,
        color=style.COLORS["reference"],
        lw=style.LINE_WIDTH_THIN,
        ls=style.LINE_STYLE_REFERENCE,
    )
    tau_line = ax.axvline(
        GROUP_TAU,
        color=style.COLORS["muted"],
        lw=style.LINE_WIDTH_THIN,
        zorder=0,
        label=f"τ = {GROUP_TAU}",
    )
    components = sorted(table["Component"].unique())
    for k in components:
        for side in SIDES:
            group = table[
                (table["Component"] == k)
                & (table["Side"] == side)
                & (table["Cells"] >= MIN_TAU_CELLS)
            ]
            ax.plot(
                group["Tau"],
                group["Difference"],
                color=PROGRAM_COLORS[k],
                lw=style.LINE_WIDTH,
                ls=SIDE_STYLES[side],
            )
    ax.set_xlabel("Tau (τ)")
    ax.set_ylabel(
        "Δ fraction NICHES+\n(CCC-RISE − cell type)",
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.set_title(
        DATASET_LABELS[dataset], fontsize=style.FONT_SIZE_LABEL, fontweight="bold"
    )
    handles = (
        [
            Line2D([], [], color=PROGRAM_COLORS[k], lw=style.LINE_WIDTH, label=f"C{k}")
            for k in components
        ]
        + [
            Line2D(
                [],
                [],
                color=style.COLORS["ink"],
                lw=style.LINE_WIDTH,
                ls=ls,
                label=side.title(),
            )
            for side, ls in SIDE_STYLES.items()
        ]
        + [tau_line]
    )
    common.legend(ax, "right", handles=handles, fontsize=style.FONT_SIZE_DENSE)


def weight_scatters(axs, dataset: str) -> None:
    """Each component's per-cell sender vs. receiver weight, relative to each
    weight's largest value, its populations (`in_population`) colored and the
    `GROUP_TAU` cutoffs drawn, on limits shared by every component. Reads the
    runner's per-cell weights (`cells_<dataset>`), so the build never loads
    the cohort."""
    table = load("cells", dataset)
    senders, receivers = (
        weights / weights.max(axis=0)
        for weights in (
            table.filter(regex=rf"^{side} C\d+$").to_numpy()
            for side in ("Sender", "Receiver")
        )
    )
    lower = min(senders.min(), receivers.min()) - SCATTER_PAD
    cells = {}
    for k, ax in enumerate(axs, start=1):
        sender, receiver = senders[:, k - 1], receivers[:, k - 1]
        sends, receives = in_population(sender), in_population(receiver)
        groups = {
            "Sender": sends & ~receives,
            "Receiver": receives & ~sends,
            "Both": sends & receives,
        }
        plot_weight_scatter(ax, sender, receiver, groups, POPULATION_COLORS)
        for line in (ax.axvline, ax.axhline):
            line(
                GROUP_TAU,
                color=style.COLORS["reference"],
                ls=style.LINE_STYLE_REFERENCE,
                lw=style.LINE_WIDTH_THIN,
            )
        ax.set(xlim=(lower, 1 + SCATTER_PAD), ylim=(lower, 1 + SCATTER_PAD))
        ax.set_title(f"C{k}", fontsize=style.FONT_SIZE_LABEL, pad=2)
        ax.set_xticks(SCATTER_TICKS)
        ax.set_yticks(SCATTER_TICKS, labels=None if k == 1 else [])
        ax.set_xlabel("Sender", fontsize=style.FONT_SIZE_DENSE)
        group = np.select(list(groups.values()), [g.lower() for g in groups], "rest")
        cells |= {
            f"Sender C{k}": sender,
            f"Receiver C{k}": receiver,
            f"Group C{k}": group,
        }
    axs[0].set_ylabel(
        f"{DATASET_LABELS[dataset]} receiver", fontsize=style.FONT_SIZE_DENSE
    )
    axs[0].legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=2.5, mfc=c, mec=c, label=g.title())
            for g, c in POPULATION_COLORS.items()
        ],
        loc="upper left",
        fontsize=style.FONT_SIZE_DENSE,
        borderaxespad=0.2,
        handletextpad=0.1,
        labelspacing=0.2,
    )
    common.source_data(
        f"cell_weights_groups_{dataset}",
        pd.DataFrame({"Cell": table["Cell"], **cells}),
        "One row per cell: its sender and receiver weight on each CPD component, "
        "each relative to that weight's largest value, and its group: sender or "
        f"receiver if only that relative weight is at least {GROUP_TAU}, both if "
        "both are, rest otherwise.",
    )
