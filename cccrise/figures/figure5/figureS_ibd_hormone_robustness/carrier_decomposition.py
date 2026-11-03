"""Per biopsy, the hormone programs' carrier cells split into how many there
are (left: among the biopsy's at most 2,000 cells) and how strongly each
carries the program (right: mean unscaled per-cell weight, biopsies with at
least `MIN_CELLS` carriers), by inflammation status (Kruskal-Wallis p):
inflammation removes carriers rather than lowering their weight."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from ... import common, style
from .. import _common as S

AXES = ("rect", "rect")
MIN_CELLS = 3
#: (label, component, role, carrier fine type)
SLOTS = [
    ("C3/C4 EE\nsender", 3, "sender", "Enteroendocrine cells"),
    ("C3 EC\nreceiver", 3, "receiver", "Enterochromaffin cells"),
    ("C4 tuft\nreceiver", 4, "receiver", "Tuft cells"),
]


def per_biopsy() -> pd.DataFrame:
    cells = S.programs("enteroendocrine_cells")
    biopsies = S.samples()[S.SPEC.condition_column].astype(str)
    status = S.samples()[S.GROUP_COLUMN].to_numpy()
    rows = []
    for label, k, role, carrier in SLOTS:
        per = (
            cells[cells["Cell type"] == carrier]
            .groupby("Biopsy")[f"C{k} {role} weight"]
            .agg(["size", "mean"])
        )
        per = per.reindex(biopsies).fillna({"size": 0})
        rows.append(
            pd.DataFrame(
                {
                    "Carrier": label.replace("\n", " "),
                    "Biopsy": biopsies,
                    "Status": status,
                    "Carrier cells": per["size"].to_numpy().astype(int),
                    "Mean weight": per["mean"]
                    .where(per["size"] >= MIN_CELLS)
                    .to_numpy(),
                }
            )
        )
    return pd.concat(rows)


def plot(axs):
    table = per_biopsy()
    common.source_data(
        "hormone_carrier_decomposition",
        table,
        f"Per biopsy: carrier cells of each hormone program side, and their mean "
        f"unscaled per-cell weight (biopsies with at least {MIN_CELLS} carriers); "
        "Kruskal-Wallis p across statuses above each.",
    )
    rng = np.random.default_rng(0)
    for ax, column in zip(axs, ("Carrier cells", "Mean weight"), strict=True):
        for x, (label, *_) in enumerate(SLOTS):
            rows = table[table["Carrier"] == label.replace("\n", " ")].dropna(
                subset=[column]
            )
            groups = [rows.loc[rows["Status"] == g, column] for g in S.GROUP_LABELS]
            S.draw_boxes(ax, x, groups, rng)
            p = kruskal(*groups).pvalue
            ax.text(
                x,
                1,
                f"p = {p:.1g}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=style.FONT_SIZE_DENSE,
                fontweight="bold" if p < common.SIG_ALPHA else "normal",
            )
        ax.set_xticks(
            range(len(SLOTS)), [s[0] for s in SLOTS], fontsize=style.FONT_SIZE_DENSE
        )
    axs[0].set_ylabel("Carrier cells")
    axs[0].set_yscale("symlog", linthresh=1)
    axs[0].set_ylim(bottom=0)
    axs[1].set_ylabel("Mean weight per carrier cell")
    S.status_legend(axs[1])


def makeFigure():
    return common.standalone_fixed(__name__)
