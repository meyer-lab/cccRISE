"""Expression of each IBD CP component's top LR pair in its carrier cells,
per biopsy and inflammation status: the ligand in the component's sender
carrier and the receptor in its receiver carrier (the fine cell types of
largest mean sender / receiver weight, `S.celltype_weights`). A complex
counts at its least-expressed subunit. Each value is a biopsy's mean
normalized expression over its cells of that type; biopsies without such
cells are left out. Above each gene: Kruskal-Wallis q across the three
groups, Benjamini-Hochberg corrected across every gene shown."""

import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy.stats import kruskal

from ...anova import adjust_pvalues
from .. import common, style
from . import _common as S
from .group_loadings import DODGE, JITTER, draw_boxes

ROLE_SIDE = {"sender": 0, "receiver": 1}  # LR pair side each role expresses
SLOT_GAP = 0.4  # extra space between components
#: Carrier cell types as the component labels abbreviate them
SHORT_NAMES = {
    "Monocytes SOD2+": "Mono SOD2+",
    "Mast cells": "Mast",
    "ILCs": "ILC",
    "Enteroendocrine cells": "EE",
    "Enterochromaffin cells": "EC",
    "Tuft cells": "Tuft",
}
AXES = ("full",)


def top_pairs() -> list[dict]:
    """Per component: its LR pair of largest |loading|, and per role its
    carrier cell type and the pair's genes on that role's side."""
    D = S.factors()["cp_D"]
    weights = S.celltype_weights()
    fine = weights[weights["level"] == "fine"]
    carrier = (
        fine.sort_values("mean_weight", ascending=False)
        .drop_duplicates(["role", "component"])
        .set_index(["role", "component"])["celltype"]
    )
    pairs = []
    for k in sorted(fine["component"].unique()):
        pair = str(D["lr_pair"].values[np.argmax(np.abs(D.values[:, k - 1]))])
        sides = pair.split("^")
        pairs.append(
            {
                "component": k,
                "pair": pair.replace("^", "–"),
                "carrier": {role: carrier[role, k] for role in ROLE_SIDE},
                "genes": {r: sides[i].split("&") for r, i in ROLE_SIDE.items()},
            }
        )
    return pairs


def expression_by_biopsy(pair: dict, role: str) -> pd.DataFrame:
    """One row per biopsy with cells of `pair`'s `role` carrier: its group,
    cell count of that type, and expression of the role's genes there
    (least-expressed subunit)."""
    table = S.carrier_expression()
    celltype, genes = pair["carrier"][role], pair["genes"][role]
    cells = table[(table["celltype"] == celltype) & table["gene"].isin(genes)]
    expression = cells.groupby("condition").agg(
        Expression=("mean_expression", "min"), Cells=("n_cells", "first")
    )
    group = S.samples().set_index(S.SPEC.condition_column)[S.GROUP_COLUMN]
    return (
        expression.rename_axis("Biopsy")
        .reset_index()
        .assign(
            Component=pair["component"],
            Role=role,
            Gene="&".join(genes),
            CellType=celltype,
            Group=lambda d: d["Biopsy"].map(group).to_numpy(),
        )
    )


def plot(axs):
    (ax,) = axs
    pairs = top_pairs()
    # Ligand then receptor, component by component
    slots = [expression_by_biopsy(p, role) for p in pairs for role in ROLE_SIDE]
    q = adjust_pvalues(
        [
            kruskal(
                *(s.loc[s["Group"] == g, "Expression"] for g in S.GROUP_LABELS)
            ).pvalue
            for s in slots
        ]
    )
    _source_data(pairs, slots, q)
    ticks = [i + SLOT_GAP * (s["Component"].iloc[0] - 1) for i, s in enumerate(slots)]
    _draw_slots(ax, ticks, slots, q)
    _label_components(ax, ticks, pairs)
    ax.set_xticks(ticks, [s["Gene"].iloc[0] for s in slots])
    ax.tick_params(axis="x", labelsize=style.FONT_SIZE_DENSE)
    ax.set_xlim(ticks[0] - 1.5 * DODGE - JITTER, ticks[-1] + 1.5 * DODGE + JITTER)
    ax.set_ylabel("Mean expression\nin carrier cells")


def _draw_slots(ax, ticks, slots: list[pd.DataFrame], q) -> None:
    """Each gene's boxes at its tick, with its q above."""
    rng = np.random.default_rng(0)
    top = blended_transform_factory(ax.transData, ax.transAxes)
    for x, slot, q_value in zip(ticks, slots, q, strict=True):
        values = [slot.loc[slot["Group"] == g, "Expression"] for g in S.GROUP_LABELS]
        draw_boxes(ax, x, values, rng)
        ax.text(
            x,
            1,
            f"q = {q_value:.1g}",
            transform=top,
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if q_value < common.SIG_ALPHA else "normal",
        )


def _label_components(ax, ticks, pairs: list[dict]) -> None:
    """Each component and its carriers, centered under its genes."""
    top = blended_transform_factory(ax.transData, ax.transAxes)
    for i, pair in enumerate(pairs):
        # One name for a cell type signalling to itself
        carriers = " → ".join(
            dict.fromkeys(SHORT_NAMES.get(t, t) for t in pair["carrier"].values())
        )
        center = np.mean(ticks[len(ROLE_SIDE) * i : len(ROLE_SIDE) * (i + 1)])
        ax.text(
            center,
            -0.2,
            f"C{pair['component']}: {carriers}",
            transform=top,
            ha="center",
            va="top",
        )


def _source_data(pairs: list[dict], slots: list[pd.DataFrame], q) -> None:
    common.source_data(
        "carrier_pairs",
        pd.DataFrame(
            {
                "Component": p["component"],
                "LR pair": p["pair"],
                "Sender carrier": p["carrier"]["sender"],
                "Receiver carrier": p["carrier"]["receiver"],
            }
            for p in pairs
        ),
        "Each CP component's LR pair of largest |loading| and its sender and "
        "receiver carriers: the fine cell types of largest mean per-cell "
        "sender and receiver weight.",
    )
    common.source_data(
        "carrier_expression_by_biopsy",
        pd.concat(slots, ignore_index=True).rename(columns={"CellType": "Cell type"}),
        "Each biopsy's mean normalized expression of the ligand (in the sender "
        "carrier) or receptor (in the receiver carrier) of each component's top "
        "LR pair, over its cells of that type (a complex at its least-expressed "
        "subunit); Cells counts them.",
    )
    common.source_data(
        "carrier_expression_kruskal",
        pd.DataFrame(
            {
                "Component": s["Component"].iloc[0],
                "Role": s["Role"].iloc[0],
                "Gene": s["Gene"].iloc[0],
                "Kruskal-Wallis q": value,
            }
            for s, value in zip(slots, q, strict=True)
        ),
        "Kruskal-Wallis test of each gene's per-biopsy expression across "
        "inflammation status, Benjamini-Hochberg corrected across the genes shown.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
