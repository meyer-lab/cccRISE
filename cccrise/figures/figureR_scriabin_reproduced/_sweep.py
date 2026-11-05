"""Extended Data Fig. 8f's four panels for one subsample size: each point
one random inDrop subsample at one R^2 threshold, jittered as the paper's;
Pearson R against R^2, ours and (grey) the paper's."""

import pandas as pd
from matplotlib.lines import Line2D

from .. import common, style
from . import _common as C

AXES = ("square",) * 4
#: Each panel's column, y label, jitter half-width and the paper's R
PANELS = {
    "Power": ("Soft power", 0.25, 0.78),
    "PercentNonSignificant": ("Non-significantly\ncorrelated programs (%)", 0.3, -0.3),
    "MeanPairsPerProgram": ("Average program size\n(LR pairs)", 0.0, -0.6),
    "PercentSingleLigandOrReceptor": (
        "Programs of only\n1 ligand or receptor (%)",
        0.3,
        0.22,
    ),
}
X_JITTER = 0.015
#: Points where no soft power reaches the threshold
FALLBACK = style.COLORS["emphasis"]


def _source_data(rows: pd.DataFrame, cells: int) -> None:
    frame = pd.DataFrame(
        {
            "Cells": rows.Cells.astype("Int64"),
            "Subsample": rows.Subsample.astype("Int64"),
            "R2 threshold": rows.R2,
            "Recommended soft power": rows.RecommendedPower,
            "Soft power used": rows.Power.astype("Int64"),
            "Programs": rows.Programs.astype("Int64"),
            "Non-significant programs (%)": rows.PercentNonSignificant,
            "Average program size (LR pairs)": rows.MeanPairsPerProgram,
            "Programs of only 1 ligand or receptor (%)": rows.PercentSingleLigandOrReceptor,
        }
    )
    paper = ", ".join(
        f"{label.split(chr(10))[0]} {r}" for label, _, r in PANELS.values()
    )
    common.source_data(
        f"soft_power_sweep_{cells}_cells",
        frame,
        f"Extended Data Fig. 8f of Wilk et al. 2024 rerun: upstream InteractionPrograms on "
        f"random subsamples of {cells} inDrop panc8 cells (SCTransform + ALRA; OmniPath), each "
        "cut at the soft power WGCNA recommends for each R2 threshold (1 where none reaches "
        "it; uncapped, as the paper's panel). Non-significant: upstream "
        "InteractionProgramSignificance, more than 5% of 10,000 random LR sets of the "
        "program's size with one-sided Wilcoxon p > 0.05. Program size: LR pairs per program. "
        "Recommended soft power empty: no power of WGCNA's default 1-20 reaches the threshold, "
        "so the power used is 1 (drawn in blue). "
        "Points are jittered in the figure, not here; R is Pearson's on these values. The "
        f"paper's R (its panels): {paper}.",
    )


def _scatter(ax, rows: pd.DataFrame, column: str, y_jitter: float, seed: int) -> None:
    """Jittered points, muted where no soft power reaches the threshold (and
    upstream falls back to power 1)."""
    x = C.jitter(rows.R2, X_JITTER, seed=seed)
    y = C.jitter(rows[column], y_jitter, seed=seed + 1)
    fallback = rows.RecommendedPower.isna().to_numpy()
    for mask, color in ((~fallback, C.INK), (fallback, FALLBACK)):
        ax.scatter(
            x[mask],
            y[mask],
            s=style.POINT_SIZE_BACKGROUND,
            color=color,
            alpha=0.6,
            lw=0,
        )


def _stats(ax, rows: pd.DataFrame, column: str, paper_r: float) -> None:
    lines = (
        (C.pearson_label(rows.R2, rows[column]), C.INK),
        (f"Paper: R = {paper_r}", C.PAPER),
    )
    for i, (text, color) in enumerate(lines):
        ax.text(
            0.03,
            0.97 - 0.09 * i,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=color,
            fontsize=style.FONT_SIZE_DENSE,
        )


def draw(axs, cells: int) -> None:
    table = C.load("soft_power")
    rows = table[table.Cells == cells]
    for k, (ax, (column, (label, y_jitter, paper_r))) in enumerate(
        zip(axs, PANELS.items(), strict=True)
    ):
        _scatter(ax, rows, column, y_jitter, seed=2 * k)
        _stats(ax, rows, column, paper_r)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.22)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("Minimum scale-free\ntopology fit (R²)")
        ax.set_ylabel(label)
    # The row's title spans its four axes, so it is left out of the first's
    # measured margins (the legend above the last reserves the row's top)
    title = axs[0].text(
        0,
        1.04,
        f"ED Fig. 8f · {cells:,} inDrop cells",
        transform=axs[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=style.FONT_SIZE_LABEL,
    )
    title.set_in_layout(False)
    axs[-1].legend(
        handles=[
            Line2D([], [], color=c, marker="o", ls="", ms=style.MARKER_SIZE, label=t)
            for c, t in ((C.INK, "Recommended power"), (FALLBACK, "None reaches R²: 1"))
        ],
        loc="upper right",
        bbox_to_anchor=(1, 1.16),
        fontsize=style.FONT_SIZE_DENSE,
        handletextpad=0.1,
        borderaxespad=0,
        frameon=False,
    )
    _source_data(rows, cells)
