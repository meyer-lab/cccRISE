"""Figure 5's flows: what STAC reports for a program beside what cccRISE resolves.
Each bar is one cell type and each of its parts that fraction of the type's cells
(sender and receiver populations by Figure 2's and Figure 4's cutoffs); STAC's
bands are drawn at full width, since it has one value per label."""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path

from ...runners.datasets import DATASETS
from .. import common, style
from ..figure2._common import (
    GRANULARITY_DATASET,
    GROUP_TAU,
    component_weights,
    split_cells,
)
from ..figure4 import _common as F4

#: Figure 2's epithelial cells.
EPITHELIAL = style.CELL_TYPE_COLORS[GRANULARITY_DATASET]["Epithelial cells"]
BELOW = "0.82"
BAR = "0.45"
BAND_ALPHA = 0.55
NODE_WIDTH = 1.6
TEXT_SIZE = style.FONT_SIZE_DENSE - 0.7
STATES = ("Villus-like", "Mixed", "Crypt-like")
#: Bands of `flow_ibd`, on axes of (0, 81) x (0, 52).
TOP, SPAN, STATE_GAP, RECEIVER_GAP = 47.0, 40.6, 0.8, 2.5
#: Length of `flow_alad`'s bars and where its horseshoes start.
LENGTH, BAR_X = 15.0, 12


def _node(ax, x: float, y0: float, y1: float, color) -> None:
    ax.add_patch(
        Rectangle(
            (x - NODE_WIDTH / 2, y0), NODE_WIDTH, y1 - y0, fc=color, ec="none", zorder=4
        )
    )


def _band(ax, x0, left: tuple, x1, right: tuple, color) -> None:
    """A band from the span `left` at x0 to the span `right` at x1."""
    (a0, a1), (b0, b1), m = left, right, (x0 + x1) / 2
    verts = [
        (x0, a0),
        (m, a0),
        (m, b0),
        (x1, b0),
        (x1, b1),
        (m, b1),
        (m, a1),
        (x0, a1),
        (x0, a0),
    ]
    codes = [
        Path.MOVETO,
        *[Path.CURVE4] * 3,
        Path.LINETO,
        *[Path.CURVE4] * 3,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(Path(verts, codes), fc=color, ec="none", alpha=BAND_ALPHA, zorder=2)
    )


def _note(ax, x: float, y: float, text: str, **kwargs) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=TEXT_SIZE,
        va="center",
        zorder=6,
        **{"ha": "center"} | kwargs,
    )


def _title(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=style.FONT_SIZE_LABEL,
        fontweight="bold",
        ha="center",
        va="center",
    )


def _horseshoe(ax, y0: float, y1: float, width: float) -> None:
    bow = 0.32 * (y0 - y1)
    arch = Path(
        [(BAR_X, y0), (BAR_X + 24, y0 + bow), (BAR_X + 24, y1 - bow), (BAR_X, y1)],
        [Path.MOVETO, *[Path.CURVE4] * 3],
    )
    ax.add_patch(
        PathPatch(
            arch,
            fc="none",
            ec=EPITHELIAL,
            lw=width,
            alpha=BAND_ALPHA,
            capstyle="butt",
            zorder=2,
        )
    )


def _flow_axes(ax, width: float) -> None:
    ax.set(xlim=(0, width), ylim=(0, 52))
    ax.set_axis_off()


def _legend(ax, handles, ncol: int) -> None:
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        fontsize=TEXT_SIZE,
        columnspacing=0.8,
        handlelength=0.9,
        handletextpad=0.3,
        borderaxespad=0,
    )


#: ALAD's one fine epithelial label as annotated, and the name the figure shows.
EPITHELIAL_LABEL = "Lung epithelium"
EPITHELIAL_NAME = "Epithelial cells"


def _epithelial_populations() -> pd.Series:
    """The epithelial label's cells, and its C5 sender and receiver cells."""
    cells = common.get_cell_projections(GRANULARITY_DATASET)
    label = DATASETS[GRANULARITY_DATASET].labellings["fine"]
    epithelium = cells.obs[label].astype(str).to_numpy() == EPITHELIAL_LABEL
    groups = split_cells(*component_weights())
    return pd.Series(
        {
            EPITHELIAL_NAME: int(epithelium.sum()),
            "Ciliated-like sender": int((groups["sender"] & epithelium).sum()),
            "Basal/secretory-like receiver": int(
                (groups["receiver"] & epithelium).sum()
            ),
        }
    )


def flow_alad(ax) -> None:
    """BTC-EGFR on epithelial cells: all of the label, then its two end populations."""
    t = _epithelial_populations()
    common.source_data(
        "c5_epithelial_cell_populations",
        t.rename("Cells").rename_axis("Population").reset_index(),
        "Epithelial cells (the one fine epithelial label), and those past Figure 2's "
        "sender and receiver cutoffs.",
    )
    sender = t["Ciliated-like sender"] / t[EPITHELIAL_NAME] * LENGTH
    receiver = t["Basal/secretory-like receiver"] / t[EPITHELIAL_NAME] * LENGTH
    _flow_axes(ax, 40)
    _title(ax, 20, 50.5, "STAC S10")
    _node(ax, BAR_X, 31, 46, EPITHELIAL)
    _note(ax, BAR_X - 2.8, 38.5, EPITHELIAL_NAME, rotation=90)
    _horseshoe(ax, 43.5, 33.5, 5)
    _title(ax, 20, 27, "CCC-RISE C5")
    top = 21.0
    bottom = top - LENGTH
    _node(ax, BAR_X, top - sender, top, style.COLORS["sender"])
    _node(ax, BAR_X, bottom + receiver, top - sender, BELOW)
    _node(ax, BAR_X, bottom, bottom + receiver, style.COLORS["receiver"])
    _note(ax, BAR_X - 2.8, top - LENGTH / 2, EPITHELIAL_NAME, rotation=90)
    _horseshoe(ax, top - sender / 2, bottom + receiver / 2, 7)
    _note(ax, BAR_X - 0.8, top + 1.6, "Ciliated-like", ha="left")
    _note(ax, BAR_X - 0.8, bottom - 1.6, "Basal/secretory-like", ha="left")
    _legend(
        ax,
        [
            Patch(color=EPITHELIAL, alpha=BAND_ALPHA, label="BTC–EGFR"),
            Patch(color=BELOW, label="Below cutoff"),
        ],
        2,
    )


def _ee_states() -> pd.DataFrame:
    """Per Figure 4 state: its EE cells, and those past the C3 and C4 sender
    cutoffs alone, both, or neither."""
    cells = F4.programs("enteroendocrine_cells")
    ee = cells[cells["Cell type"] == "Enteroendocrine cells"]
    relative = F4.hormone_angles(ee)
    pyy = relative["C3 relative"] >= GROUP_TAU
    gcg = relative["C4 relative"] >= GROUP_TAU
    t = pd.DataFrame(
        {
            "State": F4.hormone_extremes(ee).to_numpy(),
            "PYY only": (pyy & ~gcg).to_numpy(),
            "Both": (pyy & gcg).to_numpy(),
            "GCG only": (gcg & ~pyy).to_numpy(),
            "Neither": (~pyy & ~gcg).to_numpy(),
        }
    )
    return t.groupby("State").sum().assign(Cells=t["State"].value_counts())


def _receivers() -> pd.DataFrame:
    """EC and tuft cells, and those past the C3 and C4 receiver cutoffs."""
    cells = F4.programs("enteroendocrine_cells")
    rows = {}
    for name, label, component in (
        ("EC", "Enterochromaffin cells", 3),
        ("Tuft", "Tuft cells", 4),
    ):
        weight = cells[f"C{component} receiver weight"]
        typed = weight[cells["Cell type"] == label]
        rows[name] = {
            "Cells": len(typed),
            "Past": int((typed >= GROUP_TAU * weight.max()).sum()),
        }
    return pd.DataFrame(rows).T


def _stac_hormones(ax, pyy, gcg, side: float, tuft: float) -> None:
    _title(ax, 15, 50.5, "STAC S5, S4")
    _node(ax, 5, TOP - SPAN, TOP, BAR)
    _node(ax, 25, TOP - side, TOP, BAR)
    _node(ax, 25, tuft - side, tuft, BAR)
    ee = (TOP - SPAN, TOP)
    _band(ax, 5.8, ee, 24.2, (TOP - side, TOP), pyy)
    _band(ax, 5.8, ee, 24.2, (tuft - side, tuft), gcg)
    _note(ax, 2.6, TOP - SPAN / 2, "EE", rotation=90)
    _note(ax, 27.6, TOP - side / 2, "EC", ha="left")
    _note(ax, 27.6, tuft - side / 2, "Tuft", ha="left")


def _receiver_bars(ax, receivers, side: float, tuft: float) -> dict:
    """The receiver bars; each one's span past its cutoff."""
    spans = {}
    for name, top in (("EC", TOP), ("Tuft", tuft)):
        past = receivers.loc[name, "Past"] / receivers.loc[name, "Cells"] * side
        _node(ax, 72, top - past, top, BAR)
        _node(ax, 72, top - side, top - past, BELOW)
        _note(ax, 74.6, top - side / 2, name, ha="left")
        spans[name] = [top, past]
    return spans


def _state_bands(ax, t, ends, scale: float, pyy, gcg) -> float:
    """Each state's bar and bands; the y under the last."""
    sends = {"EC": t["PYY only"] + t["Both"], "Tuft": t["GCG only"] + t["Both"]}
    y = TOP
    for name in STATES:
        r = t.loc[name]
        only, both, other = (
            r["PYY only"] * scale,
            r["Both"] * scale,
            r["GCG only"] * scale,
        )
        active = only + both + other
        _node(ax, 45, y - active, y, BAR)
        _node(ax, 45, y - r["Cells"] * scale, y - active, BELOW)
        _note(ax, 43.4, y - r["Cells"] * scale / 2, name, ha="right")
        for key, span, color in (
            ("EC", (y - only - both, y), pyy),
            ("Tuft", (y - active, y - only), gcg),
        ):
            top, past = ends[key]
            width = sends[key][name] / sends[key][list(STATES)].sum() * past
            _band(ax, 45.8, span, 71.2, (top - width, top), color)
            ends[key][0] -= width
        y -= r["Cells"] * scale + STATE_GAP
    return y


def flow_ibd(ax) -> None:
    """PYY and GCG: both bands leave all of STAC's EE label; in cccRISE each
    state's bands overlap over the cells past both sender cutoffs."""
    t, receivers = _ee_states(), _receivers()
    common.source_data(
        "ee_states_and_receivers",
        pd.concat(
            [
                t.rename_axis("Population").reset_index(),
                receivers.rename_axis("Population").reset_index(),
            ]
        ),
        "EE cells by Figure 4's state, with those past the C3 (PYY) and C4 (GCG) sender "
        "cutoffs; EC and tuft cells, with those past the C3 and C4 receiver cutoffs.",
    )
    pyy, gcg = F4.HORMONE_COLORS[3], F4.HORMONE_COLORS[4]
    _flow_axes(ax, 81)
    scale = (SPAN - len(STATES) * STATE_GAP) / t["Cells"].sum()
    side = (SPAN - RECEIVER_GAP) / 2
    tuft = TOP - side - RECEIVER_GAP
    _stac_hormones(ax, pyy, gcg, side, tuft)
    _title(ax, 56, 50.5, "CCC-RISE C3, C4")
    ends = _receiver_bars(ax, receivers, side, tuft)
    y = _state_bands(ax, t, ends, scale, pyy, gcg)
    other = t.loc["Other", "Cells"] * scale
    _node(ax, 45, y - other, y, BELOW)
    _note(ax, 43.4, y - other / 2, "Other EE", ha="right")
    both = 0.45 * np.array(mcolors.to_rgb(pyy)) + 0.55 * np.array(mcolors.to_rgb(gcg))
    _legend(
        ax,
        [
            Patch(color=pyy, alpha=BAND_ALPHA, label="PYY–NPY1R"),
            Patch(color=gcg, alpha=BAND_ALPHA, label="GCG–GLP1R"),
            Patch(color=both, alpha=0.8, label="Both"),
            Patch(color=BELOW, label="Below cutoff"),
        ],
        4,
    )
