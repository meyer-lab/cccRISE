"""The eigenstates the hormone programs (C3, C4) load on, and the cell
categories (`run_ibd_programs.cell_categories`) this supplement breaks them
down by, and the zoomed map's value panels."""

import numpy as np

from ... import common, style
from .. import _common as S

#: An eigenstate is the programs' when its sender (B) or receiver (C) loading
#: on C3 or C4 reaches this
LOADING_MIN = 0.2
ROLES = {"sender": "cp_B", "receiver": "cp_C"}
#: Cell categories, epithelial first; the immune lineages are pooled in bars
CATEGORIES = [
    "EE",
    "EC",
    "Tuft",
    "Goblet",
    "Enterocyte",
    "Stem cell",
    "T cell",
    "B cell",
    "Plasma B cell",
    "Myeloid",
]
IMMUNE = ["T cell", "B cell", "Plasma B cell", "Myeloid"]
CATEGORY_COLORS = S.MAP_GROUPS | {
    "Goblet": style.CATEGORICAL[4],
    "Enterocyte": style.CATEGORICAL[5],
    "Stem cell": style.CATEGORICAL[8],
    "Immune": "0.8",
}


def eigenstates() -> dict[str, list[int]]:
    """{role: the eigenstates (1-indexed) whose C3 or C4 loading in that
    role reaches `LOADING_MIN`, C3's first (by C3 minus C4 squared loading)}."""
    out = {}
    for role, factor in ROLES.items():
        loadings = S.factors()[factor].values[:, [2, 3]].astype(float)
        rows = np.flatnonzero(np.abs(loadings).max(axis=1) >= LOADING_MIN)
        bias = loadings[rows, 0] ** 2 - loadings[rows, 1] ** 2
        out[role] = [int(j) + 1 for j in rows[np.argsort(-bias)]]
    return out


def eigenstate_rows(ax) -> list[int]:
    """Label `ax`'s rows with the programs' eigenstates, sender then receiver,
    bold in the role's color and split by a line; returns the eigenstates in
    row order."""
    groups = eigenstates()
    rows = groups["sender"] + groups["receiver"]
    ax.set_yticks(range(len(rows)), [f"E{j}" for j in rows])
    roles = [role for role, js in groups.items() for _ in js]
    for label, role in zip(ax.get_yticklabels(), roles, strict=True):
        label.set(fontweight="bold", color=style.COLORS[role])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axhline(len(groups["sender"]) - 0.5, **style.SPACER)
    return rows


def label_roles(ax) -> None:
    """Name the sender and receiver eigenstate rows in `ax`'s left margin."""
    groups = eigenstates()
    n = len(groups["sender"])
    for role, middle in (
        ("Sender", (n - 1) / 2),
        ("Receiver", n + (len(groups["receiver"]) - 1) / 2),
    ):
        ax.annotate(
            role,
            (0, middle),
            xycoords=("axes fraction", "data"),
            xytext=(-24, 0),
            textcoords="offset points",
            rotation=90,
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
        )


def zoomed_maps(axs, columns: list[str], draw, label: str) -> None:
    """Each of `columns` of the EE/EC/tuft table painted on Figure 4's zoomed
    map by `draw` (`S.weight_map` or `expression`), titled by the column, the
    colorbar (`label`) beside the last."""
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    for ax, column in zip(axs, columns, strict=True):
        points = draw(ax, xy, cells[column].to_numpy())
        S.map_axes(ax, column)
    common.colorbar(axs[-1], points, label)


def expression(ax, xy, values):
    """`values` as Figure 4H draws expression (`S.relative_expression`)."""
    return S.expression_map(ax, xy, S.relative_expression(values))
