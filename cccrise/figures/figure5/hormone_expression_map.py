"""GCG and PYY, the ligands of C4's and C3's top LR pairs, on the zoomed
EE/EC/tuft map (`ee_state_map`): each grey cell tinted by its graded expression
of both, GCG in C4's color and PYY in C3's, multiplied so cells expressing
both are darkest; each gene scaled to its `QUANTILE` over these cells."""

import numpy as np
from matplotlib.colors import to_rgb

from .. import common, style
from . import _common as S

AXES = ("square",)
GENES = {"GCG": 4, "PYY": 3}  # gene: the program whose top ligand it is
QUANTILE = 0.99
KEY_STEPS = 32
BASE = 0.88  # grey of a cell expressing neither


def tint(levels: dict[str, np.ndarray]) -> np.ndarray:
    """RGB of each cell: `BASE` grey tinted toward each gene's program color
    by its level (0-1), the tints multiplied."""
    rgb = np.full((len(next(iter(levels.values()))), 3), BASE)
    for gene, level in levels.items():
        color = np.array(to_rgb(S.HORMONE_COLORS[GENES[gene]]))
        rgb *= 1 - level[:, None] * (1 - color)
    return rgb


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    common.source_data(
        "gcg_pyy_on_map",
        cells[["Group", "PaCMAP 1", "PaCMAP 2", *GENES]],
        "Normalized GCG and PYY expression of every EE, EC and tuft cell on the "
        f"zoomed map; each scaled to its {QUANTILE:.0%} quantile (capped at 1) "
        "for the colors.",
    )
    levels = {
        g: np.clip(cells[g].to_numpy() / np.quantile(cells[g], QUANTILE), 0, 1)
        for g in GENES
    }
    order = np.argsort(sum(levels.values()))
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()[order]
    ax.scatter(
        *xy.T,
        c=tint({g: v[order] for g, v in levels.items()}),
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    S.map_axes(ax, " and ".join(GENES))
    _key(ax)


def _key(ax) -> None:
    """The two-gene color key in `ax`'s right margin."""
    grid = np.linspace(0, 1, KEY_STEPS)
    gcg, pyy = np.meshgrid(grid, grid, indexing="ij")
    rgb = tint({"GCG": gcg.ravel(), "PYY": pyy.ravel()})
    key = ax.inset_axes((1.08, 0.3, 0.4, 0.4))
    key.imshow(rgb.reshape(KEY_STEPS, KEY_STEPS, 3), origin="lower")
    key.set_xticks([])
    key.set_yticks([])
    key.set_xlabel("PYY", fontsize=style.FONT_SIZE_DENSE, labelpad=1)
    key.set_ylabel("GCG", fontsize=style.FONT_SIZE_DENSE, labelpad=1)


def makeFigure():
    return common.standalone_fixed(__name__)
