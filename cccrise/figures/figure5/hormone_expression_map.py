"""GCG and PYY, the ligands of C4's and C3's top LR pairs, on the zoomed
EE/EC/tuft map (`ee_state_map`): each cell expressing either colored by
PYY's share of the two (each gene first scaled to its `QUANTILE` over these
cells), from GCG-only (C4's side of the scale) through both to PYY-only
(C3's), the colors blended between so co-expressing cells read purple
rather than as the neutral grey of cells expressing neither. Each cell's
opacity follows its total (scaled GCG + PYY, from `MIN_ALPHA`), so stray
reads in EC and tuft cells fade rather than paint a full color."""

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from .. import common, style
from . import _common as S

AXES = ("square",)
QUANTILE = 0.99
MIN_ALPHA = 0.15
SHARE_CMAP = LinearSegmentedColormap.from_list(
    "gcg_pyy", [S.HORMONE_COLORS[4], S.HORMONE_COLORS[3]]
)


def pyy_share(cells) -> tuple[np.ndarray, np.ndarray]:
    """(PYY / (GCG + PYY), NaN for neither; GCG + PYY), each gene scaled to
    its `QUANTILE` and capped at 1."""
    gcg, pyy = (
        np.clip(cells[g].to_numpy() / np.quantile(cells[g], QUANTILE), 0, 1)
        for g in ("GCG", "PYY")
    )
    total = gcg + pyy
    share = np.divide(pyy, total, out=np.full(len(total), np.nan), where=total > 0)
    return share, total


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    share, total = pyy_share(cells)
    common.source_data(
        "gcg_pyy_on_map",
        cells[["Group", "PaCMAP 1", "PaCMAP 2", "GCG", "PYY"]].assign(
            **{"PYY share": share}
        ),
        "Normalized GCG and PYY expression of every EE, EC and tuft cell on the "
        f"zoomed map, and PYY's share of the two after scaling each to its "
        f"{QUANTILE:.0%} quantile (capped at 1; opacity follows their sum); "
        "empty where neither is detected.",
    )
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    neither = np.isnan(share)
    ax.scatter(
        *xy[neither].T,
        color=style.COLORS["muted"],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    shown = np.flatnonzero(~neither)
    shown = shown[np.argsort(total[shown])]  # strongest on top
    colors = SHARE_CMAP(share[shown])
    colors[:, 3] = MIN_ALPHA + (1 - MIN_ALPHA) * np.clip(total[shown], 0, 1)
    ax.scatter(
        *xy[shown].T,
        c=colors,
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    S.map_axes(ax, "GCG vs PYY")
    bar = common.colorbar(ax, ScalarMappable(cmap=SHARE_CMAP), "PYY share")
    bar.set_ticks([0, 0.5, 1], labels=["GCG", "Both", "PYY"])


def makeFigure():
    return common.standalone_fixed(__name__)
