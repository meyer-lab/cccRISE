"""GCG and PYY, the ligands of C4's and C3's top LR pairs, on the zoomed
EE/EC/tuft map (`ee_state_map`) as contours over the grey cells: for each
gene, a kernel density of the cells weighted by their expression, outlined
where it holds `MASS` of the gene's expression (thick: the densest; thin:
the widest), in its program's color. Overlapping outlines are where both are
expressed."""

import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from .. import common, style
from . import _common as S

AXES = ("square",)
GENES = {"GCG": 4, "PYY": 3}  # gene: the program whose top ligand it is
MASS = (0.5, 0.9)  # expression-weighted mass inside each contour
BANDWIDTH = 0.15  # `gaussian_kde` bw_method, fine enough to split EE regions
GRID = 200
PAD = 0.05  # grid margin, as a fraction of the map's extent


def density_grid(xy: np.ndarray, weights: np.ndarray, gx, gy) -> np.ndarray:
    """`weights`-weighted kernel density of `xy` on the grid `gx` x `gy`."""
    kept = weights > 0
    kde = gaussian_kde(xy[kept].T, bw_method=BANDWIDTH, weights=weights[kept])
    return kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)


def mass_levels(density: np.ndarray) -> list[float]:
    """The density levels enclosing each of `MASS` of the grid's total, the
    widest first (as `contour` wants levels increasing)."""
    values = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(values) / values.sum()
    return sorted(values[np.searchsorted(cumulative, m)] for m in MASS)


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    common.source_data(
        "gcg_pyy_contours",
        cells[["Group", "PaCMAP 1", "PaCMAP 2", *GENES]],
        "Normalized GCG and PYY expression of every EE, EC and tuft cell on the "
        "zoomed map; the contours enclose "
        + " and ".join(f"{m:.0%}" for m in MASS)
        + " of each gene's expression-weighted kernel density.",
    )
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    ax.scatter(
        *xy.T,
        color=style.COLORS["muted"],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = PAD * (high - low)
    gx, gy = np.meshgrid(
        *(
            np.linspace(a, b, GRID)
            for a, b in zip(low - margin, high + margin, strict=True)
        )
    )
    for gene, k in GENES.items():
        density = density_grid(xy, cells[gene].to_numpy(), gx, gy)
        ax.contour(
            gx,
            gy,
            density,
            levels=mass_levels(density),
            colors=S.HORMONE_COLORS[k],
            linewidths=[style.LINE_WIDTH_THIN, style.LINE_WIDTH * 1.5],
        )
    S.map_axes(ax, "GCG and PYY")
    common.legend(
        ax,
        "right",
        handles=[
            Line2D([], [], color=S.HORMONE_COLORS[k], lw=style.LINE_WIDTH, label=g)
            for g, k in GENES.items()
        ],
        fontsize=style.FONT_SIZE_DENSE,
        handlelength=1.2,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
