"""PaCMAP embedding of all cells colored by annotated cell type, and of the
lymphocyte cluster colored by each cell's projection onto RISE eigen-state 16.

Every cell's PaCMAP embedding colored by annotated cell type (left), and a
blow-up of the lymphocyte cluster colored by each cell's projection onto
`EIGENSTATE`, the RISE eigenstate behind the outcome-associated component's
sender and receiver weights (right).

Macrophages, most of the cells, are drawn first; the other types overlap in
the embedding (the lymphocytes entirely), so they are drawn over them in a
random order rather than by type: each shows in proportion to its density
wherever they overlap, none hidden beneath another. The blow-up's box keeps
the full embedding's x:y range ratio, so both square panels share one scale
per axis ratio and the cluster keeps its shape."""

from functools import cache

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ...runners.datasets import DATASETS
from .. import common, style
from ..figure4._common import map_axes
from ._common import DATASET, EIGENSTATE

BACKGROUND_TYPE = "Macrophages"  # most of the cells: drawn beneath the rest
LYMPHOCYTES = ("CD4 T cells", "CD8 T cells", "NK cells")
ZOOM_XLABEL = "PaCMAP 1 (zoom: lymphocytes)"
CLUSTER_RADIUS = 4  # PaCMAP units around the lymphocytes' median: their island
BOX_QUANTILES = (0, 1)  # of the island's lymphocytes, per axis: its full extent
BOX_PAD = 0.05  # of the box's size, on each side
AXES = ("square",) * 2


def plot(axs):
    adata = common.get_cell_projections(DATASET)
    x, y = np.asarray(adata.obsm["PaCMAP"]).T
    celltypes = adata.obs[DATASETS[DATASET].celltype_column].astype(str).to_numpy()
    projection = adata.obsm["projections"][:, EIGENSTATE - 1]
    _celltype_scatter(axs[0], x, y, celltypes)
    box, inside = blowup()
    axs[0].add_patch(
        Rectangle(
            box[:2],
            *box[2:],
            fill=False,
            linestyle=style.LINE_STYLE_REFERENCE,
            edgecolor=style.COLORS["reference"],
            linewidth=style.LINE_WIDTH_THIN,
        )
    )
    axs[0].text(
        box[0],
        box[1],
        "Lymphocytes",
        ha="left",
        va="top",
        color=style.COLORS["reference"],
    )
    points = common.plot_value_scatter(axs[1], x[inside], y[inside], projection[inside])
    points.set_sizes([style.POINT_SIZE_BACKGROUND])
    axs[1].set(xlim=(box[0], box[0] + box[2]), ylim=(box[1], box[1] + box[3]))
    map_axes(axs[1], xlabel=ZOOM_XLABEL, xlabel_size=style.FONT_SIZE_LABEL)
    common.colorbar(axs[1], points, f"Eigen-state {EIGENSTATE} projection")
    common.source_data(
        "pacmap_celltype_eigenstate",
        pd.DataFrame(
            {
                "PaCMAP 1": x,
                "PaCMAP 2": y,
                "Cell type": celltypes,
                f"Eigen-state {EIGENSTATE} projection": projection,
                "In blow-up": inside,
            },
            index=adata.obs_names.rename("Cell barcode"),
        ),
        "One row per cell: its PaCMAP coordinates, annotated cell type, and "
        f"projection onto RISE eigen-state {EIGENSTATE}; 'In blow-up' marks the "
        "lymphocyte-cluster cells drawn in the right panel.",
    )


def _celltype_scatter(ax, x, y, celltypes) -> None:
    """Cells colored by cell type, `BACKGROUND_TYPE` first and the rest in a
    random order, so overlapping types stay visible in proportion to their
    density, with the type key beside."""
    colors = style.CELL_TYPE_COLORS[DATASET]
    shuffled = np.random.default_rng(0).permutation(len(x))
    order = shuffled[np.argsort(celltypes[shuffled] != BACKGROUND_TYPE, kind="stable")]
    ax.scatter(
        x[order],
        y[order],
        c=[colors[c] for c in celltypes[order]],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    map_axes(ax, xlabel="PaCMAP 1", xlabel_size=style.FONT_SIZE_LABEL)
    common.legend(
        ax,
        "right",
        handles=common.celltype_handles(DATASET),
    )


@cache
def blowup() -> tuple[tuple[float, float, float, float], np.ndarray]:
    """The lymphocyte blow-up's (x0, y0, width, height) box, and which cells
    fall inside it."""
    adata = common.get_cell_projections(DATASET)
    x, y = np.asarray(adata.obsm["PaCMAP"]).T
    celltypes = adata.obs[DATASETS[DATASET].celltype_column].astype(str).to_numpy()
    box = _lymphocyte_box(x, y, np.isin(celltypes, LYMPHOCYTES))
    inside = (
        (x >= box[0]) & (x <= box[0] + box[2]) & (y >= box[1]) & (y <= box[1] + box[3])
    )
    return box, inside


def projection_limit() -> float:
    """The blow-up's color limit (`plot_value_scatter`'s saturation over its
    cells), for panels sharing its scale."""
    projection = common.get_cell_projections(DATASET).obsm["projections"]
    return float(
        np.percentile(
            np.abs(projection[blowup()[1], EIGENSTATE - 1]),
            common.SATURATION_PERCENTILE,
        )
    )


def _lymphocyte_box(x, y, is_lymphocyte) -> tuple[float, float, float, float]:
    """(x0, y0, width, height) around the central `BOX_QUANTILES` of the
    lymphocytes within `CLUSTER_RADIUS` of their median (~90% of them; the
    rest are scattered through other clusters), padded and widened in x or y to the full panel's x:y
    range ratio."""
    center = [np.median(v[is_lymphocyte]) for v in (x, y)]
    island = is_lymphocyte & (np.hypot(x - center[0], y - center[1]) < CLUSTER_RADIUS)
    (x0, x1), (y0, y1) = (np.quantile(v[island], BOX_QUANTILES) for v in (x, y))
    width, height = (x1 - x0) * (1 + 2 * BOX_PAD), (y1 - y0) * (1 + 2 * BOX_PAD)
    ratio = np.ptp(y) / np.ptp(x)
    width, height = max(width, height / ratio), max(height, width * ratio)
    return ((x0 + x1 - width) / 2, (y0 + y1 - height) / 2, width, height)


def makeFigure():
    return common.standalone_fixed(__name__)
