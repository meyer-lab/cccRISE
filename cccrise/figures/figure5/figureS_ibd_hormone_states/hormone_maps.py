"""GCG, PYY, NTS and SST expression on Figure 5's zoomed EE/EC/tuft map: the
C4-high region is GCG-high, the C3-high region PYY- and NTS-high, and the
L-hormone-negative EE cells SST-high."""

import numpy as np

from ... import common, style
from .. import _common as S

AXES = ("square",) * 4
GENES = ["GCG", "PYY", "NTS", "SST"]


def plot(axs):
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    common.source_data(
        "hormone_expression_on_map",
        cells[["PaCMAP 1", "PaCMAP 2", *GENES]],
        "Normalized expression of each hormone in EE, EC and tuft cells, on the zoomed map.",
    )
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    for ax, gene in zip(axs, GENES, strict=True):
        v = cells[gene].to_numpy()
        order = np.argsort(v)
        points = ax.scatter(
            *xy[order].T,
            c=v[order],
            cmap="Reds",
            vmin=0,
            vmax=np.quantile(v, 0.99),
            s=style.POINT_SIZE_CELL,
            linewidths=0,
            rasterized=True,
        )
        S.map_axes(ax, gene)
    common.colorbar(axs[-1], points, "Expression")


def makeFigure():
    return common.standalone_fixed(__name__)
