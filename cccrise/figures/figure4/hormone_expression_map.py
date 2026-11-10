"""PYY and GCG, the ligands of C3's and C4's top LR pairs, on the EE/EC
part of the zoomed map: one panel split into two axes, each a window onto
`ee_state_map`'s PaCMAP (not refit) at exactly its scale, so the EE/EC
cluster keeps its shape and size, centered on that cluster. Each EE and EC
cell is colored by `S.relative_expression` among them, as
`figureS_ibd_hormone_axes` draws hormones and receptors."""

import matplotlib.pyplot as plt
import numpy as np

from .. import common, style
from . import _common as S

AXES = ("half_square",) * 2
JOINED = True  # one panel, lettered once (`common._Joined`)
GENES = ("PYY", "GCG")
CROP = 0.005  # share of EE/EC cells outside the cluster's center estimate


def map_limits(xy_all: np.ndarray) -> np.ndarray:
    """The data limits (rows x, y) `ee_state_map`'s autoscaled square axes
    give the full map."""
    low, high = xy_all.min(axis=0), xy_all.max(axis=0)
    margin = np.array([plt.rcParams["axes.xmargin"], plt.rcParams["axes.ymargin"]])
    pad = margin * (high - low)
    return np.column_stack([low - pad, high + pad])


def plot(axs):
    mapped = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    limits = map_limits(mapped[["PaCMAP 1", "PaCMAP 2"]].to_numpy())
    cells = mapped[mapped["Group"] != "Tuft"]
    relative = {g: S.relative_expression(cells[g].to_numpy()) for g in GENES}
    quantile = f"{S.EXPRESSION_QUANTILE:.0%} quantile"
    common.source_data(
        "pyy_gcg_on_ee_ec_map",
        cells[["PaCMAP 1", "PaCMAP 2", *GENES]].assign(
            Group=S.map_group(cells["Group"]),
            **{f"{g} / {quantile}": v for g, v in relative.items()},
        ),
        "Normalized PYY and GCG expression of every EE and EC cell on the zoomed "
        f"map, and each over its {quantile} among these cells "
        "(capped at 1, the colors).",
    )
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    # A window as wide as the axes' share of a square map, on the cluster
    center = np.quantile(xy[:, 0], [CROP, 1 - CROP]).mean()
    half = np.ptp(limits[0]) * style.AXIS_SIZE["half_square"][0] / style.AXIS_HEIGHT / 2
    for ax, gene in zip(axs, GENES, strict=True):
        points = S.expression_map(ax, xy, relative[gene])
        ax.set(xlim=(center - half, center + half), ylim=limits[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(gene, fontsize=style.FONT_SIZE_LABEL)
    axs[0].set_ylabel("PaCMAP 2")
    # One x label, centered under both axes
    width = style.AXIS_SIZE["half_square"][0]
    axs[0].set_xlabel("PaCMAP 1", fontsize=style.FONT_SIZE_DENSE)
    axs[0].xaxis.label.set_x(1 + style.JOINED_GAP / width / 2)
    common.colorbar(axs[-1], points, S.EXPRESSION_LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
