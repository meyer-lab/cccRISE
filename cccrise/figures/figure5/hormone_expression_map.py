"""GCG and PYY, the ligands of C4's and C3's top LR pairs, on the EE/EC
part of the zoomed map (`ee_state_map`'s PaCMAP, cropped to those cells, not
refit): each cell's expression over the gene's `QUANTILE` among EE and EC
cells (capped at 1), highest drawn on top, as `figureS_ibd_hormone_states`
draws hormones."""

import numpy as np

from .. import common, style
from . import _common as S

AXES = ("half_square",) * 2
GENES = ("GCG", "PYY")
QUANTILE = 0.99
CROP = 0.005  # share of cells left outside the view on each side, per axis
PAD = 0.04


def plot(axs):
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    cells = cells[cells["Group"].isin(list(S.EE_GROUPS))]  # tuft cells left out
    relative = {
        g: np.clip(cells[g].to_numpy() / np.quantile(cells[g], QUANTILE), 0, 1)
        for g in GENES
    }
    common.source_data(
        "gcg_pyy_on_ee_ec_map",
        cells[["Group", "PaCMAP 1", "PaCMAP 2", *GENES]].assign(
            **{f"{g} / {QUANTILE:.0%} quantile": v for g, v in relative.items()}
        ),
        "Normalized GCG and PYY expression of every EE and EC cell on the zoomed "
        f"map, and each over its {QUANTILE:.0%} quantile among these cells "
        "(capped at 1, the colors).",
    )
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    # Crop to the cluster, leaving out the few EE/EC cells mapped far from it
    low, high = np.quantile(xy, [CROP, 1 - CROP], axis=0)
    pad = PAD * (high - low)
    for ax, gene in zip(axs, GENES, strict=True):
        ax.set(xlim=(low[0] - pad[0], high[0] + pad[0]))
        ax.set(ylim=(low[1] - pad[1], high[1] + pad[1]))
        order = np.argsort(relative[gene])
        points = ax.scatter(
            *xy[order].T,
            c=relative[gene][order],
            cmap="Reds",
            vmin=0,
            vmax=1,
            s=style.POINT_SIZE_CELL,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(gene, fontsize=style.FONT_SIZE_LABEL)
    axs[0].set_ylabel("PaCMAP 2")
    axs[0].set_xlabel("PaCMAP 1 (EE, EC)", fontsize=style.FONT_SIZE_DENSE, loc="left")
    common.colorbar(axs[-1], points, f"Expr. / {QUANTILE:.0%} quantile")


def makeFigure():
    return common.standalone_fixed(__name__)
