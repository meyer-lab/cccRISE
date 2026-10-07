"""What a cell-type mean keeps of a cccRISE component, in the cells that
carry it. PaCMAP of the cells of each component's home coarse types (>= 20%
of cells active: loading > 0.5 x its 99th percentile) coloured by their
sample-scaled receiver loading on C4 (top) and C3 (bottom), then by that
loading's (sample x type) mean at the coarse annotation (8 types), the fine
annotation (26), expression subtypes (m = 8, 38 types) and random splits of
the same sizes. Each panel gives the variance it keeps within those cells
(R^2)."""

import numpy as np

from .. import common, style
from ._common import load_ladder

ROWS = ("receiver_C4", "receiver_C3")
LEVELS = {
    "": "per cell",
    "_coarse": "coarse (8)",
    "_fine": "fine (26)",
    "_expr_m8": "expression (38)",
    "_rand_m8": "random (38)",
}
HOME_ACTIVE = 0.2


def home_cells(table, name) -> np.ndarray:
    """Cells of the coarse types where >= `HOME_ACTIVE` of cells are active."""
    values = table[name].to_numpy()
    active = values > 0.5 * np.quantile(values, 0.99)
    frac = table.assign(active=active).groupby("coarse")["active"].mean()
    return table["coarse"].isin(frac[frac >= HOME_ACTIVE].index).to_numpy()


def _draw(ax, sub, shown, lim):
    order = np.argsort(shown)
    ax.scatter(
        sub["pacmap_1"].to_numpy()[order],
        sub["pacmap_2"].to_numpy()[order],
        c=shown[order],
        s=0.8,
        lw=0,
        cmap=style.CMAPS["sequential"],
        vmin=lim[0],
        vmax=lim[1],
        rasterized=True,
    )
    for dim, setter in (("pacmap_1", ax.set_xlim), ("pacmap_2", ax.set_ylim)):
        center = np.median(sub[dim])
        spread = 4 * np.median(np.abs(sub[dim] - center))
        setter(center - spread, center + spread)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot(fig):
    table = load_ladder("strip")
    axes = fig.subplots(len(ROWS), len(LEVELS))
    for r, name in enumerate(ROWS):
        sub = table[home_cells(table, name)]
        values = sub[name].to_numpy()
        lim = np.quantile(values, [0.01, 0.99])
        for c, (suffix, title) in enumerate(LEVELS.items()):
            ax = axes[r, c]
            shown = sub[name + suffix].to_numpy()
            _draw(ax, sub, shown, lim)
            if suffix:
                r2 = 1 - np.sum((values - shown) ** 2) / np.sum(
                    (values - values.mean()) ** 2
                )
                ax.set_xlabel(f"$R^2$ {r2:.2f}", fontsize=style.FONT_SIZE_DENSE)
            if r == 0:
                ax.set_title(title, fontsize=style.FONT_SIZE_LABEL)
            if c == 0:
                types = sorted(t.replace(" cells", "") for t in sub["coarse"].unique())
                ax.set_ylabel(
                    f"{name.replace('receiver_', '')} receiver\n{', '.join(types)}",
                    fontsize=style.FONT_SIZE_DENSE,
                )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
