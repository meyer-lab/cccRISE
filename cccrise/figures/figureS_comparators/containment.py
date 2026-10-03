"""How much of each cccRISE component lies in each comparator fit's span, on
the condition and LR modes (``||P_X a_k||^2``). Boxed: contained (>= 0.85 and
above the row-permutation null's 95th percentile); a component contained on
both modes is shared with that fit (bold label)."""

import numpy as np

from .. import common, style
from ._common import MODE_LABELS, fit_label, fit_order, load


def plot(fig):
    table = load("containment")
    table = table[table["direction"] == "cccRISE in comparator"]
    fits = fit_order(table)
    components = sorted(table["component"].unique())
    axes = fig.subplots(1, 2, sharey=True)
    for ax, mode in zip(axes, ("condition", "lr"), strict=True):
        image = _heatmap(ax, table[table["mode"] == mode], fits, components)
        ax.set_title(MODE_LABELS[mode])
    axes[0].set_ylabel("cccRISE component")
    fig.colorbar(image, ax=axes, label="Containment in comparator span", shrink=0.8)


def _heatmap(ax, sub, fits, components):
    """One mode's components x fits containment, contained cells boxed."""
    sub = sub.set_index(["method", "rank_mode", "component"])

    def grid(column):
        return np.array(
            [[sub.loc[(f[0], f[1], k), column] for f in fits] for k in components]
        )

    values, contained = grid("containment"), grid("contained")
    image = ax.imshow(
        values, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="auto"
    )
    for (i, j), v in np.ndenumerate(values):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            color="white" if v > 0.6 else "0.2",
        )
        if contained[i, j]:
            ax.add_patch(_box(j, i))
    ax.set_xticks(
        range(len(fits)), [fit_label(f) for f in fits], rotation=30, ha="right"
    )
    ax.set_yticks(range(len(components)), [f"C{k}" for k in components])
    ax.grid(False)
    return image


def _box(x, y):
    from matplotlib.patches import Rectangle

    return Rectangle(
        (x - 0.5, y - 0.5), 1, 1, fill=False, lw=1.2, edgecolor=style.COLORS["ink"]
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
