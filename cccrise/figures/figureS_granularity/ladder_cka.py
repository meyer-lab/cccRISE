"""Linear CKA of each comparator with cccRISE as the annotation gets finer, at
rank 5. Solid: every coarse type split by its own expression; dashed: random
splits of the same sizes; star: the fine annotation (26 types); open marks:
the comparators' own-rank fits; dash: the larger null 95th percentile at
the coarse annotation."""

from .. import common, style
from ._common import METHODS, SHORT, draw_ladder, load_fits

MODES = {"condition": "Condition mode", "lr": "LR mode"}


def plot(fig):
    table = load_fits()
    colors = dict(zip(METHODS, style.CATEGORICAL[1:], strict=False))
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (mode, title) in zip(axes, MODES.items(), strict=True):
        for method in METHODS:
            draw_ladder(ax, table, method, f"cka_{mode}", colors[method])
            ref = table[(table["method"] == method) & (table["level"] == "coarse")]
            ax.scatter(
                [8],
                ref[f"cka_{mode}_q95"].iloc[:1],
                marker="_",
                s=60,
                color=colors[method],
                alpha=0.6,
            )
        ax.set_title(title)
        ax.set_xlabel("cell types")
    axes[0].set_ylabel("linear CKA with cccRISE")
    for method in METHODS:
        axes[1].plot([], [], color=colors[method], label=SHORT[method])
    axes[1].plot(
        [], [], color=style.COLORS["ink"], ls="--", lw=0.5, label="random split"
    )
    axes[1].legend(fontsize=style.FONT_SIZE_DENSE, frameon=False)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
