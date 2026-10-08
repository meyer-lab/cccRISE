"""What, who, where (draft panel B): every CP comparator column (own rank),
grouped by fit (PB coarse, PB fine, TC2C coarse, TC2C fine), against its best
cccRISE match (the component with the largest per-pair min loading on their
shared top-10 pairs; left bar, in the component's colour). Cells: the
|cosine| of their LR columns, of their sender and receiver cell-type profiles
(cccRISE: mean sample-scaled per-cell loading per type), and of their
condition columns. Right bar: the column's class (`lr_class_min`: the LR
pairs match when the per-pair min reaches 0.5; recovered when sender,
receiver and samples also reach 0.5 |cosine|; otherwise one sample or broad
by the column's sample participation ratio). STACCato is left out: its
Tucker core does not tie a column to one sender, receiver and sample
profile."""

import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle

from .. import common, style
from ..figureS_lr_concordance._common import ANNOTATIONS, CLASSES, SHORT, load

AGREEMENTS = {
    "lr_abs_cos": "LR",
    "sender_cos": "sender",
    "receiver_cos": "receiver",
    "condition_cos": "samples",
}
FITS = [
    (method, annotation)
    for method in ("Pseudobulk CPD", "Tensor-cell2cell")
    for annotation in ANNOTATIONS
]
CLASS_COLORS = dict(
    zip(
        CLASSES,
        (
            style.COLORS["emphasis"],
            "#86b6ef",  # light blue: the LR pairs agree, the rest does not
            style.COLORS["up"],
            style.COLORS["muted"],
        ),
        strict=True,
    )
)
CLASS_NAMES = dict(
    zip(
        CLASSES,
        ("recovered", "same LR, other who/where", "one sample", "broad"),
        strict=True,
    )
)
GAP = 0.6  # rows of space between fits
BAR = (-0.45, -0.08)  # x extent of the left bar; the right bar mirrors it


def rows():
    """CP columns in `FITS` order, each with its row position `y`."""
    matches = load("matches")
    matches = matches[matches["sender_cos"].notna()]
    order = {fit: i for i, fit in enumerate(FITS)}
    matches = matches.assign(
        fit=[order[f] for f in zip(matches["method"], matches["annotation"])]
    ).sort_values(["fit", "comparator_component"], ignore_index=True)
    return matches.assign(y=matches.index + GAP * matches["fit"])


def _cells(ax, table):
    cmap, norm = style.CMAPS["sequential"], Normalize(0, 1)
    for j, column in enumerate(AGREEMENTS):
        for y, v in zip(table["y"], table[column], strict=True):
            ax.add_patch(Rectangle((j, y), 1, 1, color=cmap(norm(v)), lw=0))
            ax.text(
                j + 0.5, y + 0.5, f"{v:.2f}", ha="center", va="center",
                fontsize=style.FONT_SIZE_DENSE - 1,
                color="white" if v > 0.6 else style.COLORS["ink"],
            )  # fmt: skip
    return ScalarMappable(norm, cmap)


RIGHT = len(AGREEMENTS) - BAR[1]  # left edge of the right bar


def _bars(ax, table, palette):
    """Left: the matched component per row; right: the row's class."""
    for row in table.itertuples():
        c = int(row.cccrise_component)
        ax.add_patch(
            Rectangle(
                (BAR[0], row.y + 0.05), BAR[1] - BAR[0], 0.9,
                color=palette[c - 1], lw=0, clip_on=False,
            )
        )  # fmt: skip
        ax.text(
            sum(BAR) / 2, row.y + 0.5, f"C{c}", ha="center", va="center",
            color="white", fontsize=style.FONT_SIZE_DENSE - 1, fontweight="bold",
        )  # fmt: skip
        ax.add_patch(
            Rectangle(
                (RIGHT, row.y + 0.05), BAR[1] - BAR[0], 0.9,
                color=CLASS_COLORS[row.lr_class_min], lw=0, clip_on=False,
            )
        )  # fmt: skip


def _fit_labels(ax, table):
    for fit, sub in table.groupby("fit"):
        method, annotation = FITS[fit]
        ax.text(
            BAR[0] - 0.12, sub["y"].mean() + 0.5,
            f"{SHORT[method]}\n{annotation}", ha="right", va="center",
            fontsize=style.FONT_SIZE_DENSE,
        )  # fmt: skip


def _legend(fig):
    fig.legend(
        handles=[
            Patch(color=color, label=CLASS_NAMES[name])
            for name, color in CLASS_COLORS.items()
        ],
        loc="outside lower center",
        ncol=2,
        fontsize=style.FONT_SIZE_DENSE,
        frameon=False,
        title="class (right bar)",
        title_fontsize=style.FONT_SIZE_DENSE,
    )


def plot(fig):
    table = rows()
    ax = fig.subplots()
    mappable = _cells(ax, table)
    _bars(ax, table, sns.color_palette("Dark2", 5))
    _fit_labels(ax, table)
    width = BAR[1] - BAR[0]
    ax.set_xlim(BAR[0], RIGHT + width)
    ax.set_ylim(table["y"].max() + 1, 0)
    ax.set_xticks(
        [sum(BAR) / 2, *(np.arange(len(AGREEMENTS)) + 0.5), RIGHT + width / 2],
        ["cccRISE", *AGREEMENTS.values(), "class"],
    )
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.colorbar(
        mappable, ax=ax, label="|cosine|", shrink=0.6, location="bottom", pad=0.02
    )
    _legend(fig)


def makeFigure():
    return common.standalone(plot, (100 * style.MM, 95 * style.MM))
