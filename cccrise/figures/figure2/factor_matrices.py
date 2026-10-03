"""The canonical ALAD CP factor matrices: condition (A), sender eigenstate
(B), receiver eigenstate (C), and LR pair (D) loadings per component, each
divided by its largest magnitude within the component (keeping sign).
Only the `TOP_LR_FRACTION` of LR pairs with the largest such loading on any
component are shown, plus each component's top pairs, which are outlined and
labeled: those with |loading| at least 0.4 x the component's largest, the
rule `nichenet` picks its tested ligands by (`validation.nichenet.top_pairs`),
here over every LR pair rather than only those NicheNet can score.
Conditions and LR pairs are ordered by their most heavily loaded component,
then by that loading."""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ...runners.datasets import DATASETS
from ...validation.nichenet import top_pairs
from .. import common, style
from ._common import DATASET

MODES = {  # factor matrix: row label, whether to order rows
    "A": ("Condition", True),
    "B": ("Sender eigenstate", False),
    "C": ("Receiver eigenstate", False),
    "D": ("LR pair", True),
}
EIGENSTATE_TICKS = (1, 5, 10, 15)
TOP_LR_FRACTION = 0.15
# LR pair labels overlaid on the heatmap: below FONT_SIZE_DENSE so the names
# fit inside it, stacked LABEL_GAP apart, each LEADER_LENGTH (columns) from
# its cell
LABEL_FONT_SIZE = 5
LABEL_GAP = 1.9 * style.MM
LEADER_LENGTH = 0.2
AXES = ("square",) * len(MODES)


def plot(axs):
    adata = common.load_factors_to_anndata(DATASETS[DATASET].factors_path())
    for ax, (key, (ylabel, ordered)) in zip(axs, MODES.items(), strict=True):
        loadings = np.asarray(adata.uns[key], dtype=float)
        loadings = loadings / np.abs(loadings).max(axis=0)
        labeled = {}
        if key == "D":
            rows, labeled = _shown_lr_pairs(adata, loadings)
            loadings = loadings[rows]
            ylabel = f"Top {TOP_LR_FRACTION:.0%} LR pair"
        order = _dominant_order(loadings) if ordered else np.arange(len(loadings))
        image = ax.imshow(
            loadings[order],
            cmap=style.CMAPS["diverging"],
            vmin=-1,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        components = np.arange(loadings.shape[1])
        ax.set_xticks(components, components + 1)
        if ordered:
            ax.set_yticks([])
            ylabel += f"s (n={len(loadings)})"
        else:
            ax.set_yticks([t - 1 for t in EIGENSTATE_TICKS], EIGENSTATE_TICKS)
        ax.set(xlabel="CP component", ylabel=ylabel)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        position = np.argsort(order)  # row index -> drawn row
        _label_pairs(
            ax,
            [(position[r], c, name) for r, (c, name) in labeled.items()],
            loadings.shape,
        )
    common.colorbar(axs[-1], image, "Loading / max |loading|")


def _shown_lr_pairs(adata, loadings) -> tuple[np.ndarray, dict[int, tuple]]:
    """Rows of D shown (the top `TOP_LR_FRACTION` by largest normalized
    loading, plus the top pairs), and {shown row: (0-based component, pair
    name)} for the top pairs."""
    names = pd.Series(np.asarray(adata.uns["lr_pairs"]).astype(str))
    top = _top_pairs(adata.uns["D"], names)
    strongest = np.abs(loadings).max(axis=1)
    n_top = round(TOP_LR_FRACTION * len(loadings))
    rows = np.union1d(np.argsort(-strongest)[:n_top], list(top))
    shown = {r: i for i, r in enumerate(rows)}
    return rows, {shown[r]: (c - 1, names[r].replace("^", "–")) for r, c in top.items()}


def _top_pairs(D, names: pd.Series) -> dict[int, int]:
    """{row of D: 1-based component} of each component's top pairs: NicheNet's
    relative-loading rule over every LR pair, not only those NicheNet can
    score."""
    lr_pairs = pd.DataFrame({"ligand": names.str.split("^").str[0]})
    ligands = set(lr_pairs["ligand"])
    return {
        int(row): c
        for c in range(1, np.shape(D)[1] + 1)
        for row in top_pairs(D, lr_pairs, c, ligands)
    }


def _label_pairs(ax, entries, shape) -> None:
    """Outline each (drawn row, component, name) cell and label it beside its
    column, toward the wider blank side: rightward up to the middle
    component, leftward after it, each side's labels stacked `LABEL_GAP`
    apart."""
    n_rows, n_cols = shape
    gap = LABEL_GAP / (ax.bbox.height / ax.figure.dpi) * n_rows  # in rows
    for rightward in (True, False):
        side = sorted(e for e in entries if (e[1] <= n_cols // 2) == rightward)
        heights = _spread([row for row, _, _ in side], gap, n_rows)
        for (row, col, name), y in zip(side, heights, strict=True):
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=style.LINE_WIDTH_THIN,
                )
            )
            edge = col + (0.5 if rightward else -0.5)
            x = edge + (LEADER_LENGTH if rightward else -LEADER_LENGTH)
            ax.plot([edge, x], [row, y], color="black", lw=style.LINE_WIDTH_THIN)
            ax.text(
                x,
                y,
                name,
                ha="left" if rightward else "right",
                va="center",
                fontsize=LABEL_FONT_SIZE,
            )


def _spread(rows: list[float], gap: float, n_rows: int) -> np.ndarray:
    """Heights for sorted `rows` at least `gap` apart, centered on them and
    kept within the heatmap."""
    placed = np.array(rows, dtype=float)
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    if len(placed):
        placed -= placed.mean() - np.mean(rows)
        placed -= max(placed[-1] - (n_rows - 1), 0) + min(placed[0], 0)
    return placed


def _dominant_order(loadings: np.ndarray) -> np.ndarray:
    """Rows grouped by their largest-magnitude component, then by that loading."""
    dominant = np.abs(loadings).argmax(axis=1)
    value = np.abs(loadings[np.arange(len(loadings)), dominant])
    return np.lexsort((-value, dominant))


def makeFigure():
    return common.standalone_fixed(__name__)
