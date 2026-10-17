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

from ...runners.datasets import DATASETS
from ...validation.nichenet import top_pairs
from .. import common

DATASET = "alad"

MODES = {  # factor matrix: row label, whether to order rows
    "A": ("Condition", True),
    "B": ("Sender eigenstate", False),
    "C": ("Receiver eigenstate", False),
    "D": ("LR pair", True),
}
EIGENSTATE_TICKS = (1, 5, 10, 15)
TOP_LR_FRACTION = 0.15
AXES = ("square",) * len(MODES)
REL_CUTOFF = 0.4  # percent of a components top-weighted ligand above which LR pairs should be labeled


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
        order = common.dominant_order(loadings) if ordered else np.arange(len(loadings))
        image = common.plot_factor_heatmap(ax, loadings[order], "CP component")
        if ordered:
            ax.set_yticks([])
            ylabel += f"s (n={len(loadings)})"
        else:
            ax.set_yticks([t - 1 for t in EIGENSTATE_TICKS], EIGENSTATE_TICKS)
        ax.set_ylabel(ylabel)
        position = np.argsort(order)  # row index -> drawn row
        common.label_heatmap_cells(
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
        for row in top_pairs(D, lr_pairs, c, ligands, rel_cutoff=REL_CUTOFF)
    }


def makeFigure():
    return common.standalone_fixed(__name__)
