"""CP factor matrices: condition (A), sender eigenstate (B), receiver
eigenstate (C), and LR pair (D) loadings per component, each divided by its
largest magnitude within the component (keeping sign); Figure 2 draws the
canonical ALAD fit (`draw` draws any registered dataset's).
Only the `TOP_LR_FRACTION` of LR pairs with the largest such loading on any
component are shown, plus each component's top pairs, which are outlined and
labeled: those with |loading| at least 0.4 x the component's largest, the
rule `nichenet` picks its tested ligands by (`validation.nichenet.top_pairs`),
here over every LR pair rather than only those NicheNet can score.
Conditions are grouped by ALAD status, then six-month outcome
(`common.condition_strips`), and within a group, like LR pairs, ordered by
their most heavily loaded component, then by that loading."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ...runners.datasets import DATASETS
from ...validation.nichenet import top_pairs
from .. import common
from ..figure3._common import condition_obs

DATASET = "alad"

MODES = {  # factor matrix: row label, whether to order rows
    "A": ("Condition", True),
    "B": ("Sender eigenstate", False),
    "C": ("Receiver eigenstate", False),
    "D": ("LR pair", True),
}
EIGENSTATE_TICKS = (1, 5, 10, 15)
TOP_LR_FRACTION = 0.15
AXES = ("slim", "slim", "slim", "square")
REL_CUTOFF = 0.4  # percent of a components top-weighted ligand above which LR pairs should be labeled


def plot(axs):
    draw(axs, DATASET, condition_obs(), common.CONDITION_STRIPS)


def draw(
    axs,
    dataset: str,
    groups: pd.DataFrame,
    strips: dict,
    eigenstate_ticks: Sequence[int] = EIGENSTATE_TICKS,
    top_lr_fraction: float = TOP_LR_FRACTION,
    compact: bool = False,
):
    """`dataset`'s CP factor matrices on `axs` (one per mode), conditions
    grouped by their `groups` (obs indexed by condition) under `strips`
    (`common.condition_strips`), the eigenstate axes ticked at
    `eigenstate_ticks` and the `top_lr_fraction` of LR pairs shown. With
    `compact`, the strips are named at brackets instead of a legend and each
    LR label is placed clear of every outlined cell and other label, the
    colorbar right of any that has to sit past the matrix."""
    adata = common.load_factors_to_anndata(DATASETS[dataset].factors_path())
    for ax, (key, (ylabel, ordered)) in zip(axs, MODES.items(), strict=True):
        loadings = np.asarray(adata.uns[key], dtype=float)
        loadings = loadings / np.abs(loadings).max(axis=0)
        labeled = {}
        if key == "D":
            rows, labeled = _shown_lr_pairs(adata, loadings, top_lr_fraction)
            _source_data(key, adata, loadings, rows)
            loadings = loadings[rows]
            ylabel = f"Top {top_lr_fraction:.0%} LR pair"
        else:
            _source_data(key, adata, loadings)
        order = common.dominant_order(loadings) if ordered else np.arange(len(loadings))
        if key == "A":
            groups = groups.loc[adata.obs_names]
            order = common.group_conditions(groups, order, strips)
        image = common.plot_factor_heatmap(ax, loadings[order], "CP component")
        if ordered:
            ax.set_yticks([])
            ylabel += f"s (n={len(loadings)})"
        else:
            ax.set_yticks([t - 1 for t in eigenstate_ticks], eigenstate_ticks)
        ax.set_ylabel(ylabel)
        if key == "A":
            common.condition_strips(ax, groups.iloc[order], strips, brackets=compact)
        position = np.argsort(order)  # row index -> drawn row
        common.label_heatmap_cells(
            ax,
            [(position[r], c, name) for r, (c, name) in labeled.items()],
            loadings.shape,
            clear=compact,
        )
    start = common.text_right_edge(axs[-1]) if compact else 1.0
    common.colorbar(axs[-1], image, "Loading / max |loading|", start=start)


def _source_data(key: str, adata, loadings, shown=None) -> None:
    """Factor matrix `key`'s normalized loadings, one row per condition,
    eigenstate, or LR pair (every one, with whether it's shown)."""
    ylabel = MODES[key][0]
    labels = {"A": adata.obs_names, "D": np.asarray(adata.uns["lr_pairs"])}
    df = pd.DataFrame(
        loadings,
        columns=[f"Component {c}" for c in range(1, loadings.shape[1] + 1)],
    )
    df.insert(0, ylabel, labels.get(key, np.arange(1, len(df) + 1)))
    if shown is not None:
        df["Shown"] = np.isin(np.arange(len(df)), shown)
    common.source_data(
        f"cp_{ylabel.lower().replace(' ', '_')}_loadings",
        df,
        f"CP factor matrix {key}: each {ylabel.lower().replace('lr', 'LR')}'s "
        "loading on every CP "
        "component, divided by the component's largest |loading| in this matrix."
        + (" 'Shown' marks the LR pairs drawn." if shown is not None else ""),
        part=list(MODES).index(key),
    )


def _shown_lr_pairs(
    adata, loadings, top_lr_fraction: float
) -> tuple[np.ndarray, dict[int, tuple]]:
    """Rows of D shown (the top `top_lr_fraction` by largest normalized
    loading, plus the top pairs), and {shown row: (0-based component, pair
    name)} for the top pairs."""
    names = pd.Series(np.asarray(adata.uns["lr_pairs"]).astype(str))
    top = _top_pairs(adata.uns["D"], names)
    strongest = np.abs(loadings).max(axis=1)
    n_top = round(top_lr_fraction * len(loadings))
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
