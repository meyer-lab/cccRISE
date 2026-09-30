"""Loaders and comparisons shared by the communication-score comparison
panels (this figure and `figureS_scoring_interpretation`).

Every method, the default included, is read from
``VALIDATION_DIR/alt_score_methods/<method>/`` as written by
``uv run cccrise-altcommscore``, which shares one RISE fit across methods.
Alternate methods' CP components are paired with the default's on the
condition and LR modes, so sender/receiver (B/C) comparisons stay an
independent check of the pairing.
"""

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ...factors import load_factors_netcdf, load_factors_to_anndata
from ...runners.datasets import (
    COMMUNICATION_SCORES,
    COMPARISON_PROVENANCE_FILENAME,
    DATASETS,
    DEFAULT_COMMUNICATION_SCORE,
    comparison_output_dir,
)
from ...runners.provenance import read_metadata_at
from ...validation.anova_stability import align_to_reference_components
from .. import common, style

DATASET = "alad"
SPEC = DATASETS[DATASET]
DEFAULT_SCORE = DEFAULT_COMMUNICATION_SCORE
ALL_SCORES = list(COMMUNICATION_SCORES)
ALT_SCORES = [m for m in ALL_SCORES if m != DEFAULT_SCORE]
SCORE_LABELS = {
    "expression_product": "Product (default)",
    "expression_mean": "Mean",
    "expression_gmean": "Geometric mean",
    "expression_hill": "Hill",
}
#: Modes CP components are paired across methods on: condition and LR.
MATCH_MODES = (0, 3)
#: K values top-K LR-pair and cell-type overlaps are computed at.
LR_TOP_KS = range(1, 20, 2)
CELLTYPE_KS = range(1, 4)
HINT = (
    f"    uv run cccrise-altcommscore --dataset {DATASET} --rise-rank <N> --cp-rank <M>"
)


@cache
def cp_factors(method: str) -> tuple[list[np.ndarray], np.ndarray]:
    """`method`'s CP factors [A, B, C, D] and the LR-pair labels of D's rows."""
    ds = load_factors_netcdf(comparison_output_dir(method) / SPEC.factors_filename)
    return [ds[f"cp_{mode}"].values for mode in "ABCD"], ds.coords["lr_pair"].values


@cache
def _raw_dataset():
    return SPEC.load()


@cache
def celltype_loadings(method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean sender (sc_B) and receiver (rc_C) loading per cell type (rows)
    and component (columns), rows in `style.CELL_TYPES` order."""
    ad = load_factors_to_anndata(
        comparison_output_dir(method) / SPEC.factors_filename,
        adata=_raw_dataset(),
        condition_column=SPEC.condition_column,
    )
    celltypes = ad.obs[SPEC.celltype_column].to_numpy()
    means = [
        pd.DataFrame(ad.obsm[key], index=celltypes).groupby(level=0).mean()
        for key in ("sc_B", "rc_C")
    ]
    order = [c for c in style.CELL_TYPES[DATASET] if c in means[0].index]
    return means[0].loc[order], means[1].loc[order]


@cache
def match_components(method: str) -> pd.DataFrame:
    """Per default-score component, `method`'s 'MatchedComponent' and its
    'MatchScore' (`align_to_reference_components` on `MATCH_MODES`)."""
    ref, _ = cp_factors(DEFAULT_SCORE)
    alt, _ = cp_factors(method)
    return align_to_reference_components(
        ref, alt, np.arange(1, ref[0].shape[1] + 1), MATCH_MODES
    )


def matched_pairs(method: str) -> list[tuple[int, int]]:
    """0-based (default component, `method` component) pairs."""
    matching = match_components(method)
    return [
        (int(c) - 1, int(m) - 1)
        for c, m in zip(
            matching["Component"], matching["MatchedComponent"], strict=True
        )
    ]


def load_method_output(filename: str, method: str) -> pd.DataFrame:
    return common.load_runner_output(
        filename, root=comparison_output_dir(method), hint=HINT
    )


def default_provenance(section: str, key: str) -> Any:
    """`key` from `section` of the default score's consolidated provenance,
    where fields shared across sections are hoisted to the top level."""
    meta = read_metadata_at(
        comparison_output_dir(DEFAULT_SCORE) / COMPARISON_PROVENANCE_FILENAME
    )
    return meta[key] if key in meta else meta[section][key]


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def top_k_jaccard(ref: pd.Series, alt: pd.Series, ks) -> list[dict]:
    """Overlap of the `k` largest-magnitude labels of `ref` and `alt`, per k."""

    def top(s, k):
        return set(s.abs().nlargest(k).index)

    return [{"K": k, "Jaccard": jaccard(top(ref, k), top(alt, k))} for k in ks]


def method_legend(fig) -> None:
    """One legend of all methods' colors, above `fig`."""
    handles = [
        Patch(color=style.COMMUNICATION_SCORE_COLORS[m], label=SCORE_LABELS[m])
        for m in ALL_SCORES
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles))


def standalone(plot, size):
    """A panel on its own figure, with the method legend."""
    fig = common.standalone(plot, size)
    method_legend(fig)
    return fig


def grouped_bar(
    ax: plt.Axes, data: pd.DataFrame, ylabel: str, sig: pd.DataFrame | None = None
) -> None:
    """One bar group per row of `data` (a component), one bar per method
    column; `sig` marks significant bars with an asterisk."""
    n_methods = data.shape[1]
    width = 0.8 / n_methods
    x = np.arange(len(data))
    for i, method in enumerate(data.columns):
        offset = x + (i - (n_methods - 1) / 2) * width
        ax.bar(
            offset,
            data[method],
            width,
            label=SCORE_LABELS[method],
            color=style.COMMUNICATION_SCORE_COLORS[method],
        )
        if sig is not None:
            for xi, value in zip(
                offset[sig[method]], data[method][sig[method]], strict=True
            ):
                ax.text(xi, value, "*", ha="center", va="bottom")
    ax.set_xticks(x, data.index)
    ax.set_xlabel("CP component")
    ax.set_ylabel(ylabel)


def draw_top_k_overlap(ax: plt.Axes, overlaps: pd.DataFrame, xlabel: str) -> None:
    """Per alternate method, top-K Jaccard overlap with the default against K
    (mean +/- SD over components; no bar when there is one row per K)."""
    for key, rows in overlaps.groupby("Method", sort=False):
        method = str(key)
        stats = rows.groupby("K")["Jaccard"].agg(["mean", "std"])
        ax.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"] if stats["std"].notna().any() else None,
            marker="o",
            capsize=2,
            label=SCORE_LABELS[method],
            color=style.COMMUNICATION_SCORE_COLORS[method],
        )
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Overlap with default (Jaccard)")
