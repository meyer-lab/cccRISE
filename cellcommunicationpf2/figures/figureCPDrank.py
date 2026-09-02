"""
Figure for CPD rank selection.

Fits RISE once per dataset to build the communication tensor, then fits CP at
every rank in `DEFAULT_CP_RANKS` to derive three panels: full-tensor fit R2X,
a component-collinearity/degeneracy screen, and the CORCONDIA core-consistency
diagnostic. Each panel's DataFrame is cached under DATA_DIR via
`load_or_compute_dataframe`, so only the first figure build pays for the CP
sweep.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..ccc_rise import cp_decomposition
from ..import_data import import_alad, import_ibdverse, import_ligand_receptor_pairs
from ..rank_selection import cp_core_consistency_sweep, cpd_component_collinearity
from ..tensor import run_rise_and_score
from .common import getSetup, load_or_compute_dataframe, subplotLabel

# Easily selectable dataset registry: (loader_func, condition_column)
DATASET_LOADERS: dict[str, tuple[Callable[[], anndata.AnnData], str]] = {
    "ibd": (import_ibdverse, "sanger_sample_id"),
    "alad": (import_alad, "dsco_id"),
}

# Pre-selected RISE ranks per dataset. IBD's is not yet pinned (#166); 25 is a
# placeholder matching ALAD's, not a settled value.
DATASET_RISE_RANKS: dict[str, int] = {
    "ibd": 25,
    "alad": 25,
}

# Change this variable to switch dataset
SELECTED_DATASET = "alad"
DEFAULT_CP_RANKS = list(range(1, 26))

N_ITER_MAX = 100
TOL = 1e-3
COMPLEX_SEP = "&"
RANDOM_STATE = 42

# Heuristics drawn by makeFigure, in panel order
ENABLED_HEURISTICS = ["fit_r2x", "collinearity", "core_consistency"]
PANELS_PER_ROW = 3


@cache
def get_interaction_tensor(dataset_name: str, rise_rank: int) -> np.ndarray:
    """Build the communication tensor every heuristic decomposes, fitting RISE once.

    Cached per (dataset, RISE rank) so several heuristics in the same figure
    build share one RISE solution.
    """
    loader, cond_col = DATASET_LOADERS[dataset_name]
    adata = loader()
    lr_pairs = import_ligand_receptor_pairs()

    _adata, interaction_tensor, _lr_symbols = run_rise_and_score(
        adata,
        rise_rank=rise_rank,
        lr_pairs=lr_pairs,
        condition_column=cond_col,
        complex_sep=COMPLEX_SEP,
        n_iter_max=N_ITER_MAX,
        tol=TOL,
        random_state=RANDOM_STATE,
    )
    return interaction_tensor


@cache
def get_cp_fits_by_rank(dataset_name: str, rise_rank: int) -> dict:
    """Fit CP once per rank in `DEFAULT_CP_RANKS`; every heuristic below reads
    from this shared fit rather than re-fitting CP itself.

    Returns
    -------
    dict[int, tuple[np.ndarray, Factors, float]]
        Maps CP rank to (weights, factors, FitR2X).
    """
    tensor = get_interaction_tensor(dataset_name, rise_rank)
    return {
        rank: cp_decomposition(
            tensor,
            rank,
            n_iter_max=N_ITER_MAX,
            tol=TOL,
            random_state=RANDOM_STATE,
        )
        for rank in DEFAULT_CP_RANKS
    }


def _factors_by_rank(dataset_name: str, rise_rank: int) -> dict:
    """The (weights, factors) half of `get_cp_fits_by_rank`, for heuristics
    that operate on factor models rather than fit quality."""
    return {
        rank: (weights, factors)
        for rank, (weights, factors, _r2x) in get_cp_fits_by_rank(
            dataset_name, rise_rank
        ).items()
    }


# =============================================================================
# Heuristic: full-tensor fit R2X
# =============================================================================


def compute_fit_r2x(dataset_name: str, rise_rank: int) -> pd.DataFrame:
    fits = get_cp_fits_by_rank(dataset_name, rise_rank)
    return pd.DataFrame(
        {"Rank": rank, "FitR2X": r2x} for rank, (_w, _f, r2x) in sorted(fits.items())
    )


def plot_fit_r2x(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    ax.plot(df["Rank"], df["FitR2X"], marker="s")
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Fit $R^2X$")
    ax.set_title(title)


# =============================================================================
# Heuristic: component-collinearity / degeneracy screen
# =============================================================================


def compute_collinearity(dataset_name: str, rise_rank: int) -> pd.DataFrame:
    return cpd_component_collinearity(_factors_by_rank(dataset_name, rise_rank))


def plot_collinearity(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    ax.plot(df["Rank"], df["MaxModesCollinear"], color="0.7", zorder=1)

    canceling = df["WorstPairCanceling"] == True
    duplicate = df["WorstPairCanceling"] == False
    none_found = df["WorstPairCanceling"].isna()

    ax.scatter(
        df.loc[none_found, "Rank"],
        df.loc[none_found, "MaxModesCollinear"],
        color="0.5",
        marker="o",
        label="no collinear pair",
        zorder=2,
    )
    ax.scatter(
        df.loc[duplicate, "Rank"],
        df.loc[duplicate, "MaxModesCollinear"],
        color="tab:blue",
        marker="o",
        label="near-duplicate pair",
        zorder=2,
    )
    ax.scatter(
        df.loc[canceling, "Rank"],
        df.loc[canceling, "MaxModesCollinear"],
        color="tab:red",
        marker="D",
        label="canceling pair",
        zorder=3,
    )

    ax.axhline(2, color="0.8", linestyle="--", linewidth=0.8)
    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Max modes worst pair is collinear in")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper left")


# =============================================================================
# Heuristic: core consistency diagnostic (CORCONDIA)
# =============================================================================


def compute_core_consistency(dataset_name: str, rise_rank: int) -> pd.DataFrame:
    tensor = get_interaction_tensor(dataset_name, rise_rank)
    factors_by_rank = _factors_by_rank(dataset_name, rise_rank)
    return cp_core_consistency_sweep(tensor, factors_by_rank)


def plot_core_consistency(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    ax.plot(df["Rank"], df["CoreConsistency"], marker="o")
    ax.axhline(90, color="tab:green", linestyle="--", linewidth=0.8, label="90%")
    ax.axhline(50, color="tab:orange", linestyle="--", linewidth=0.8, label="50%")
    # Core consistency collapses by orders of magnitude past the valid-model
    # rank; symlog keeps the 0-100% transition readable while still showing
    # the full collapse instead of one axis extreme crushing the rest flat.
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Core consistency (%)")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper right")


# =============================================================================
# Heuristic registry
# =============================================================================


@dataclass(frozen=True)
class RankHeuristic:
    label: str
    cache_name: Callable[[str, int], str]
    compute: Callable[[str, int], pd.DataFrame]
    plot: Callable[[pd.DataFrame, plt.Axes, str], None]


HEURISTICS: dict[str, RankHeuristic] = {
    "fit_r2x": RankHeuristic(
        label="CPD Fit $R^2X$",
        cache_name=lambda d, r: f"cpd_rank_fit_{d}_rise{r}.csv",
        compute=compute_fit_r2x,
        plot=plot_fit_r2x,
    ),
    "collinearity": RankHeuristic(
        label="Component Collinearity Screen",
        cache_name=lambda d, r: f"cpd_rank_collinearity_{d}_rise{r}.csv",
        compute=compute_collinearity,
        plot=plot_collinearity,
    ),
    "core_consistency": RankHeuristic(
        label="Core Consistency (CORCONDIA)",
        cache_name=lambda d, r: f"cpd_rank_core_consistency_{d}_rise{r}.csv",
        compute=compute_core_consistency,
        plot=plot_core_consistency,
    ),
}


def get_heuristic_data(
    key: str,
    dataset_name: str = SELECTED_DATASET,
    rise_rank: int | None = None,
    recompute: bool = False,
) -> pd.DataFrame:
    heuristic = HEURISTICS[key]
    if rise_rank is None:
        rise_rank = DATASET_RISE_RANKS.get(dataset_name, 25)

    return load_or_compute_dataframe(
        heuristic.cache_name(dataset_name, rise_rank),
        lambda: heuristic.compute(dataset_name, rise_rank),
        recompute=recompute,
    )


def makeFigure():
    ncols = min(PANELS_PER_ROW, len(ENABLED_HEURISTICS))
    nrows = -(-len(ENABLED_HEURISTICS) // ncols)  # ceil
    ax, f = getSetup((4 * ncols, 3.5 * nrows), (nrows, ncols))
    subplotLabel(ax)

    rise_rank = DATASET_RISE_RANKS.get(SELECTED_DATASET, 25)

    for axis, key in zip(ax, ENABLED_HEURISTICS, strict=False):
        heuristic = HEURISTICS[key]
        df = get_heuristic_data(key, SELECTED_DATASET, rise_rank)
        heuristic.plot(df, axis, f"{heuristic.label} ({SELECTED_DATASET.upper()})")

    for axis in ax[len(ENABLED_HEURISTICS) :]:
        axis.set_visible(False)

    return f
