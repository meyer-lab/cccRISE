"""
Figure for CPD (interaction-tensor) rank-selection via entry-masked
cross-validation.

Draws on cached sweep CSVs produced by `cellcommunicationpf2.rank_selection`.

**RISE rank is treated as fixed/pre-determined** for the masked-CV sweep:
those tensors come from `get_covid_data(project_cells=True)` /
`get_alad_data(project_cells=True)`, which load the cached *published* RISE
factors (rank 35 COVID, rank 15 ALAD) and project cells onto them. The joint
(RISE rank x CPD rank) held-out-imputation heatmap varies the RISE rank on a
coarse grid (RISE in {25, 35, 40}) built from raw data via
`rise_cpd_joint_heatmap`.
"""

import pandas as pd

from ..import_data import import_alad, import_balf_covid, import_ligand_receptor_pairs
from ..rank_selection import cp_masked_reconstruction_sweep, rise_cpd_joint_heatmap
from ..tensor import calculate_interaction_tensor
from .common import (
    get_alad_data,
    get_covid_data,
    getSetup,
    load_or_compute_dataframe,
    subplotLabel,
)

# Full CPD-rank grid: cheap here (small, dense tensor -- CP fits take
# single-digit seconds, unlike the RISE stage).
CPD_RANKS = list(range(1, 17))


def _covid_tensor():
    X = get_covid_data(project_cells=True)
    lr_pairs = import_ligand_receptor_pairs()
    return calculate_interaction_tensor(X, lr_pairs), X, lr_pairs


def _alad_tensor():
    X = get_alad_data(project_cells=True)
    lr_pairs = import_ligand_receptor_pairs()
    return calculate_interaction_tensor(X, lr_pairs), X, lr_pairs


# ---------------------------------------------------------------------------
# Missing-entry cross-validation (masking strategy)
# ---------------------------------------------------------------------------


# `init="svd"` is pathological on this tensor/environment for ranks >= 13:
# a single ALS iteration at rank 12 with `init="svd"` takes ~1.5s, but rank
# 13 does not return within 25s (measured directly; consistent with a known
# class of LAPACK divide-and-conquer SVD degeneracies on some BLAS backends
# for certain matrix-size/rank combinations). `init="random"` avoids this
# entirely (~0.1s/iteration at rank 16) and is therefore the default for
# every CPD-fitting function in `rank_selection.py`. `n_masks` (replicate
# count) is reduced to 3, and `n_iter_max` to 200 rather than the library
# default of 1000.
MASKED_CV_N_MASKS = 3
MASKED_CV_N_ITER_MAX = 200


def _compute_covid_masked_cv() -> pd.DataFrame:
    tensor, _X, _lr = _covid_tensor()
    return cp_masked_reconstruction_sweep(
        tensor,
        ranks=CPD_RANKS,
        n_masks=MASKED_CV_N_MASKS,
        mask_frac=0.1,
        n_iter_max=MASKED_CV_N_ITER_MAX,
        random_state=0,
    )


def _compute_alad_masked_cv() -> pd.DataFrame:
    tensor, _X, _lr = _alad_tensor()
    return cp_masked_reconstruction_sweep(
        tensor,
        ranks=CPD_RANKS,
        n_masks=MASKED_CV_N_MASKS,
        mask_frac=0.1,
        n_iter_max=MASKED_CV_N_ITER_MAX,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Joint (RISE rank x CPD rank) held-out imputation heatmap.
# Coarse grid: RISE in {25, 35, 40} (subset of the ranks already fit during
# the RISE bi-cross-validation sweep, which found a plateau across 25-40 on
# COVID and a peak near 25 on ALAD), CPD in {4, 8, 12} (brackets the
# published rank of 8). Each RISE rank requires one fresh parafac2_nd fit
# (~1-3 min); the CPD sweep on top of a fixed RISE fit is cheap, so this is
# 3 RISE fits x 3 CPD ranks x 2 masks per dataset.
# ---------------------------------------------------------------------------

JOINT_RISE_RANKS = [25, 35, 40]
JOINT_CPD_RANKS = [4, 8, 12]


def _compute_covid_joint_heatmap() -> pd.DataFrame:
    X = import_balf_covid()
    lr_pairs = import_ligand_receptor_pairs()
    return rise_cpd_joint_heatmap(
        X,
        lr_pairs,
        rise_ranks=JOINT_RISE_RANKS,
        cpd_ranks=JOINT_CPD_RANKS,
        condition_column="sample",
        mask_frac=0.1,
        n_masks=2,
        rise_n_iter_max=100,
        rise_tol=1e-6,
        cp_n_iter_max=200,
        cp_init="random",
        random_state=0,
    )


def _compute_alad_joint_heatmap() -> pd.DataFrame:
    X = import_alad()
    lr_pairs = import_ligand_receptor_pairs()
    return rise_cpd_joint_heatmap(
        X,
        lr_pairs,
        rise_ranks=JOINT_RISE_RANKS,
        cpd_ranks=JOINT_CPD_RANKS,
        condition_column="dsco_id",
        mask_frac=0.1,
        n_masks=2,
        rise_n_iter_max=100,
        rise_tol=1e-6,
        cp_n_iter_max=200,
        cp_init="random",
        random_state=0,
    )


def makeFigure():
    ax, f = getSetup((9.6, 6), (2, 2))
    subplotLabel(ax)

    masked_cv = load_or_compute_dataframe(
        "interaction_rank_masked_cv_covid.csv", _compute_covid_masked_cv
    )
    covid_joint = load_or_compute_dataframe(
        "interaction_rank_joint_heatmap_covid.csv", _compute_covid_joint_heatmap
    )

    g = masked_cv.groupby("Rank")["HeldOutR2X"].agg(["mean", "std"])
    ax[0].errorbar(g.index, g["mean"], yerr=g["std"], marker="o")
    ax[0].set_title("Masked-entry held-out R2X (COVID)")
    ax[0].set_xlabel("CPD Rank")
    ax[0].set_ylabel("Held-out R2X")

    pivot = (
        covid_joint.groupby(["RISE_Rank", "CPD_Rank"])["HeldOutR2X"].mean().unstack()
    )
    im = ax[1].imshow(pivot.values, aspect="auto", cmap="viridis")
    ax[1].set_xticks(range(len(pivot.columns)))
    ax[1].set_xticklabels(pivot.columns)
    ax[1].set_yticks(range(len(pivot.index)))
    ax[1].set_yticklabels(pivot.index)
    ax[1].set_xlabel("CPD Rank")
    ax[1].set_ylabel("RISE Rank")
    ax[1].set_title("Joint held-out R2X (COVID)")
    f.colorbar(im, ax=ax[1], fraction=0.046)

    alad_masked_cv = load_or_compute_dataframe(
        "interaction_rank_masked_cv_alad.csv", _compute_alad_masked_cv
    )
    alad_joint = load_or_compute_dataframe(
        "interaction_rank_joint_heatmap_alad.csv", _compute_alad_joint_heatmap
    )

    g2 = alad_masked_cv.groupby("Rank")["HeldOutR2X"].agg(["mean", "std"])
    ax[2].errorbar(g2.index, g2["mean"], yerr=g2["std"], marker="o", color="firebrick")
    ax[2].set_title("Masked-entry held-out R2X (ALAD)")
    ax[2].set_xlabel("CPD Rank")
    ax[2].set_ylabel("Held-out R2X")

    pivot2 = (
        alad_joint.groupby(["RISE_Rank", "CPD_Rank"])["HeldOutR2X"].mean().unstack()
    )
    im2 = ax[3].imshow(pivot2.values, aspect="auto", cmap="viridis")
    ax[3].set_xticks(range(len(pivot2.columns)))
    ax[3].set_xticklabels(pivot2.columns)
    ax[3].set_yticks(range(len(pivot2.index)))
    ax[3].set_yticklabels(pivot2.index)
    ax[3].set_xlabel("CPD Rank")
    ax[3].set_ylabel("RISE Rank")
    ax[3].set_title("Joint held-out R2X (ALAD)")
    f.colorbar(im2, ax=ax[3], fraction=0.046)

    return f
