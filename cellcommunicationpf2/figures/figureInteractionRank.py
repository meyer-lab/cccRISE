"""
Provisional figure(s) for CPD (interaction-tensor) rank-selection criteria
(PLAN_INTERACTION.md).

Draws on cached sweep CSVs produced by `cellcommunicationpf2.rank_selection`.
Not wired into the manuscript; provisional name so as not to collide with the
existing figureS1c_d / figureS5f_g numbering, or with figureRISErank.py.

**RISE rank is treated as fixed/pre-determined for Approaches A-E**: those
tensors come from `get_covid_data(project_cells=True)` /
`get_alad_data(project_cells=True)`, which load the cached *published* RISE
factors (rank 35 COVID, rank 15 ALAD) and project cells onto them. Sec. 4's
joint (RISE rank x CPD rank) held-out-imputation heatmap varies the RISE
rank on a coarse grid (RISE in {25, 35, 40}, a subset of the ranks already
validated by PLAN_RISE.md's bi-cross-validation sweep) built from raw data
via `rise_cpd_joint_heatmap`. Sec. 5 estimates cross-validated AUC of
ALAD status at each CPD rank via `cpd_rank_auc_sweep`, for reporting
alongside the unsupervised criterion only -- see INTERACTION_RANK_REPORT.md
for the compute-budget notes on both reduced grids.
"""

import numpy as np
import pandas as pd

from ..import_data import (
    add_cond_idxs,
    import_alad,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from ..rank_selection import (
    corcondia_sweep,
    cp_condition_holdout_error,
    cp_masked_reconstruction_sweep,
    cpd_component_significance,
    cpd_degeneracy_diagnostics,
    cpd_permutation_null_sweep,
    cpd_rank_auc_sweep,
    cpd_replication_sweep,
    rise_cpd_joint_heatmap,
)
from ..tensor import calculate_interaction_tensor
from .common import (
    get_alad_data,
    get_covid_data,
    getSetup,
    load_or_compute_dataframe,
    subplotLabel,
)
from .commonFuncs.plotGeneral import binarize_group, get_sample_to_group

# Full plan-suggested CPD-rank grid: cheap here (small, dense tensor -- CP
# fits take single-digit seconds, unlike the RISE stage).
CPD_RANKS = list(range(1, 17))
# Reduced grid for the more expensive multi-fit approaches (D: replication,
# E: multi-start degeneracy) -- still brackets the published rank of 8 and
# the reviewer's suggested 6-12 window.
CPD_RANKS_REDUCED = [2, 4, 8, 12, 16]


def _covid_tensor():
    X = get_covid_data(project_cells=True)
    lr_pairs = import_ligand_receptor_pairs()
    return calculate_interaction_tensor(X, lr_pairs), X, lr_pairs


def _alad_tensor():
    X = get_alad_data(project_cells=True)
    lr_pairs = import_ligand_receptor_pairs()
    return calculate_interaction_tensor(X, lr_pairs), X, lr_pairs


# ---------------------------------------------------------------------------
# Approach A - missing-entry cross-validation
# ---------------------------------------------------------------------------


# `init="svd"` is pathological on this tensor/environment for ranks >= 13:
# a single ALS iteration at rank 12 with `init="svd"` takes ~1.5s, but rank
# 13 does not return within 25s (measured directly; consistent with a known
# class of LAPACK divide-and-conquer SVD degeneracies on some BLAS backends
# for certain matrix-size/rank combinations). `init="random"` avoids this
# entirely (~0.1s/iteration at rank 16) and is therefore the default for
# every CPD-fitting function in `rank_selection.py`, not only the masked
# ones -- see INTERACTION_RANK_REPORT.md for the full benchmark. Masked/
# EM-imputation fits are additionally expensive regardless of init (the
# `svd_mask_repeats` imputation loop only applies with `init="svd"`, so
# switching to random also removes that cost); `n_masks` (replicate count)
# is reduced from the plan's implicit "several" to 3, and `n_iter_max` to
# 200 rather than the library default of 1000, both documented in
# INTERACTION_RANK_REPORT.md.
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


def _compute_covid_condition_holdout() -> pd.DataFrame:
    tensor, _X, _lr = _covid_tensor()
    rows = []
    for rank in (4, 8, 12):
        df = cp_condition_holdout_error(
            tensor, rank=rank, n_iter_max=MASKED_CV_N_ITER_MAX, random_state=0
        )
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _compute_alad_condition_holdout() -> pd.DataFrame:
    tensor, _X, _lr = _alad_tensor()
    # ALAD has 47 conditions; a full per-condition refit at every rank is
    # expensive relative to the rest of this sweep, so hold out a fixed
    # random subset of 10 conditions rather than all 47 (compute-budget
    # reduction, documented in INTERACTION_RANK_REPORT.md).
    rng = np.random.default_rng(0)
    subset = rng.choice(tensor.shape[0], size=10, replace=False)
    rows = []
    for rank in (4, 8, 12):
        df = cp_condition_holdout_error(
            tensor,
            rank=rank,
            n_iter_max=MASKED_CV_N_ITER_MAX,
            random_state=0,
            conditions=subset,
        )
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Approach B - permutation null / parallel analysis, + significance sketch
# ---------------------------------------------------------------------------


# Permutation-null tensors have no real communication structure to converge
# to, so ALS on them tends to run close to n_iter_max rather than stopping
# early on `tol` -- capped at 200 (matching MASKED_CV_N_ITER_MAX) and on the
# CPD_RANKS_REDUCED grid rather than the full 16-point one, per the same
# compute-budget reasoning as Approach A (benchmarked: full CPD_RANKS at
# n_iter_max=1000 did not finish within 10 minutes for COVID alone).
PERMUTATION_NULL_N_ITER_MAX = 200


def _compute_covid_permutation_null() -> pd.DataFrame:
    tensor, _X, _lr = _covid_tensor()
    return cpd_permutation_null_sweep(
        tensor,
        ranks=CPD_RANKS_REDUCED,
        n_null_replicates=2,
        n_iter_max=PERMUTATION_NULL_N_ITER_MAX,
        random_state=0,
    )


def _compute_alad_permutation_null() -> pd.DataFrame:
    tensor, _X, _lr = _alad_tensor()
    return cpd_permutation_null_sweep(
        tensor,
        ranks=CPD_RANKS_REDUCED,
        n_null_replicates=2,
        n_iter_max=PERMUTATION_NULL_N_ITER_MAX,
        random_state=0,
    )


def _compute_covid_component_significance() -> pd.DataFrame:
    tensor, _X, _lr = _covid_tensor()
    return cpd_component_significance(
        tensor,
        rank=8,
        n_null=20,
        n_iter_max=PERMUTATION_NULL_N_ITER_MAX,
        random_state=0,
    )


def _compute_alad_component_significance() -> pd.DataFrame:
    tensor, _X, _lr = _alad_tensor()
    return cpd_component_significance(
        tensor,
        rank=8,
        n_null=20,
        n_iter_max=PERMUTATION_NULL_N_ITER_MAX,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Approach C - CORCONDIA
# ---------------------------------------------------------------------------


def _compute_covid_corcondia() -> pd.DataFrame:
    tensor, _X, _lr = _covid_tensor()
    return corcondia_sweep(tensor, ranks=CPD_RANKS, n_iter_max=200, random_state=0)


def _compute_alad_corcondia() -> pd.DataFrame:
    tensor, _X, _lr = _alad_tensor()
    return corcondia_sweep(tensor, ranks=CPD_RANKS, n_iter_max=200, random_state=0)


# ---------------------------------------------------------------------------
# Approach D - replicable component counts, multiple resampling axes
# ---------------------------------------------------------------------------


def _compute_covid_replication(axis: str) -> pd.DataFrame:
    tensor, _X, _lr = _covid_tensor()
    return cpd_replication_sweep(
        tensor,
        ranks=CPD_RANKS_REDUCED,
        resample_axis=axis,
        n_boot=3,
        n_iter_max=200,
        random_state=0,
    )


def _compute_alad_replication(axis: str) -> pd.DataFrame:
    tensor, _X, _lr = _alad_tensor()
    return cpd_replication_sweep(
        tensor,
        ranks=CPD_RANKS_REDUCED,
        resample_axis=axis,
        n_boot=3,
        n_iter_max=200,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Approach E - degeneracy / multi-start diagnostics
# ---------------------------------------------------------------------------


def _compute_covid_degeneracy() -> pd.DataFrame:
    tensor, _X, _lr = _covid_tensor()
    # figureS1c_d.py uses svd_init="random" for COVID.
    return cpd_degeneracy_diagnostics(
        tensor,
        ranks=CPD_RANKS_REDUCED,
        n_inits=3,
        n_iter_max=200,
        init="random",
        random_state=0,
    )


def _compute_alad_degeneracy() -> pd.DataFrame:
    tensor, _X, _lr = _alad_tensor()
    # figureS5f_g.py also uses svd_init="random" for ALAD -- same init as
    # COVID, so any COVID/ALAD discrepancy is not explained by an init
    # mismatch (see PLAN_INTERACTION.md Approach E note).
    return cpd_degeneracy_diagnostics(
        tensor,
        ranks=CPD_RANKS_REDUCED,
        n_inits=3,
        n_iter_max=200,
        init="random",
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Section 4 - joint (RISE rank x CPD rank) held-out imputation heatmap.
# Coarse grid: RISE in {25, 35, 40} (subset of the ranks already fit during
# PLAN_RISE.md's bi-cross-validation sweep, which found a plateau across
# 25-40 on COVID and a peak near 25 on ALAD), CPD in {4, 8, 12} (brackets
# the published rank of 8). Each RISE rank requires one fresh parafac2_nd
# fit (~1-3 min); the CPD sweep on top of a fixed RISE fit is cheap, so this
# is 3 RISE fits x 3 CPD ranks x 2 masks per dataset -- a deliberate
# reduction from the plan's suggested RISE in {20,30,35,40,50} x CPD in
# {4,6,8,10,12,16}, documented in INTERACTION_RANK_REPORT.md.
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


# ---------------------------------------------------------------------------
# Section 5 - consequence for the ALAD predictive analysis. AUC at each CPD
# rank, computed for reporting only, alongside (never in place of) the
# unsupervised criterion -- see INTERACTION_RANK_REPORT.md Sec. 5.
# ---------------------------------------------------------------------------


def _compute_alad_auc_sweep() -> pd.DataFrame:
    tensor, X, _lr = _alad_tensor()
    X = add_cond_idxs(X, "dsco_id")
    sample_to_group = get_sample_to_group(X, "dsco_id", "ALADstatus")
    y = binarize_group(sample_to_group, "control", "control", "alad").to_numpy()
    return cpd_rank_auc_sweep(
        tensor,
        y,
        ranks=CPD_RANKS,
        n_iter_max=200,
        random_state=0,
        n_splits=5,
        n_repeats=5,
    )


def makeFigure():
    ax, f = getSetup((19.2, 12), (3, 5))
    subplotLabel(ax)

    masked_cv = load_or_compute_dataframe(
        "interaction_rank_masked_cv_covid.csv", _compute_covid_masked_cv
    )
    perm = load_or_compute_dataframe(
        "interaction_rank_permutation_null_covid.csv", _compute_covid_permutation_null
    )
    corcondia_df = load_or_compute_dataframe(
        "interaction_rank_corcondia_covid.csv", _compute_covid_corcondia
    )
    repl_lr = load_or_compute_dataframe(
        "interaction_rank_replication_lr_covid.csv",
        _compute_covid_replication,
        "lr",
    )
    repl_cond = load_or_compute_dataframe(
        "interaction_rank_replication_conditions_covid.csv",
        _compute_covid_replication,
        "conditions",
    )
    degen = load_or_compute_dataframe(
        "interaction_rank_degeneracy_covid.csv", _compute_covid_degeneracy
    )
    covid_joint = load_or_compute_dataframe(
        "interaction_rank_joint_heatmap_covid.csv", _compute_covid_joint_heatmap
    )

    g = masked_cv.groupby("Rank")["HeldOutR2X"].agg(["mean", "std"])
    ax[0].errorbar(g.index, g["mean"], yerr=g["std"], marker="o")
    ax[0].set_title("A: Masked-entry held-out R2X (COVID)")
    ax[0].set_xlabel("CPD Rank")
    ax[0].set_ylabel("Held-out R2X")

    for null_type, sub in perm.groupby("NullType"):
        gg = sub.groupby("Rank")["R2X"].mean()
        ax[1].plot(gg.index, gg.values, marker="o", label=null_type)
    ax[1].set_title("B: Observed vs null R2X (COVID)")
    ax[1].set_xlabel("CPD Rank")
    ax[1].set_ylabel("R2X")
    ax[1].legend(fontsize=6)

    ax[2].plot(corcondia_df["Rank"], corcondia_df["CORCONDIA"], marker="o")
    ax[2].set_title("C: CORCONDIA (COVID)")
    ax[2].set_xlabel("CPD Rank")
    ax[2].set_ylabel("Core consistency")

    for tau_col in [c for c in repl_lr.columns if c.startswith("n_replicated")]:
        gg = repl_lr.groupby("Rank")[tau_col].mean()
        ax[3].plot(gg.index, gg.values, marker="o", label=f"LR-boot {tau_col}")
    for tau_col in [c for c in repl_cond.columns if c.startswith("n_replicated")]:
        gg = repl_cond.groupby("Rank")[tau_col].mean()
        ax[3].plot(
            gg.index,
            gg.values,
            marker="s",
            linestyle="--",
            label=f"cond-boot {tau_col}",
        )
    ax[3].set_title("D: Replicated components (COVID)")
    ax[3].set_xlabel("CPD Rank")
    ax[3].set_ylabel("# components")
    ax[3].legend(fontsize=5)

    ax[4].plot(degen["Rank"], degen["MaxPairwiseTripleCongruence"], marker="o")
    ax[4].set_title("E: Max triple congruence (COVID)")
    ax[4].set_xlabel("CPD Rank")
    ax[4].set_ylabel("max congruence")

    pivot = (
        covid_joint.groupby(["RISE_Rank", "CPD_Rank"])["HeldOutR2X"].mean().unstack()
    )
    im = ax[5].imshow(pivot.values, aspect="auto", cmap="viridis")
    ax[5].set_xticks(range(len(pivot.columns)))
    ax[5].set_xticklabels(pivot.columns)
    ax[5].set_yticks(range(len(pivot.index)))
    ax[5].set_yticklabels(pivot.index)
    ax[5].set_xlabel("CPD Rank")
    ax[5].set_ylabel("RISE Rank")
    ax[5].set_title("Sec 4: Joint held-out R2X (COVID)")
    f.colorbar(im, ax=ax[5], fraction=0.046)

    alad_masked_cv = load_or_compute_dataframe(
        "interaction_rank_masked_cv_alad.csv", _compute_alad_masked_cv
    )
    alad_perm = load_or_compute_dataframe(
        "interaction_rank_permutation_null_alad.csv", _compute_alad_permutation_null
    )
    alad_corcondia = load_or_compute_dataframe(
        "interaction_rank_corcondia_alad.csv", _compute_alad_corcondia
    )
    alad_repl_lr = load_or_compute_dataframe(
        "interaction_rank_replication_lr_alad.csv",
        _compute_alad_replication,
        "lr",
    )
    alad_repl_cond = load_or_compute_dataframe(
        "interaction_rank_replication_conditions_alad.csv",
        _compute_alad_replication,
        "conditions",
    )
    alad_degen = load_or_compute_dataframe(
        "interaction_rank_degeneracy_alad.csv", _compute_alad_degeneracy
    )
    alad_auc = load_or_compute_dataframe(
        "interaction_rank_auc_sweep_alad.csv", _compute_alad_auc_sweep
    )
    alad_joint = load_or_compute_dataframe(
        "interaction_rank_joint_heatmap_alad.csv", _compute_alad_joint_heatmap
    )

    g2 = alad_masked_cv.groupby("Rank")["HeldOutR2X"].agg(["mean", "std"])
    ax[6].errorbar(g2.index, g2["mean"], yerr=g2["std"], marker="o", color="firebrick")
    ax[6].set_title("A: Masked-entry held-out R2X (ALAD)")
    ax[6].set_xlabel("CPD Rank")
    ax[6].set_ylabel("Held-out R2X")

    for null_type, sub in alad_perm.groupby("NullType"):
        gg = sub.groupby("Rank")["R2X"].mean()
        ax[7].plot(gg.index, gg.values, marker="o", label=null_type)
    ax[7].set_title("B: Observed vs null R2X (ALAD)")
    ax[7].set_xlabel("CPD Rank")
    ax[7].set_ylabel("R2X")
    ax[7].legend(fontsize=6)

    ax[8].plot(
        alad_corcondia["Rank"],
        alad_corcondia["CORCONDIA"],
        marker="o",
        color="firebrick",
    )
    ax[8].set_title("C: CORCONDIA (ALAD)")
    ax[8].set_xlabel("CPD Rank")
    ax[8].set_ylabel("Core consistency")

    for tau_col in [c for c in alad_repl_lr.columns if c.startswith("n_replicated")]:
        gg = alad_repl_lr.groupby("Rank")[tau_col].mean()
        ax[9].plot(
            gg.index,
            gg.values,
            marker="o",
            color="firebrick",
            label=f"LR-boot {tau_col}",
        )
    for tau_col in [c for c in alad_repl_cond.columns if c.startswith("n_replicated")]:
        gg = alad_repl_cond.groupby("Rank")[tau_col].mean()
        ax[9].plot(
            gg.index,
            gg.values,
            marker="s",
            linestyle="--",
            color="darkred",
            label=f"cond-boot {tau_col}",
        )
    ax[9].set_title("D: Replicated components (ALAD)")
    ax[9].set_xlabel("CPD Rank")
    ax[9].set_ylabel("# components")
    ax[9].legend(fontsize=5)

    ax[10].plot(
        alad_degen["Rank"],
        alad_degen["MaxPairwiseTripleCongruence"],
        marker="o",
        color="firebrick",
    )
    ax[10].set_title("E: Max triple congruence (ALAD)")
    ax[10].set_xlabel("CPD Rank")
    ax[10].set_ylabel("max congruence")

    ax[11].errorbar(
        alad_auc["Rank"],
        alad_auc["AUC_mean"],
        yerr=alad_auc["AUC_std"],
        marker="o",
        color="firebrick",
    )
    ax[11].set_title("Sec 5: ALAD-status AUC vs CPD rank")
    ax[11].set_xlabel("CPD Rank")
    ax[11].set_ylabel("CV ROC-AUC")

    pivot2 = (
        alad_joint.groupby(["RISE_Rank", "CPD_Rank"])["HeldOutR2X"].mean().unstack()
    )
    im2 = ax[12].imshow(pivot2.values, aspect="auto", cmap="viridis")
    ax[12].set_xticks(range(len(pivot2.columns)))
    ax[12].set_xticklabels(pivot2.columns)
    ax[12].set_yticks(range(len(pivot2.index)))
    ax[12].set_yticklabels(pivot2.index)
    ax[12].set_xlabel("CPD Rank")
    ax[12].set_ylabel("RISE Rank")
    ax[12].set_title("Sec 4: Joint held-out R2X (ALAD)")
    f.colorbar(im2, ax=ax[12], fraction=0.046)

    for i in (13, 14):
        ax[i].axis("off")

    return f
