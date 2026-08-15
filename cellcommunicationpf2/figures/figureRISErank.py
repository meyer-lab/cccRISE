"""
Provisional figure(s) for RISE rank-selection criteria (PLAN_RISE.md).

Draws on cached sweep CSVs produced by `cellcommunicationpf2.rank_selection`.
Not wired into the manuscript; provisional name so as not to collide with the
existing figureS1a_b / figureS5f_g numbering.
"""

import pandas as pd

from ..import_data import import_alad, import_balf_covid
from ..rank_selection import (
    bicv_reconstruction_error,
    bootstrap_replication,
    degeneracy_diagnostics,
    leave_one_sample_out_r2x,
    nested_rank_subspace_stability,
    permutation_null_r2x_sweep,
    split_half_replication,
)
from .common import getSetup, load_or_compute_dataframe, subplotLabel

# Reduced from the plan's suggested {5,...,50} grid to fit within the compute
# budget of a single execution session (see RISE_RANK_REPORT.md compute-budget
# note): 3 points still bracketing the published rank (35 COVID / ALAD TBD).
COARSE_RANKS = [10, 25, 40]
FINE_RANKS = [10, 25, 40]


def _compute_covid_loo(ranks=None, n_folds=2, random_state=0) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_balf_covid()
    rows = []
    for r in ranks:
        df = leave_one_sample_out_r2x(
            X,
            rank=r,
            condition_column="sample",
            n_iter_max=100,
            tol=1e-6,
            random_state=random_state,
            n_folds=n_folds,
        )
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _compute_alad_loo(ranks=None, n_folds=2, random_state=0) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_alad()
    rows = []
    for r in ranks:
        df = leave_one_sample_out_r2x(
            X,
            rank=r,
            condition_column="dsco_id",
            n_iter_max=100,
            tol=1e-6,
            random_state=random_state,
            n_folds=n_folds,
        )
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _compute_covid_bicv(ranks=None, n_folds=1) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_balf_covid()
    rows = []
    for r in ranks:
        for fold in range(n_folds):
            res = bicv_reconstruction_error(
                X,
                rank=r,
                condition_column="sample",
                gene_test_frac=0.1,
                n_iter_max=100,
                tol=1e-6,
                random_state=fold,
            )
            res["Fold"] = fold
            rows.append(res)
    return pd.DataFrame(rows)


def _compute_alad_bicv(ranks=None, n_folds=1) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_alad()
    rows = []
    for r in ranks:
        for fold in range(n_folds):
            res = bicv_reconstruction_error(
                X,
                rank=r,
                condition_column="dsco_id",
                gene_test_frac=0.1,
                n_iter_max=100,
                tol=1e-6,
                random_state=fold,
            )
            res["Fold"] = fold
            rows.append(res)
    return pd.DataFrame(rows)


def _compute_covid_permutation_null(ranks=None, n_null_replicates=1) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_balf_covid()
    return permutation_null_r2x_sweep(
        X,
        ranks=ranks,
        condition_column="sample",
        n_null_replicates=n_null_replicates,
        n_iter_max=100,
        tol=1e-6,
        random_state=0,
    )


def _compute_alad_permutation_null(ranks=None, n_null_replicates=1) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_alad()
    return permutation_null_r2x_sweep(
        X,
        ranks=ranks,
        condition_column="dsco_id",
        n_null_replicates=n_null_replicates,
        n_iter_max=100,
        tol=1e-6,
        random_state=0,
    )


def _compute_covid_split_half(ranks=None, n_splits=2) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_balf_covid()
    return split_half_replication(
        X,
        ranks=ranks,
        condition_column="sample",
        n_splits=n_splits,
        n_iter_max=100,
        tol=1e-6,
        random_state=0,
    )


def _compute_alad_split_half(ranks=None, n_splits=2) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_alad()
    return split_half_replication(
        X,
        ranks=ranks,
        condition_column="dsco_id",
        n_splits=n_splits,
        n_iter_max=100,
        tol=1e-6,
        random_state=0,
    )


def _compute_covid_bootstrap_replication(ranks=None, n_boot=2) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_balf_covid()
    return bootstrap_replication(
        X,
        ranks=ranks,
        condition_column="sample",
        n_boot=n_boot,
        n_iter_max=100,
        tol=1e-6,
        random_state=0,
    )


def _compute_alad_bootstrap_replication(ranks=None, n_boot=2) -> pd.DataFrame:
    ranks = ranks or COARSE_RANKS
    X = import_alad()
    return bootstrap_replication(
        X,
        ranks=ranks,
        condition_column="dsco_id",
        n_boot=n_boot,
        n_iter_max=100,
        tol=1e-6,
        random_state=0,
    )


def _compute_covid_nested_subspace(ranks=None) -> pd.DataFrame:
    ranks = ranks or FINE_RANKS
    X = import_balf_covid()
    return nested_rank_subspace_stability(
        X, ranks=ranks, condition_column="sample", n_iter_max=100, tol=1e-6
    )


def _compute_alad_nested_subspace(ranks=None) -> pd.DataFrame:
    ranks = ranks or FINE_RANKS
    X = import_alad()
    return nested_rank_subspace_stability(
        X, ranks=ranks, condition_column="dsco_id", n_iter_max=100, tol=1e-6
    )


def _compute_covid_degeneracy(ranks=None, n_inits=2) -> pd.DataFrame:
    ranks = ranks or FINE_RANKS
    X = import_balf_covid()
    return degeneracy_diagnostics(
        X,
        ranks=ranks,
        condition_column="sample",
        n_inits=n_inits,
        n_iter_max=100,
        tol=1e-6,
    )


def _compute_alad_degeneracy(ranks=None, n_inits=2) -> pd.DataFrame:
    ranks = ranks or FINE_RANKS
    X = import_alad()
    return degeneracy_diagnostics(
        X,
        ranks=ranks,
        condition_column="dsco_id",
        n_inits=n_inits,
        n_iter_max=100,
        tol=1e-6,
    )


def makeFigure():
    ax, f = getSetup((12, 9), (3, 3))
    subplotLabel(ax)

    bicv = load_or_compute_dataframe("rise_rank_bicv_covid.csv", _compute_covid_bicv)
    perm = load_or_compute_dataframe(
        "rise_rank_permutation_null_covid.csv", _compute_covid_permutation_null
    )
    splithalf = load_or_compute_dataframe(
        "rise_rank_splithalf_covid.csv", _compute_covid_split_half
    )
    boot = load_or_compute_dataframe(
        "rise_rank_bootstrap_replication_covid.csv",
        _compute_covid_bootstrap_replication,
    )
    degen = load_or_compute_dataframe(
        "rise_rank_degeneracy_covid.csv", _compute_covid_degeneracy
    )
    nested = load_or_compute_dataframe(
        "rise_rank_nested_subspace_covid.csv", _compute_covid_nested_subspace
    )

    bicv_g = bicv.groupby("Rank")["HeldOutBlockR2X"].agg(["mean", "std"])
    ax[0].errorbar(bicv_g.index, bicv_g["mean"], yerr=bicv_g["std"], marker="o")
    ax[0].set_title("1b: BiCV held-out R2X (COVID)")
    ax[0].set_xlabel("Rank")
    ax[0].set_ylabel("Held-out block R2X")

    for null_type, sub in perm.groupby("NullType"):
        g = sub.groupby("Rank")["R2X"].mean()
        ax[1].plot(g.index, g.values, marker="o", label=null_type)
    ax[1].set_title("3: Observed vs null R2X (COVID)")
    ax[1].set_xlabel("Rank")
    ax[1].set_ylabel("R2X")
    ax[1].legend(fontsize=6)

    for tau_col in [c for c in splithalf.columns if c.startswith("n_replicated")]:
        g = splithalf.groupby("Rank")[tau_col].mean()
        ax[2].plot(g.index, g.values, marker="o", label=tau_col)
    ax[2].set_title("4: Split-half replicated components (COVID)")
    ax[2].set_xlabel("Rank")
    ax[2].set_ylabel("# components")
    ax[2].legend(fontsize=6)

    for tau_col in [c for c in boot.columns if c.startswith("n_replicated")]:
        g = boot.groupby("Rank")[tau_col].mean()
        ax[3].plot(g.index, g.values, marker="o", label=tau_col)
    ax[3].set_title("4: Bootstrap replicated components (COVID)")
    ax[3].set_xlabel("Rank")
    ax[3].set_ylabel("# components")
    ax[3].legend(fontsize=6)

    g_splithalf_sub = splithalf.groupby("Rank")["mean_cos_principal_angle"].mean()
    g_boot_sub = boot.groupby("Rank")["mean_cos_principal_angle"].mean()
    ax[4].plot(
        g_splithalf_sub.index, g_splithalf_sub.values, marker="o", label="split-half"
    )
    ax[4].plot(g_boot_sub.index, g_boot_sub.values, marker="o", label="bootstrap")
    ax[4].set_title("5: Subspace agreement (COVID)")
    ax[4].set_xlabel("Rank")
    ax[4].set_ylabel("mean cos(principal angle)")
    ax[4].legend(fontsize=6)

    ax[5].plot(nested["RankHi"], nested["mean_cos_principal_angle"], marker="o")
    ax[5].set_title("5: Nested subspace containment r -> r+ (COVID)")
    ax[5].set_xlabel("Rank (hi)")
    ax[5].set_ylabel("mean cos(principal angle)")

    ax[6].plot(degen["Rank"], degen["MaxPairwiseAbsCos_C"], marker="o")
    ax[6].set_title("6: Max pairwise |cos| in C (COVID)")
    ax[6].set_xlabel("Rank")
    ax[6].set_ylabel("max |cos|")

    ax[7].plot(degen["Rank"], degen["CondNumber_B"], marker="o", label="cond(B)")
    ax[7].plot(degen["Rank"], degen["CondNumber_C"], marker="o", label="cond(C)")
    ax[7].set_yscale("log")
    ax[7].set_title("6: Conditioning (COVID)")
    ax[7].set_xlabel("Rank")
    ax[7].set_ylabel("condition number")
    ax[7].legend(fontsize=6)

    ax[8].plot(
        degen["Rank"], degen["InitSubspaceAgreement_mean"], marker="o", label="mean"
    )
    ax[8].plot(
        degen["Rank"], degen["InitSubspaceAgreement_min"], marker="o", label="min"
    )
    ax[8].set_title("6: Init-to-init subspace agreement (COVID)")
    ax[8].set_xlabel("Rank")
    ax[8].set_ylabel("cos(principal angle)")
    ax[8].legend(fontsize=6)

    return f
