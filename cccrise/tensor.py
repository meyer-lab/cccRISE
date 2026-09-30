import anndata
import numpy as np
import pandas as pd
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition import parafac
from tlviz.factor_tools import factor_match_score as fms

from .ccc_rise import (
    calc_communication_score_pseudobulk,
    calculate_interaction_tensor,
    fit_rise,
    pseudobulk_cp_decomposition,
    save_ccc_rise_results,
)
from .import_data import add_cond_idxs, import_ligand_receptor_pairs
from .rank_selection import calculate_r2x
from .utils import pseudobulk_X, resample_tensor


def calculate_fms_cpd(weightsA, factorsA, weightsB, factorsB):
    """FMS between two CP decompositions, ignoring weights and the LR mode."""
    A_CP = CPTensor(
        (
            weightsA,
            factorsA,
        )
    )
    B_CP = CPTensor(
        (
            weightsB,
            factorsB,
        )
    )
    return fms(A_CP, B_CP, consider_weights=False, skip_mode=3)


def calculate_fms_rise(A: anndata.AnnData, B: anndata.AnnData):
    """FMS between two RISE fits, ignoring weights and the B (`Pf2_B`) mode."""
    factors = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.varm["Pf2_C"]]
    A_CP = CPTensor(
        (
            A.uns["Pf2_weights"],
            factors,
        )
    )

    factors = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.varm["Pf2_C"]]
    B_CP = CPTensor(
        (
            B.uns["Pf2_weights"],
            factors,
        )
    )
    return fms(A_CP, B_CP, consider_weights=False, skip_mode=1)


def run_pseudobulk_workflow(
    adata: anndata.AnnData,
    cp_rank: int = 8,
    lr_pairs: pd.DataFrame | None = None,
    condition_column: str = "sample",
    groupby: str = "celltype",
    type: str = "mean",
    n_iter_max: int = 1000,
    tol: float = 1e-11,
    random_state: int | None = None,
    complex_sep: str | None = "&",
) -> tuple[anndata.AnnData, float]:
    """
    Executes the complete pseudobulk CP decomposition workflow:
    aggregates single cells into pseudobulk matrices per (condition, groupby),
    computes the 4D cell-cell communication tensor, fits CP decomposition,
    and stores factors, weights, and LR pairs in the AnnData object's `.uns`.
    The CP fit is unconstrained (signed), not non-negative.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with single-cell expression data.
    cp_rank : int, default=8
        The rank for the CP decomposition.
    lr_pairs : pd.DataFrame | None, optional
        The ligand-receptor pairs used in the decomposition. If None, loaded automatically.
    condition_column : str, default="sample"
        The column in `adata.obs` that defines the conditions/samples.
    groupby : str, default="celltype"
        The column in `adata.obs` that defines the cell types.
    type : str, default="mean"
        The aggregation method for pseudobulking ('mean' or 'fraction'); any
        other value yields zero expression (see `utils.pseudobulk_X`).
    n_iter_max : int, default=1000
        Maximum number of iterations for CP decomposition.
    tol : float, default=1e-11
        Convergence tolerance for CP decomposition.
    random_state : int | None, optional
        Random seed for reproducibility.
    complex_sep : str | None, default="&"
        Symbol that separates protein subunits in multimeric complexes.

    Returns
    -------
    tuple[anndata.AnnData, float]
        A tuple containing the updated AnnData object with stored pseudobulk results
        in `.uns` and the R2X value.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    adata = add_cond_idxs(adata, condition_column)
    appended_pseudobulk = pseudobulk_X(
        adata, condition_name=condition_column, groupby=groupby, type=type
    )
    interaction_tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(
        appended_pseudobulk, lr_pairs=lr_pairs, complex_sep=complex_sep
    )
    cpd_weights, cpd_factors, r2x = pseudobulk_cp_decomposition(
        interaction_tensor,
        cp_rank=cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
    )
    adata = save_ccc_rise_results(adata, cpd_factors, cpd_weights, filtered_lr_pairs)
    adata.uns["r2x"] = r2x
    return adata, r2x


def run_fms_r2x_analysis(
    interaction_tensor: np.ndarray,
    rank_list: list[int] | None = None,
    runs: int = 1,
    svd_init: str = "svd",
) -> pd.DataFrame:
    """
    Run FMS and R2X analysis across different CP ranks and bootstrap runs.

    **Legacy function**
    TODO: Remove this function in place of @cccrise/rank_selection.py
    """
    if rank_list is None:
        rank_list = list(range(1, 4))

    rows = []
    for j in rank_list:
        cp_weights, cp_factors = parafac(
            tensor=interaction_tensor,
            rank=j,
            n_iter_max=1000,
            init=svd_init,
            normalize_factors=True,
            linesearch=True,
        )

        for i in range(runs):
            r2x = calculate_r2x(cp_weights, cp_factors, interaction_tensor)
            cp_boot_weights, cp_boot_factors = parafac(
                tensor=resample_tensor(interaction_tensor),
                rank=j,
                n_iter_max=1000,
                init=svd_init,
                normalize_factors=True,
                linesearch=True,
            )
            fms_score = calculate_fms_cpd(
                cp_weights, cp_factors, cp_boot_weights, cp_boot_factors
            )
            rows.append({"Run": i, "Component": j, "FMS": fms_score, "R2X": r2x})
            print(f"Run {i}, Component {j}, FMS: {fms_score}, R2X: {r2x}")

    return pd.DataFrame(rows)


def run_fms_r2x_data_percentage_analysis(
    X_filtered: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    rise_rank: int,
    cp_rank: int,
    percentage_list: list[int] | None = None,
    runs: int = 1,
    svd_init: str = "svd",
) -> pd.DataFrame:
    if percentage_list is None:
        percentage_list = list(range(100, 45, -5))

    if "projections" not in X_filtered.obsm:
        X_filtered, _ = fit_rise(
            X_filtered, rank=rise_rank, n_iter_max=1000, tolerance=1e-9
        )

    # 0) Reference on FULL data (no bootstrapping here)
    full_tensor, _ = calculate_interaction_tensor(X_filtered, lr_pairs)
    w_full, f_full = parafac(
        tensor=full_tensor,
        rank=cp_rank,
        n_iter_max=1000,
        init=svd_init,
        normalize_factors=True,
    )

    rows = []
    n_cells = X_filtered.n_obs
    for run_idx in range(runs):
        for pct in percentage_list:
            # 1) Subsample WITHOUT replacement
            if pct < 100:
                n_keep = int(n_cells * pct / 100)
                keep_idx = np.random.choice(n_cells, n_keep, replace=False)
                X_sub = X_filtered[keep_idx]
            else:
                X_sub = X_filtered

            # 2) Tensor on subsample
            sub_tensor, _ = calculate_interaction_tensor(X_sub, lr_pairs)

            # 3) CP on subsample
            w_sub, f_sub = parafac(
                tensor=sub_tensor,
                rank=cp_rank,
                n_iter_max=1000,
                init=svd_init,
                normalize_factors=True,
            )

            # 4) R²X on subsample
            r2x_sub = calculate_r2x(w_sub, f_sub, sub_tensor)

            # 5) FMS vs FULL reference (NOT vs bootstrap)
            fms_vs_full = calculate_fms_cpd(w_full, f_full, w_sub, f_sub)

            rows.append(
                {
                    "Run": run_idx,
                    "Data_Percentage": pct,
                    "FMS": fms_vs_full,
                    "R2X": r2x_sub,
                }
            )

    return pd.DataFrame(rows)
