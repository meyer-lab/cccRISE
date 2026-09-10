from typing import Any

import anndata
import numpy as np
import pandas as pd
from pacmap import PaCMAP
from parafac2.parafac2 import parafac2_nd, store_pf2
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition import parafac
from tlviz.factor_tools import factor_match_score as fms

from .ccc import calculate_r2x
from .ccc_rise import (
    calc_communication_score,
    calc_communication_score_pseudobulk,
    ccc_rise,
    pseudobulk_cp_decomposition,
    save_ccc_rise_results,
    standardize_cc_pf2,
)
from .import_data import add_cond_idxs, condition_matrices, import_ligand_receptor_pairs
from .utils import pseudobulk_X


def resample_tensor(
    interaction_tensors: np.ndarray,
    random_seed: int | None = None,
    return_indices: bool = False,
) -> Any:
    """Bootstrap the communication tensor's ligand-receptor mode."""
    rng = np.random.default_rng(random_seed)
    n_lr = interaction_tensors.shape[-1]
    indices = rng.choice(n_lr, size=n_lr, replace=True)
    resampled = interaction_tensors[..., indices]
    if return_indices:
        return resampled, indices
    return resampled


def rise_store_r2x(
    X: anndata.AnnData,
    rank: int,
    n_iter_max: int = 100,
    tolerance: float = 1e-6,
    random_state: int | None = None,
):
    """Runs RISE and stores the results."""
    pf2_out, r2x = parafac2_nd(
        X, rank=rank, random_state=random_state, tol=tolerance, n_iter_max=n_iter_max
    )
    X = store_pf2(X, pf2_out)

    return X, r2x


def calculate_fms_cpd(weightsA, factorsA, weightsB, factorsB):
    """Calculates FMS between 2 factors"""
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
    """Calculates FMS between 2 factors"""
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


def run_ccc_rise_workflow(
    adata: anndata.AnnData,
    rise_rank: int,
    lr_pairs: pd.DataFrame,
    cp_rank: int | None = None,
    condition_column: str = "sample",
    n_iter_max: int = 100,
    tol: float = 1e-3,
    random_state: int | None = None,
    complex_sep: str | None = None,
    doEmbedding: bool = True,
    svd_init: str = "svd",
) -> tuple[anndata.AnnData, float]:
    """
    Executes the complete cccRISE workflow: decomposition, standardization,
    condition factor correction, and result storage.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with expression data.
    rank : int
        The rank for the decomposition.
    lr_pairs : pd.DataFrame
        The ligand-receptor pairs used in the decomposition.
    cp_rank : int, optional
        The rank for the final CP decomposition. If None, defaults to `rank`.
    condition_column : str, default="sample"
        The column in `adata.obs` that defines the conditions.
    n_iter_max : int
        Maximum number of iterations for the decomposition.
    tol : float
        Convergence tolerance for the decomposition.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple[anndata.AnnData, float]
        A tuple containing the updated AnnData object with stored results
        and the R2X value of the decomposition.
    """
    # Ensure condition indices are present before running the model
    adata = add_cond_idxs(adata, condition_column)

    # 1. Run the CC-RISE decomposition
    results, r2x, filtered_lr_pairs = ccc_rise(
        adata,
        rise_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        cp_rank=cp_rank,
        random_state=random_state,
        complex_sep=complex_sep,
        lr_pairs=lr_pairs,
        svd_init=svd_init,
    )
    (cp_weights, factors), projections = results

    # 2. Standardize the factors for interpretability
    weights, factors = standardize_cc_pf2(cp_weights, factors)

    # Store factors in AnnData object for easy access by plotting functions
    adata.uns["A"] = factors[0]  # Condition factor
    adata.uns["B"] = factors[1]  # Sender cells factor
    adata.uns["C"] = factors[2]  # Receiver cells factor
    adata.uns["D"] = factors[3]  # LR pairs factor

    # Store the LR pairs and R2X
    adata.uns["lr_pairs"] = filtered_lr_pairs["interaction_symbol"].values
    adata.uns["r2x"] = r2x
    adata.uns["weights"] = weights
    sg_index = adata.obs["condition_unique_idxs"]
    adata.obsm["projections"] = np.zeros((adata.shape[0], rise_rank))
    for i, p in enumerate(projections):
        adata.obsm["projections"][sg_index == i, :] = p
    adata.obsm["sc_B"] = adata.obsm["projections"] @ factors[1]
    adata.obsm["rc_C"] = adata.obsm["projections"] @ factors[2]

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        adata.obsm["PaCMAP"] = pcm.fit_transform(adata.obsm["projections"])

    return adata, r2x


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
        The aggregation method for pseudobulking ('mean' or 'sum').
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


def calculate_interaction_tensor(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate interaction tensor from an AnnData object that has already been processed with RISE.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with single-cell expression and RISE projections in `adata.obsm["projections"]`.
    lr_pairs : pd.DataFrame
        DataFrame of ligand-receptor pairs.
    complex_sep : str, optional, default="&"
        Separator for multimeric complexes.
    communication_score : str, default='expression_product'
        Scoring function combining ligand/receptor eigen-state expression
        into a communication score; see
        `cccrise.ccc.combine_communication_score` for the
        available options ('expression_product', 'expression_mean',
        'expression_gmean', 'expression_hill'). Useful for checking that
        reported components are robust to the choice of scoring strategy
        (issue #139).
    hill_k : float, default=0.5
        Half-saturation constant, used only when
        `communication_score='expression_hill'`.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        A tuple of (interaction_tensor, filtered_lr_pairs) where:
        - interaction_tensor is the 4D cell-cell communication interaction tensor
          of shape (n_conditions, rise_rank, rise_rank, n_lr_pairs).
        - filtered_lr_pairs is the DataFrame of ligand-receptor pairs retained
          in the tensor, matching the order of the last tensor dimension.
    """
    assert "projections" in adata.obsm, (
        "AnnData object must have 'projections' in obsm (processed with RISE)."
    )

    sgIndex = (
        np.asarray(adata.obs["condition_unique_idxs"], dtype=int)
        if "condition_unique_idxs" in adata.obs
        else np.zeros(adata.n_obs, dtype=int)
    )

    # Project matrices
    X_list = condition_matrices(adata)
    projected_matrices = [
        adata.obsm["projections"][sgIndex == i].T @ tensor
        for i, tensor in enumerate(X_list)
    ]

    # Calculate cell-cell communication scores
    gene_names = list(adata.var_names)
    gene_means = (
        adata.var["means"].to_numpy()
        if "means" in adata.var
        else np.zeros(len(gene_names))
    )
    interaction_tensor, filtered_lr_pairs = calc_communication_score(
        projected_matrices,
        gene_names=gene_names,
        lr_pairs=lr_pairs,
        complex_sep=complex_sep,
        gene_means=gene_means,
        communication_score=communication_score,
        hill_k=hill_k,
    )

    return interaction_tensor, filtered_lr_pairs


def run_fms_r2x_analysis(
    interaction_tensor: np.ndarray,
    rank_list: list[int] | None = None,
    runs: int = 1,
    svd_init: str = "svd",
) -> pd.DataFrame:
    """Run FMS and R2X analysis across different CP ranks and bootstrap runs."""
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
        X_filtered, _ = rise_store_r2x(
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
