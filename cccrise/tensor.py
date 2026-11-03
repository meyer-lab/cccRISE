import anndata
import pandas as pd
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms

from .ccc_rise import (
    calc_communication_score_pseudobulk,
    pseudobulk_cp_decomposition,
    save_ccc_rise_results,
)
from .import_data import add_cond_idxs, import_ligand_receptor_pairs
from .utils import pseudobulk_X


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
