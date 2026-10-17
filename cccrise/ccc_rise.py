from typing import Any

import anndata
import numpy as np
import pandas as pd
from pacmap import PaCMAP
from parafac2.parafac2 import parafac2_nd, store_pf2
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.decomposition import parafac

from .ccc import (
    build_context_ccc_tensor,
    get_genes_from_complexes,
)
from .import_data import (
    add_cond_idxs,
    condition_matrices,
    import_ligand_receptor_pairs,
)
from .rank_selection import (
    Factors,
    calculate_r2x,
    candelinc_compress,
    candelinc_r2x,
    cp_core_consistency,
    cpd_component_collinearity,
)

#: PARAFAC2 options every CCC-RISE fit uses, including `bicv` in `run_rise_rank`.
RISE_FIT_OPTIONS: dict[str, Any] = {"normalize_slices": True, "fix_B_identity": True}
#: Compression options for runs with `compress=True`. RISE's are parafac2's
#: default, passed explicitly so `fit_rise` and `bicv` (`compression_kwarg`,
#: which must be non-empty for `bicv` to compress once per fold) provably match.
RISE_COMPRESSION_OPTIONS: dict[str, Any] = {"n_power_iter": 20}
CP_COMPRESSION_OPTIONS: dict[str, Any] = {"oversample": 5}
#: CP options every CCC-RISE fit uses, including the rank sweep in `run_cpd_rank`.
CP_FIT_OPTIONS: dict[str, Any] = {
    "init": "svd",
    "svd": "randomized_svd",
    "normalize_factors": True,
}


def _matrices_to_rnaseq_dataframes(
    projected_matrices: list[np.ndarray],
    gene_names: list[str],
    gene_means: np.ndarray | None,
) -> list[pd.DataFrame]:
    """Optionally re-add gene means, then convert projected matrices to
    genes x cells DataFrames for `build_context_ccc_tensor`."""
    if gene_means is not None:
        means_row = np.asarray(gene_means).reshape(1, -1)
        projected_matrices = [
            np.clip(matrix + means_row, 0.0, None) for matrix in projected_matrices
        ]

    return [
        pd.DataFrame(
            matrix.T,
            index=pd.Index(gene_names),
            columns=pd.Index([f"rank_{j}" for j in range(matrix.shape[0])]),
        )
        for matrix in projected_matrices
    ]


def _rename_lr_columns(lr_pairs: pd.DataFrame) -> pd.DataFrame:
    """Rename ligand/receptor columns to Cell2Cell's A/B convention, if present."""
    if "ligand" in lr_pairs.columns and "receptor" in lr_pairs.columns:
        return lr_pairs.rename(columns={"ligand": "A", "receptor": "B"})
    return lr_pairs


def _apply_complex_aggregation(
    rnaseq_matrices: list[pd.DataFrame],
    lr_pairs_renamed: pd.DataFrame,
    complex_sep: str | None,
    complex_agg_method: str,
    verbose: bool,
) -> list[pd.DataFrame]:
    """Aggregate multimeric complex subunits into single rows, if requested."""
    if complex_sep is None:
        return [df.copy() for df in rnaseq_matrices]

    if verbose:
        print("Getting expression values for protein complexes")
    _, _, _, _, complexes = get_genes_from_complexes(
        ppi_data=lr_pairs_renamed,
        complex_sep=complex_sep,
        interaction_columns=("A", "B"),
    )
    return [
        add_complexes_to_expression(rnaseq, complexes, agg_method=complex_agg_method)
        for rnaseq in rnaseq_matrices
    ]


def calc_communication_score(
    projected_matrices: list[np.ndarray],
    gene_names: list[str] | None = None,
    lr_pairs: pd.DataFrame | None = None,
    complex_sep: str | None = None,
    complex_agg_method: str = "min",
    gene_means: np.ndarray | None = None,
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate cell-cell communication scores using build_context_ccc_tensor
    from Tensor Cell2Cell for all conditions at once.

    Parameters:
    -----------
    projected_matrices : list[np.ndarray]
        List of matrices of shape (rank, genes) representing projected cell expressions
        across different conditions
    gene_names : list[str], optional
        List of gene names corresponding to columns in the matrices
    lr_pairs : pd.DataFrame, optional
        DataFrame with 'ligand' and 'receptor' columns
    gene_means : np.ndarray, optional
        Per-gene mean expression over the full dataset, i.e. the same vector
        that was subtracted from every condition slice before fitting
        RISE/PARAFAC2 (`adata.var["means"]`). `projected_matrices` holds
        eigen-state loadings relative to that mean, so entries can be
        negative even though real expression cannot. When `gene_means` is
        given, it is added back to each matrix before scoring, turning
        `prot_a_exp`/`prot_b_exp` into an approximation of actual
        non-negative expression rather than a mean-centered loading: this is
        the fix for the communication-score sign ambiguity in issue #139 (a
        product of two centered loadings collapses to only the interaction
        term of `(mean + deviation) x (mean + deviation)`, which lets two
        negative deviations produce a spurious positive score with no
        relationship to actual signaling strength). Small negative residuals
        left by the linear reconstruction are clipped to 0. If None
        (default), matrices are used unchanged, preserving prior behavior.
    communication_score : str, default='expression_product'
        Scoring function combining each ligand/receptor pair's eigen-state
        expression into a communication score; see
        `cccrise.ccc.combine_communication_score` for the
        available options ('expression_product', 'expression_mean',
        'expression_gmean', 'expression_hill').
    hill_k : float, default=0.5
        Half-saturation constant, used only when
        `communication_score='expression_hill'`.

    Returns:
    --------
    np.ndarray
        4D interaction tensor of shape (conditions, rank, rank, n_lr_pairs)
    pd.DataFrame
        The filtered ligand-receptor pairs that correspond to the tensor's last dimension.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(projected_matrices[0].shape[1])]

    rnaseq_matrices = _matrices_to_rnaseq_dataframes(
        projected_matrices, gene_names, gene_means
    )
    lr_pairs_renamed = _rename_lr_columns(lr_pairs)
    mod_rnaseq_matrices = _apply_complex_aggregation(
        rnaseq_matrices, lr_pairs_renamed, complex_sep, complex_agg_method, verbose
    )

    # Generate communication tensor for all contexts
    tensors, _, _, ppi_names, _ = build_context_ccc_tensor(
        rnaseq_matrices=mod_rnaseq_matrices,
        ppi_data=lr_pairs_renamed,
        how="inner",
        communication_score=communication_score,
        hill_k=hill_k,
        complex_sep=complex_sep,
        upper_letter_comparison=False,
        interaction_columns=("A", "B"),
        group_ppi_by=None,
        group_ppi_method="gmean",
        verbose=verbose,
    )

    # Filter the original lr_pairs to match the pairs used in the tensor
    lr_pair_names = (
        lr_pairs["ligand"].astype(str) + "^" + lr_pairs["receptor"].astype(str)
    )
    filtered_lr_pairs = lr_pairs[lr_pair_names.isin(ppi_names)].reset_index(drop=True)

    if not ppi_names:
        raise ValueError(
            "No ligand-receptor pair matched the expression data, so the "
            "interaction tensor would have a zero-length LR mode. "
            f"First few gene names seen: {list(rnaseq_matrices[0].index[:5])}"
        )

    # Transpose to the expected format
    # From: (context, ppi_idx, rank, rank)
    # To:   (context, rank, rank, ppi_idx)
    interaction_tensor = np.transpose(np.asarray(tensors), (0, 2, 3, 1))

    return interaction_tensor, filtered_lr_pairs


def validate_cp_rank(cp_rank: int, tensor_shape: tuple[int, ...]) -> None:
    """Reject a CP rank larger than the tensor's smallest mode.

    Above that, tensorly's SVD initialization falls back to a full SVD of a
    mode unfolding -- an allocation quadratic in the unfolding's long axis, so
    hundreds of GiB for a realistic communication tensor -- and pads the excess
    components with uniform random columns, starting them from noise rather
    than from the data.

    The communication tensor's modes are ``(n_conditions, rise_rank,
    rise_rank, n_lr_pairs)``, so in practice the constraint is
    ``min(n_conditions, rise_rank)``.

    Raises
    ------
    ValueError
        If `cp_rank` exceeds the smallest mode of `tensor_shape`.
    """
    smallest_mode = min(tensor_shape)
    if cp_rank > smallest_mode:
        raise ValueError(
            f"cp_rank={cp_rank} exceeds the smallest mode of the communication "
            f"tensor (shape {tuple(tensor_shape)}, smallest mode {smallest_mode}). "
            f"Use cp_rank <= {smallest_mode}, or raise the binding mode -- for "
            "the communication tensor that means more conditions or a larger "
            "rise_rank."
        )


def run_ccc_rise_workflow(
    adata: anndata.AnnData,
    rise_rank: int,
    lr_pairs: pd.DataFrame,
    cp_rank: int | None = None,
    condition_column: str = "sample",
    random_state: int | None = None,
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    backend: str | None = None,
    compress: bool = False,
    doEmbedding: bool = False,
    keep_tensor: bool = False,
    *,
    rise_n_iter_max: int,
    cp_n_iter_max: int,
    rise_tol: float,
    cp_tol: float,
) -> dict[str, Any]:
    """The CCC-RISE pipeline: RISE (`fit_rise`) -> communication tensor -> CP
    (`fit_ccc_cp`).

    Writes into `adata` (callers wanting it pristine pass a copy): the RISE
    factors via `fit_rise`, the standardized CP factors via
    `save_ccc_rise_results` (`uns` A-D, 'weights', 'lr_pairs'), `uns["r2x"]`
    (CP R2X), and per-cell `obsm` 'sc_B' / 'rc_C' (+ 'PaCMAP' if
    `doEmbedding`) via `add_cell_embeddings`.

    Parameters
    ----------
    adata : anndata.AnnData
        Expression data with condition annotations in `obs`.
    rise_rank : int
        Rank of the RISE decomposition.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs to score.
    cp_rank : int, optional
        Rank of the CP decomposition; defaults to `rise_rank`.
    condition_column : str, default="sample"
        Column in `adata.obs` defining the conditions.
    random_state : int, optional
        Seed for the RISE fit, the CP fit and PaCMAP.
    complex_sep, communication_score, hill_k
        Communication-tensor options; see `calculate_interaction_tensor`.
    backend : str, optional
        RISE backend ('cpu', 'cupy', 'mlx', or None to auto-detect).
    compress : bool, default=False
        Fit both RISE and CP compressed (see `fit_rise` and `fit_cp`).
    doEmbedding : bool, default=False
        Fit a PaCMAP embedding of the per-cell projections.
    keep_tensor : bool, default=False
        Return the communication tensor.
    rise_n_iter_max, cp_n_iter_max, rise_tol, cp_tol
        Iteration caps and convergence tolerances for the RISE and CP fits.

    Returns
    -------
    dict[str, Any]
        'adata', 'tensor' (None unless `keep_tensor`), 'filtered_lr_pairs',
        'cp_weights', 'cp_factors' (standardized), 'r2x_rise', 'r2x_cp',
        'diagnostics' (see `fit_cp`).
    """
    adata = add_cond_idxs(adata, condition_column)
    adata, r2x_rise = fit_rise(
        adata,
        rank=rise_rank,
        n_iter_max=rise_n_iter_max,
        tolerance=rise_tol,
        random_state=random_state,
        backend=backend,
        compress=compress,
    )
    fit = fit_ccc_cp(
        adata,
        lr_pairs,
        cp_rank if cp_rank is not None else rise_rank,
        complex_sep=complex_sep,
        communication_score=communication_score,
        hill_k=hill_k,
        random_state=random_state,
        compress=compress,
        cp_n_iter_max=cp_n_iter_max,
        cp_tol=cp_tol,
    )
    cp_weights, cp_factors = fit["cp_weights"], fit["cp_factors"]
    adata = save_ccc_rise_results(
        adata, cp_factors, cp_weights, fit["filtered_lr_pairs"]
    )
    adata.uns["r2x"] = fit["r2x_cp"]
    adata = add_cell_embeddings(
        adata, cp_factors[1], cp_factors[2], doEmbedding, random_state
    )
    return fit | {
        "adata": adata,
        "tensor": fit["tensor"] if keep_tensor else None,
        "r2x_rise": r2x_rise,
    }


def fit_ccc_cp(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    cp_rank: int,
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    random_state: int | None = None,
    compress: bool = False,
    *,
    cp_n_iter_max: int,
    cp_tol: float,
) -> dict[str, Any]:
    """The post-RISE half of `run_ccc_rise_workflow`: communication tensor ->
    CP, on an AnnData already fit by `fit_rise`.

    `adata` is only read, so one RISE fit can back several CP fits (e.g. one
    per communication score).

    Returns
    -------
    dict[str, Any]
        'tensor', 'filtered_lr_pairs', 'cp_weights', 'cp_factors'
        (standardized), 'r2x_cp', 'diagnostics' (see `fit_cp`).
    """
    tensor, filtered_lr_pairs = calculate_interaction_tensor(
        adata,
        lr_pairs,
        complex_sep=complex_sep,
        communication_score=communication_score,
        hill_k=hill_k,
    )
    cp_weights, cp_factors, r2x_cp, diagnostics = fit_cp(
        tensor,
        cp_rank,
        n_iter_max=cp_n_iter_max,
        tol=cp_tol,
        random_state=random_state,
        compress=compress,
    )
    return {
        "tensor": tensor,
        "filtered_lr_pairs": filtered_lr_pairs,
        "cp_weights": cp_weights,
        "cp_factors": cp_factors,
        "r2x_cp": r2x_cp,
        "diagnostics": diagnostics,
    }


def fit_rise(
    X: anndata.AnnData,
    rank: int,
    n_iter_max: int,
    tolerance: float,
    random_state: int | None = None,
    backend: str | None = None,
    compress: bool = False,
):
    """Fit RISE with `RISE_FIT_OPTIONS` and store it on `X` via `store_pf2`
    (`uns` Pf2_A/Pf2_B/Pf2_weights, `varm["Pf2_C"]`, `obsm["projections"]`)
    plus `uns["r2x_pf2"]`.

    `backend` is 'cpu', 'cupy', 'mlx', or None to auto-detect. `compress`
    fits in a compressed gene/cell subspace sized from `rank`, with
    `RISE_COMPRESSION_OPTIONS` -- the compression `bicv` applies in
    `run_rise_rank` (parafac2 ignores those options when `compress` is off).
    """
    pf2_out, r2x = parafac2_nd(
        X,
        rank=rank,
        random_state=random_state,
        tol=tolerance,
        n_iter_max=n_iter_max,
        backend=backend,
        compress=compress,
        **RISE_COMPRESSION_OPTIONS,
        **RISE_FIT_OPTIONS,
    )
    X = store_pf2(X, pf2_out)
    X.uns["r2x_pf2"] = r2x
    return X, r2x


def condition_projections(adata: anndata.AnnData) -> list[np.ndarray]:
    """Per-condition RISE projections P_k, sliced from `obsm["projections"]`."""
    cond_idx = adata.obs["condition_unique_idxs"].to_numpy().astype(int)
    return [adata.obsm["projections"][cond_idx == i] for i in range(cond_idx.max() + 1)]


def calculate_interaction_tensor(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Communication tensor from an AnnData already fit by `fit_rise`.

    Projects each mean-centered condition matrix onto its RISE projections
    (``P_k^T X_k``) and scores ligand-receptor communication between the
    resulting eigen-states. All conditions are densified at once.

    Parameters
    ----------
    adata : anndata.AnnData
        Expression data with `obsm["projections"]` and
        `obs["condition_unique_idxs"]`.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs to score.
    complex_sep : str, optional, default="&"
        Separator for multimeric complexes.
    communication_score : str, default='expression_product'
        Scoring function; see `cccrise.ccc.combine_communication_score`
        ('expression_product', 'expression_mean', 'expression_gmean',
        'expression_hill').
    hill_k : float, default=0.5
        Half-saturation constant, used only when
        `communication_score='expression_hill'`.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        The (n_conditions, rise_rank, rise_rank, n_lr_pairs) tensor, and the
        ligand-receptor pairs retained along its last mode.
    """
    assert "projections" in adata.obsm, "Fit RISE first (`fit_rise`)."
    projected_matrices = [
        P.T @ X
        for P, X in zip(
            condition_projections(adata), condition_matrices(adata), strict=True
        )
    ]
    # Per-gene mean subtracted before fitting RISE (see `prepare_dataset`); added
    # back during scoring to keep communication scores non-negative (issue #139).
    gene_means = (
        adata.var["means"].to_numpy()
        if "means" in adata.var
        else np.zeros(adata.n_vars)
    )
    return calc_communication_score(
        projected_matrices,
        gene_names=list(adata.var_names),
        lr_pairs=lr_pairs,
        complex_sep=complex_sep,
        gene_means=gene_means,
        communication_score=communication_score,
        hill_k=hill_k,
    )


def fit_cp(
    tensor: np.ndarray,
    rank: int,
    n_iter_max: int,
    tol: float,
    random_state: int | None = None,
    compress: bool = False,
) -> tuple[np.ndarray, Factors, float, dict[str, Any]]:
    """Fit CP at a fixed rank with `CP_FIT_OPTIONS` and standardize the factors
    (`standardize_cc_pf2`).

    `compress` fits against a CANDELINC core sized for `rank`
    (`rank_selection.candelinc_compress` with `CP_COMPRESSION_OPTIONS`)
    instead of the full tensor, then maps the factors back onto the tensor's
    own modes with ``A_n = U_n F_n``.
    R2X (via `rank_selection.candelinc_r2x`), core consistency and
    collinearity are each invariant to that substitution, so compression is a
    speed optimization only.

    Returns weights, factors (in `tensor`'s own space), R2X against `tensor`,
    and the `_cpd_diagnostics` dict -- each computed in the (cheaper) core
    space when compressed.
    """
    validate_cp_rank(rank, tensor.shape)
    fit_target = tensor
    bases: Factors | None = None
    lost_sq = 0.0
    if compress:
        fit_target, bases, lost_sq = candelinc_compress(
            tensor, max_rank=rank, **CP_COMPRESSION_OPTIONS
        )

    weights, factors = parafac(
        fit_target,
        rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
        **CP_FIT_OPTIONS,
    )
    diagnostics = _cpd_diagnostics(fit_target, weights, factors, rank)
    r2x = candelinc_r2x(
        calculate_r2x(weights, factors, fit_target), fit_target, lost_sq
    )

    if bases is not None:
        factors = [b @ f for b, f in zip(bases, factors, strict=True)]

    weights, factors = standardize_cc_pf2(weights, factors)
    return weights, factors, r2x, diagnostics


def _cpd_diagnostics(
    tensor: np.ndarray, weights: np.ndarray, factors: Factors, rank: int
) -> dict[str, Any]:
    """CP fit diagnostics at the fixed rank: core consistency and collinearity.

    Lets a wide replicate spread traceable to a poor or degenerate fit be
    told apart from genuine component instability.
    """
    collinearity = cpd_component_collinearity({rank: (weights, factors)}).iloc[0]
    return {
        "CoreConsistency": cp_core_consistency(tensor, weights, factors),
        "MaxModesCollinear": int(collinearity["MaxModesCollinear"]),
        "WorstPairCanceling": collinearity["WorstPairCanceling"],
    }


def add_cell_embeddings(
    adata: anndata.AnnData,
    B: np.ndarray,
    C: np.ndarray,
    do_embedding: bool = False,
    random_state: int | None = None,
) -> anndata.AnnData:
    """Per-cell sender (`obsm["sc_B"]`) and receiver (`obsm["rc_C"]`) loadings
    from `obsm["projections"]` and the CP B / C factors, plus an optional
    PaCMAP embedding of the projections (`obsm["PaCMAP"]`)."""
    projections = np.asarray(adata.obsm["projections"])
    adata.obsm["sc_B"] = projections @ B
    adata.obsm["rc_C"] = projections @ C
    if do_embedding:
        pcm = PaCMAP(random_state=random_state)
        adata.obsm["PaCMAP"] = pcm.fit_transform(adata.obsm["projections"])
    return adata


def standardize_cc_pf2(
    weights: np.ndarray, factors: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Standardize CP factors for better interpretability.
    This function expects all inputs to be NumPy arrays on the CPU.

    Parameters
    ----------
    factors : list[np.ndarray]
        CP factors from the decomposition.
    weights : np.ndarray, optional
        Component weights from the CP decomposition. If None, they are initialized to ones.

    Returns
    -------
    tuple
        (weights, factors) after standardization.
    """
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]
    if weights is not None:
        weights = weights[gini_idx]

    weights, factors = cp_flip_sign(cp_normalize((weights, factors)), mode=1)

    return weights, factors


def calc_communication_score_pseudobulk(
    pseudobulk_matrices_df: list[pd.DataFrame],
    gene_names: list[str] | None = None,
    lr_pairs: pd.DataFrame | None = None,
    complex_sep: str | None = "&",
    complex_agg_method: str = "min",
    verbose: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate cell-cell communication scores for pseudobulk
    Parameters:
    -----------
    pseudobulk_matrices_df : list[pd.DataFrame]
        List of dataframes of shape (genes, groupby)
    gene_names : list[str], optional
        List of gene names corresponding to columns in the matrices
    lr_pairs : pd.DataFrame, optional
        DataFrame with 'ligand' and 'receptor' columns
    complex_sep : str, default="&"
        Symbol that separates the protein subunits in a multimeric complex
    complex_agg_method : str, default="min"
        Method to aggregate expression values for complexes
    verbose : bool, default=False
        Print verbose output
    Returns:
    --------
    np.ndarray
        4D interaction tensor of shape (conditions, sender, receiver, n_lr_pairs)
    pd.DataFrame
        The filtered ligand-receptor pairs that correspond to the tensor's last dimension.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(pseudobulk_matrices_df[0].shape[1])]

    # Rename columns to match Cell2Cell convention
    if "ligand" in lr_pairs.columns and "receptor" in lr_pairs.columns:
        lr_pairs_renamed = lr_pairs.rename(columns={"ligand": "A", "receptor": "B"})
        interaction_columns = ("A", "B")
    else:
        lr_pairs_renamed = lr_pairs
        interaction_columns = ("A", "B")

    # Generate expression values for protein complexes in PPI data
    if complex_sep is not None:
        if verbose:
            print("Getting expression values for protein complexes")
        _, _, _, _, complexes = get_genes_from_complexes(
            ppi_data=lr_pairs_renamed,
            complex_sep=complex_sep,
            interaction_columns=interaction_columns,
        )
        mod_rnaseq_matrices = [
            add_complexes_to_expression(
                rnaseq, complexes, agg_method=complex_agg_method
            )
            for rnaseq in pseudobulk_matrices_df
        ]
    else:
        mod_rnaseq_matrices = [df.copy() for df in pseudobulk_matrices_df]

    tensors, _, _, ppi_names, _ = build_context_ccc_tensor(
        rnaseq_matrices=mod_rnaseq_matrices,
        ppi_data=lr_pairs_renamed,
        how="inner",
        communication_score="expression_product",
        complex_sep=complex_sep,
        upper_letter_comparison=True,
        interaction_columns=interaction_columns,
        group_ppi_by=None,
        group_ppi_method="gmean",
        verbose=verbose,
    )

    # Only keep ppi_names to match the pairs used in the tensor in the interaction symbol column of the lr_pairs DataFrame
    lr_pair_names = (
        lr_pairs["ligand"].astype(str) + "^" + lr_pairs["receptor"].astype(str)
    )
    filtered_lr_pairs = lr_pairs[lr_pair_names.isin(ppi_names)].reset_index(drop=True)

    # Convert to numpy and transpose to expected format
    # From: (context, ppi_idx, sender, receiver)
    # To:   (context, sender, receiver, ppi_idx)
    interaction_tensor = np.array(tensors)
    interaction_tensor = np.transpose(interaction_tensor, (0, 2, 3, 1))

    return interaction_tensor, filtered_lr_pairs


def pseudobulk_cp_decomposition(
    interaction_tensors: np.ndarray,
    cp_rank: int,
    n_iter_max: int,
    tol: float | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """
    Perform non-negative CP decomposition on the pseudobulk interaction tensors.
    Parameters
    ----------
    interaction_tensors : np.ndarray
        The interaction tensors to decompose.
    cp_rank : int
        The rank for the CP decomposition.
    n_iter_max : int
        The maximum number of iterations for the decomposition.
    tol: float | None
        Tolerance for convergence.
    random_state : int | None
        Random seed for reproducibility.
    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], float]
        A tuple containing the CP weights, CP factors, and the R2X value.
    """
    # CP decomposition
    cp_weights, cp_factors = parafac(
        interaction_tensors,
        cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        init="svd",  # Use SVD initialization
        svd="randomized_svd",
        normalize_factors=True,
        random_state=random_state,
    )

    # Calculate R2X for the CP decomposition of the interaction tensor
    r2x = calculate_r2x(cp_weights, cp_factors, interaction_tensors)

    return cp_weights, cp_factors, r2x


def save_ccc_rise_results(
    X: anndata.AnnData,
    cpd_factors: list[np.ndarray],
    weights: np.ndarray,
    lr_pairs: pd.DataFrame,
):
    """Save CPD results in an AnnData object."""
    X.uns["A"] = cpd_factors[0]  # Condition factor
    X.uns["B"] = cpd_factors[1]  # Sender cell types factor
    X.uns["C"] = cpd_factors[2]  # Receiver cell types factor
    X.uns["D"] = cpd_factors[3]  # LR pairs factor
    X.uns["lr_pairs"] = lr_pairs["interaction_symbol"].values  # LR pairs
    X.uns["weights"] = weights  # Component weights

    return X


def add_complexes_to_expression(rnaseq_data, complexes, agg_method="min"):
    """
    Adds multimeric complexes into the gene expression matrix.
    Their gene expressions are the minimum expression value
    among the respective subunits composing them.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    complexes : dict
        Dictionary where keys are the complex names in the list of PPIs, while
        values are list of subunits for the respective complex names.

    agg_method : str, default='min'
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    Returns
    -------
    tmp_rna : pandas.DataFrame
        Gene expression data for RNA-seq experiment containing multimeric
        complex names. Their gene expressions are the minimum expression value
        among the respective subunits composing them. Columns are
        cell-types/tissues/samples and rows are genes.
    """
    tmp_rna = rnaseq_data.copy()
    for k, v in complexes.items():
        if isinstance(v, set):
            v = list(v)
        elif isinstance(v, list):
            pass  # No need to convert, already a list
        else:
            raise TypeError(
                "Values in the `complexes`dictionary must be sets or lists."
            )
        if all(g in tmp_rna.index for g in v):
            df = tmp_rna.loc[v, :]
            if agg_method == "min":
                tmp_rna.loc[k] = df.min().values.tolist()
            elif agg_method == "mean":
                tmp_rna.loc[k] = df.mean().values.tolist()
            elif agg_method == "gmean":
                tmp_rna.loc[k] = df.apply(
                    lambda x: np.exp(np.mean(np.log(x)))
                ).values.tolist()
            else:
                raise ValueError(f"{agg_method} is not a valid agg_method")
        else:
            tmp_rna.loc[k] = [0] * tmp_rna.shape[1]
    return tmp_rna
