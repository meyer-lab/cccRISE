from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.linear_model import LinearRegression


def resample(
    data: anndata.AnnData, condition_name: str, random_seed: int | None = None
) -> anndata.AnnData:
    """Perform stratified bootstrap sampling by resampling cells within each sample.

    This maintains the same number of cells per sample in the resampled dataset.
    """
    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    resampled_indices = []
    for sample in data.obs[condition_name].unique():
        # Get indices of cells in this sample
        sample_cell_indices = np.where(data.obs[condition_name] == sample)[0]
        n_cells = len(sample_cell_indices)

        # Sample with replacement within this sample's cells
        random_local_indices = np.random.randint(0, n_cells, size=n_cells)
        resampled_indices.extend(sample_cell_indices[random_local_indices])

    # Create new AnnData from resampled indices
    resampled_data = data[resampled_indices].copy()
    resampled_data.obs_names_make_unique()
    return resampled_data


def correct_conditions(X: anndata.AnnData) -> np.ndarray:
    """Correct the conditions factors by overall read depth."""
    sgIndex = X.obs["condition_unique_idxs"]

    counts = np.zeros((np.amax(sgIndex.to_numpy()) + 1, 1))

    cond_mean = np.linalg.norm(X.uns["A"], axis=1)

    x_count = cast(Any, X.X).sum(axis=1)

    for ii in range(counts.size):
        counts[ii] = np.sum(x_count[X.obs["condition_unique_idxs"] == ii])

    lr = LinearRegression()
    lr.fit(counts, cond_mean.reshape(-1, 1))

    counts_correct = lr.predict(counts)

    return X.uns["A"] / counts_correct


def pseudobulk_X(
    X: anndata.AnnData, condition_name: str, groupby: str, type: str
) -> list[pd.DataFrame]:
    """
    Calculate average gene expression for each groupby in each sample
    """
    # Get unique samples and cell types
    samples = np.unique(X.obs[condition_name].unique())
    groupby_names = np.unique(X.obs[groupby].unique())
    gene_names = X.var_names

    total_df = []
    for sample in samples:
        # Subset to current sample
        sample_mask = X.obs[condition_name] == sample
        X_sample = X[sample_mask, :]
        results = []
        for group_name in groupby_names:
            # Subset to current groupby
            group_mask = X_sample.obs[groupby] == group_name
            adata_subset = X_sample[group_mask, :]
            result_dict = {}
            subset_X = cast(Any, adata_subset.X)
            if adata_subset.n_obs > 0 and type == "mean":
                mean_expression = np.mean(subset_X.toarray(), axis=0)
                for i, gene in enumerate(gene_names):
                    result_dict[gene] = mean_expression[i]
            elif adata_subset.n_obs > 0 and type == "fraction":
                ct_df = subset_X.toarray()
                cell_fraction = (ct_df > 0).sum(axis=0) / ct_df.shape[0]
                for i, gene in enumerate(gene_names):
                    result_dict[gene] = cell_fraction[i]
            else:
                # Set zero expression for missing combinations
                for gene in gene_names:
                    result_dict[gene] = 0.0

            results.append(result_dict)

        # Dataframe with genes as rows and groupby as columns
        results_df = pd.DataFrame(results, columns=gene_names, index=groupby_names)
        total_df.append(results_df.T)

    return total_df


def add_obs_cmp_unique_one(X: anndata.AnnData, cmp: int):
    """Creates AnnData observation column for a single component."""
    label_col = f"Cmp{cmp}"
    X.obs["Label"] = np.where(X.obs[label_col], label_col, "NoLabel")

    return X


def add_obs_cmp_label(
    X: anndata.AnnData,
    cmp: int,
    pos: bool = True,
    top_perc: float = 1,
    type: str = "receiver",
):
    """Adds a boolean label to X.obs for cells in the top or bottom percentage of a single component."""
    if type == "sender":
        factor_type = cast(np.ndarray, X.obsm["sc_B"])
    elif type == "receiver":
        factor_type = cast(np.ndarray, X.obsm["rc_C"])

    if pos:
        thres_value = 100 - top_perc
        threshold = np.percentile(factor_type, thres_value, axis=0)
        idx = factor_type[:, cmp - 1] > threshold[cmp - 1]
    else:
        thres_value = top_perc
        threshold = np.percentile(factor_type, thres_value, axis=0)
        idx = factor_type[:, cmp - 1] < threshold[cmp - 1]

    X.obs[f"Cmp{cmp}"] = idx

    return X


def _gene_vector(X: anndata.AnnData, gene: str) -> np.ndarray:
    """Extract a gene's expression as a dense 1D array, whether X.X is sparse or dense."""
    expr = X[:, gene].X
    if isinstance(expr, spmatrix):
        return cast(Any, expr).toarray().flatten()
    return np.ravel(np.array(expr))


def expression_product_matrix(
    X1: anndata.AnnData, X2: anndata.AnnData, ligand: str, receptor: str
):
    """
    For each cell in X1 and each cell in X2, compute the product:
    X1[cell_i, ligand] * X2[cell_j, receptor]
    Returns a DataFrame with X1 cells as rows and X2 cells as columns.
    """
    # Ensure gene names are present
    assert ligand in X1.var_names, f"{ligand} not in X1"
    assert receptor in X2.var_names, f"{receptor} not in X2"

    expr1 = _gene_vector(X1, ligand)
    expr2 = _gene_vector(X2, receptor)

    # Compute outer product
    product_matrix = np.outer(expr1, expr2)

    # Build DataFrame
    df = pd.DataFrame(product_matrix, index=X1.obs_names, columns=X2.obs_names)
    return df


def aggregate_communication_scores_by_group(
    X_sender: anndata.AnnData,
    X_receiver: anndata.AnnData,
    ligand: str,
    receptor: str,
    sender_group: np.ndarray | pd.Series | None = None,
    receiver_group: np.ndarray | pd.Series | None = None,
    stat: str = "mean",
) -> pd.DataFrame:
    """
    Compute a communication-score statistic for a ligand-receptor pair, aggregated over
    sender/receiver cell groups.

    Parameters
    ----------
    sender_group, receiver_group : array-like, optional
        Group label for each sender/receiver cell (aligned to X_sender/X_receiver obs
        order). Cells are left ungrouped (a single "All" group) if omitted.
    stat : str, default='mean'
        Aggregation statistic within each sender-group x receiver-group block.
        Options are:

        - 'mean' : Mean communication score.
        - 'pct_nonzero' : Percentage of sender-receiver cell pairs with a nonzero score.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with one row per (sender_group, receiver_group)
        combination present, containing the aggregated score and cell counts.
    """
    product_matrix = expression_product_matrix(X_sender, X_receiver, ligand, receptor)

    sender_group = (
        np.full(X_sender.n_obs, "All")
        if sender_group is None
        else np.asarray(sender_group)
    )
    receiver_group = (
        np.full(X_receiver.n_obs, "All")
        if receiver_group is None
        else np.asarray(receiver_group)
    )

    if stat == "mean":
        agg = lambda values: values.mean() if values.size else np.nan
    elif stat == "pct_nonzero":
        agg = lambda values: 100 * np.mean(values > 0) if values.size else np.nan
    else:
        raise ValueError(f"Unknown stat: {stat}")

    values = product_matrix.values
    rows = []
    for s_label in np.unique(sender_group):
        s_idx = np.where(sender_group == s_label)[0]
        for r_label in np.unique(receiver_group):
            r_idx = np.where(receiver_group == r_label)[0]
            if s_idx.size == 0 or r_idx.size == 0:
                continue

            block = values[np.ix_(s_idx, r_idx)]
            rows.append(
                {
                    "sender_group": s_label,
                    "receiver_group": r_label,
                    "communication_score": agg(block),
                    "n_sender_cells": s_idx.size,
                    "n_receiver_cells": r_idx.size,
                    "n_interactions": s_idx.size * r_idx.size,
                }
            )

    return pd.DataFrame(rows)


def _binned_means(expr: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean of `expr` within `n_bins` contiguous blocks (last block absorbs the remainder)."""
    n = len(expr)
    group_size = n // n_bins
    groups = np.clip(np.arange(n) // group_size, 0, n_bins - 1)
    return np.array([expr[groups == g].mean() for g in range(n_bins)])


def binned_expression_product_matrix(
    X1: anndata.AnnData,
    X2: anndata.AnnData,
    ligand: str,
    receptor: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Equivalent to binning `expression_product_matrix(X1, X2, ligand, receptor)` into an
    `n_bins` x `n_bins` matrix of block means, but computed without ever materializing the
    full len(X1) x len(X2) outer product: since block entries are a product of independent
    per-cell terms, the mean of a block is just the product of the row- and column-block
    means of the two expression vectors.
    """
    assert ligand in X1.var_names, f"{ligand} not in X1"
    assert receptor in X2.var_names, f"{receptor} not in X2"

    expr1 = _gene_vector(X1, ligand)
    expr2 = _gene_vector(X2, receptor)

    row_means = _binned_means(expr1, n_bins)
    col_means = _binned_means(expr2, n_bins)

    product_matrix = np.outer(row_means, col_means)
    return pd.DataFrame(
        product_matrix, index=np.arange(n_bins), columns=np.arange(n_bins)
    )


def binned_complex_ligand_product_matrix(
    X1: anndata.AnnData,
    X2: anndata.AnnData,
    ligand_subunits: list[str],
    receptor: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Same as `binned_expression_product_matrix`, but for a multi-subunit ligand complex
    (e.g. LTA&LTB). The complex's per-cell expression is the minimum across its subunits,
    the limiting-subunit convention used elsewhere for scoring ligand complexes.
    """
    for subunit in ligand_subunits:
        assert subunit in X1.var_names, f"{subunit} not in X1"
    assert receptor in X2.var_names, f"{receptor} not in X2"

    expr1 = np.minimum.reduce([_gene_vector(X1, s) for s in ligand_subunits])
    expr2 = _gene_vector(X2, receptor)

    row_means = _binned_means(expr1, n_bins)
    col_means = _binned_means(expr2, n_bins)

    product_matrix = np.outer(row_means, col_means)
    return pd.DataFrame(
        product_matrix, index=np.arange(n_bins), columns=np.arange(n_bins)
    )


def add_obs_cmp_both_label(
    X: anndata.AnnData,
    cmp1: int,
    cmp2: int,
    pos1=True,
    pos2=True,
    top_perc=1,
    type="sender",
):
    """Adds if cells in top/bot percentage"""
    X = add_obs_cmp_label(X, cmp1, pos1, top_perc, type)
    X = add_obs_cmp_label(X, cmp2, pos2, top_perc, type)

    X.obs["Both"] = X.obs[f"Cmp{cmp1}"] & X.obs[f"Cmp{cmp2}"]

    return X


def add_obs_cmp_unique_two(X: anndata.AnnData, cmp1: int, cmp2: int):
    """Creates AnnData observation column"""
    obs = cast(pd.DataFrame, X.obs)
    obs.loc[((obs[f"Cmp{cmp1}"] == True) & (obs[f"Cmp{cmp2}"] == False), "Label")] = (
        f"Cmp{cmp1}"
    )
    obs.loc[(obs[f"Cmp{cmp1}"] == False) & (obs[f"Cmp{cmp2}"] == True), "Label"] = (
        f"Cmp{cmp2}"
    )
    obs.loc[(obs[f"Cmp{cmp1}"] == True) & (obs[f"Cmp{cmp2}"] == True), "Label"] = "Both"
    obs.loc[(obs[f"Cmp{cmp1}"] == False) & (obs[f"Cmp{cmp2}"] == False), "Label"] = (
        "NoLabel"
    )

    return X
