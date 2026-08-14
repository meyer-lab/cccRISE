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

    # Get expression vectors

    # Ensure 1D dense arrays
    # Convert to dense 1D arrays, even if sparse
    expr1 = X1[:, ligand].X
    if isinstance(expr1, spmatrix):
        expr1 = cast(Any, expr1).toarray().flatten()
    else:
        expr1 = np.ravel(np.array(expr1))

    expr2 = X2[:, receptor].X
    if isinstance(expr2, spmatrix):
        expr2 = cast(Any, expr2).toarray().flatten()
    else:
        expr2 = np.ravel(np.array(expr2))

    # Compute outer product
    product_matrix = np.outer(expr1, expr2)

    # Build DataFrame
    df = pd.DataFrame(product_matrix, index=X1.obs_names, columns=X2.obs_names)
    return df


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
    if type == "sender":
        factor_type = cast(np.ndarray, X.obsm["sc_B"])
    elif type == "receiver":
        factor_type = cast(np.ndarray, X.obsm["rc_C"])

    pos_neg = [pos1, pos2]

    for i, cmp in enumerate([cmp1, cmp2]):
        if i == 0:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold1 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] > threshold1[cmp - 1]

            else:
                thres_value = top_perc
                threshold1 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] < threshold1[cmp - 1]

        if i == 1:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold2 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] > threshold2[cmp - 1]
            else:
                thres_value = top_perc
                threshold2 = np.percentile(factor_type, thres_value, axis=0)
                idx = factor_type[:, cmp - 1] < threshold2[cmp - 1]

        X.obs[f"Cmp{cmp}"] = idx

    if pos1 is True and pos2 is True:
        idx = (factor_type[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )
    elif pos1 is False and pos2 is False:
        idx = (factor_type[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] < threshold2[cmp2 - 1]
        )
    elif pos1 is True and pos2 is False:
        idx = (factor_type[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] < threshold2[cmp2 - 1]
        )
    elif pos1 is False and pos2 is True:
        idx = (factor_type[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            factor_type[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )

    X.obs["Both"] = idx

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


def average_product_matrix_ccc(df):
    """
    Groups a DataFrame into a 10x10 matrix by binning rows and columns and averaging within bins.
    Prints shape information for debugging.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    row_group_size = n_rows // 10
    col_group_size = n_cols // 10
    row_groups = np.arange(n_rows) // row_group_size
    col_groups = np.arange(n_cols) // col_group_size
    row_groups = np.clip(row_groups, 0, 9)
    col_groups = np.clip(col_groups, 0, 9)

    # Group rows
    df_grouped = df.groupby(row_groups).mean()

    # Group columns by transposing, grouping, then transposing back
    df_grouped = df_grouped.T.groupby(col_groups).mean().T

    return df_grouped
