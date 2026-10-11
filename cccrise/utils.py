from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .factors import compute_cell_projections, load_factors_netcdf
from .import_data import add_cond_idxs


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


def jackknife_units(data: anndata.AnnData, patient_name: str) -> np.ndarray:
    """The patients `leave_out_patient` drops one at a time, in sorted order,
    so jackknife replicate ``i`` always leaves out the same patient."""
    return np.unique(np.asarray(data.obs[patient_name]).astype(str))


def jackknife_groups(
    units: np.ndarray, n_folds: int | None = None, random_state: int = 0
) -> list[np.ndarray]:
    """The units each jackknife replicate leaves out: every unit on its own,
    in `units`' order, when `n_folds` is None; otherwise `units` shuffled by
    `random_state` and split into `n_folds` near-equal folds (a grouped,
    delete-d jackknife), for cohorts with too many units to refit once per
    unit."""
    if n_folds is None:
        return [units[i : i + 1] for i in range(len(units))]
    if not 2 <= n_folds <= len(units):
        raise ValueError(f"n_folds must be in [2, {len(units)}], got {n_folds}.")
    shuffled = np.random.default_rng(random_state).permutation(units)
    return [np.sort(fold) for fold in np.array_split(shuffled, n_folds)]


def leave_out_patient(
    data: anndata.AnnData, patient_name: str, patient: str | Sequence[str]
) -> anndata.AnnData:
    """Drop every cell of one patient (or of each of several) -- all of its
    conditions -- for a jackknife replicate.

    Nothing is drawn or duplicated: the replicate is the cohort minus the
    patients, so each condition keeps its own label and the replicate's roster
    is a subset of the full cohort's.

    Parameters
    ----------
    data : anndata.AnnData
        Single-cell expression data.
    patient_name : str
        Column in `data.obs` identifying the patient; a patient contributing
        several conditions loses all of them.
    patient : str | Sequence[str]
        The patient to leave out, as in `jackknife_units`, or several.

    Returns
    -------
    anndata.AnnData
        A copy without `patient`'s cells, with `var["means"]` recomputed over
        the remaining cells. That column is load-bearing twice over:
        `parafac2.anndata_to_list` centers each condition slice with it before
        the RISE fit, and `ccc_rise.calc_communication_score` adds it back to
        un-center the projected matrices before scoring, so a stale column
        would bias both stages and surface as spurious instability.
    """
    patients = [patient] if isinstance(patient, str) else list(patient)
    keep = ~np.isin(np.asarray(data.obs[patient_name]).astype(str), patients)
    if keep.all():
        raise ValueError(f"No cells of patient {patient!r} in {patient_name!r}.")
    left = data[keep].copy()
    left.var["means"] = np.asarray(cast(Any, left.X).mean(axis=0)).ravel()
    return left


def replicate_seed(random_state: int, replicate: int) -> int:
    """Seed for one replicate: ``random_state * 1000 + replicate + 1``.

    One integer per (axis, replicate) drives the RISE refit and the CP
    restarts, so a replicate's RISE and CPD rows describe the same fit by
    construction rather than by coincidence. The ``+ 1``
    leaves `random_state` itself to the reference pass, so no replicate reuses
    the reference's starting point.
    """
    return random_state * 1000 + replicate + 1


def load_rise_reference(
    reference_path: str | Path,
    adata: anndata.AnnData,
    condition_column: str,
) -> anndata.AnnData:
    """Populate `adata` with a RISE (PARAFAC2) fit loaded from a factors.nc,
    instead of fitting PARAFAC2.

    Mimics `parafac2.store_pf2`'s effect on `adata` (`Pf2_weights`/`Pf2_A`/
    `Pf2_B` in `.uns`, `Pf2_C` in `.varm`, `.obsm["projections"]`), with the
    per-cell projections recomputed by `factors.compute_cell_projections`, so
    downstream code that expects a RISE-fit AnnData works unchanged.

    Parameters
    ----------
    reference_path : str | Path
        A factors.nc holding a RISE fit (see `factors.save_ccc_rise_fit`).
    adata : anndata.AnnData
        The raw dataset the reference was fit on; its cells and genes must
        match the reference's exactly, in the same order (checked, not
        assumed).
    condition_column : str
        Column in `adata.obs` holding the condition/patient identity.

    Returns
    -------
    anndata.AnnData
        `adata`, mutated in place and returned.
    """
    ds = load_factors_netcdf(reference_path)
    adata = add_cond_idxs(adata, condition_column)
    adata.obsm["projections"] = compute_cell_projections(ds, adata)
    adata.uns["Pf2_weights"] = ds["pf2_weights"].values
    adata.uns["Pf2_A"] = ds["pf2_A"].values
    adata.uns["Pf2_B"] = ds["pf2_B"].values
    adata.varm["Pf2_C"] = ds["pf2_C"].values
    adata.uns["r2x_pf2"] = float(ds.attrs.get("r2x_pf2", 0.0))
    return adata


def resample_tensor(
    interaction_tensors: np.ndarray,
    random_seed: int | None = None,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Bootstrap the communication tensor's ligand-receptor mode."""
    rng = np.random.default_rng(random_seed)
    n_lr = interaction_tensors.shape[-1]
    indices = rng.choice(n_lr, size=n_lr, replace=True)
    resampled = interaction_tensors[..., indices]
    if return_indices:
        return resampled, indices
    return resampled


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
    if X.is_view:
        X = X.copy()
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
    if X.is_view:
        X = X.copy()
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
    expr: Any = X[:, gene].X
    if hasattr(expr, "toarray"):
        return np.ravel(expr.toarray())
    if hasattr(expr, "to_memory"):
        return np.ravel(expr.to_memory())
    return np.ravel(np.asarray(expr))


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

    sender_group = _fill_default_group(sender_group, X_sender.n_obs)
    receiver_group = _fill_default_group(receiver_group, X_receiver.n_obs)
    agg = _group_aggregator(stat)

    rows = _score_group_blocks(product_matrix.values, sender_group, receiver_group, agg)
    return pd.DataFrame(rows)


def _fill_default_group(group: np.ndarray | pd.Series | None, n_obs: int) -> np.ndarray:
    """Return `group` as an array, or a single "All" label per cell if omitted."""
    return np.full(n_obs, "All") if group is None else np.asarray(group)


def _group_aggregator(stat: str):
    """Return the reduction function for a given `stat` option."""
    if stat == "mean":
        return lambda values: values.mean() if values.size else np.nan
    if stat == "pct_nonzero":
        return lambda values: 100 * np.mean(values > 0) if values.size else np.nan
    raise ValueError(f"Unknown stat: {stat}")


def _score_group_blocks(
    values: np.ndarray,
    sender_group: np.ndarray,
    receiver_group: np.ndarray,
    agg,
) -> list[dict]:
    """Aggregate `values` over every present sender-group x receiver-group block."""
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
    return rows


def _binned_means(expr: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean of `expr` within `n_bins` contiguous blocks (last block absorbs the remainder)."""
    n = len(expr)
    if n == 0:
        return np.zeros(n_bins)
    group_size = max(1, n // n_bins)
    groups = np.clip(np.arange(n) // group_size, 0, n_bins - 1)
    return np.array(
        [
            float(np.mean(expr[groups == g])) if np.any(groups == g) else 0.0
            for g in range(n_bins)
        ]
    )


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


# `obsm` key holding the per-cell loadings for each `factor_type`.
LOADING_KEYS = {"sender": "sc_B", "receiver": "rc_C", "eigenstate": "projections"}


def celltype_component_association(
    X: anndata.AnnData,
    factor_type: str = "sender",
    celltype_col: str = "celltype",
    stat: str = "mean",
    top_perc: float = 10,
) -> pd.DataFrame:
    """
    Quantify how strongly each cell type is associated with each CPD component, or
    with each RISE eigenstate.

    The pseudobulk decomposition aggregates cells by type before factorization, so
    its sender/receiver factors (`uns["B"]`, `uns["C"]`) are already cell type x
    component matrices that can be plotted directly. CCC-RISE instead resolves the
    two cell dimensions into RISE eigenstates, so cell-type associations have to be
    recovered from the per-cell loadings `obsm["sc_B"]` / `obsm["rc_C"]`
    (`projections @ B` and `projections @ C`) by aggregating them within cell type.

    Parameters
    ----------
    X : anndata.AnnData
        AnnData carrying per-cell sender/receiver loadings in
        `obsm["sc_B"]` / `obsm["rc_C"]` and cell type labels in `.obs`.
    factor_type : {"sender", "receiver", "eigenstate"}, default="sender"
        Whether to use the sender (`sc_B`) or receiver (`rc_C`) loadings, or the
        raw per-cell eigenstate projections (`obsm["projections"]`).
    celltype_col : str, default="celltype"
        Column of `.obs` holding cell type labels.
    stat : {"mean", "effect_size", "enrichment", "pct_top"}, default="mean"
        How the association is quantified:

        - 'mean' : Mean per-cell loading within the cell type. The direct analogue
          of a row of the pseudobulk sender/receiver factor, and on the same scale
          across components (so components can be compared to one another).
        - 'effect_size' : Cohen's d of the cell type's loadings against all other
          cells. Scale-free, so it measures how distinctly a cell type is loaded
          rather than how large the component is.
        - 'enrichment' : log2 fold enrichment of the cell type among the top
          `top_perc` percent of cells for that component, using the same
          top-percentile convention as `add_obs_cmp_label`. Counts are computed
          with a pseudocount of one cell so that absent cell types stay finite.
          Saturates near log2(100 / `top_perc`) once most of a cell type sits in
          the tail, so it cannot separate two strongly loaded cell types.
        - 'pct_top' : percent of the cell type's own cells that fall in that top
          `top_perc` percent. Unlike 'mean' it survives a bimodal population,
          whose positive and negative halves cancel in the mean.

    Returns
    -------
    pandas.DataFrame
        Cell types (sorted) as rows, 1-based component (or eigenstate) numbers as
        columns.
    """
    if factor_type not in LOADING_KEYS:
        raise ValueError(f"Unknown factor_type: {factor_type}")
    loadings = np.asarray(X.obsm[LOADING_KEYS[factor_type]], dtype=float)

    labels = np.asarray(cast(pd.DataFrame, X.obs)[celltype_col]).astype(str)
    celltypes = np.unique(labels)

    if stat in ("enrichment", "pct_top"):
        is_top = loadings > np.percentile(loadings, 100 - top_perc, axis=0)
        top_rate_all = is_top.mean(axis=0)

    rows = []
    for celltype in celltypes:
        mask = labels == celltype
        if stat == "mean":
            values = loadings[mask].mean(axis=0)
        elif stat == "effect_size":
            inside, outside = loadings[mask], loadings[~mask]
            pooled_sd = np.sqrt(
                (inside.var(axis=0, ddof=1) + outside.var(axis=0, ddof=1)) / 2
            )
            values = (inside.mean(axis=0) - outside.mean(axis=0)) / np.where(
                pooled_sd > 0, pooled_sd, np.nan
            )
        elif stat == "enrichment":
            top_rate_in = (is_top[mask].sum(axis=0) + 1) / (mask.sum() + 1)
            values = np.log2(top_rate_in / top_rate_all)
        elif stat == "pct_top":
            values = 100 * is_top[mask].mean(axis=0)
        else:
            raise ValueError(f"Unknown stat: {stat}")

        rows.append(values)

    return pd.DataFrame(
        np.vstack(rows),
        index=pd.Index(celltypes, name=celltype_col),
        columns=pd.Index(
            np.arange(1, loadings.shape[1] + 1),
            name="Eigenstate" if factor_type == "eigenstate" else "Component",
        ),
    )
