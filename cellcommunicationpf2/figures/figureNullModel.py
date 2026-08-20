"""
Figure NullModel: Statistical Significance and Null Modeling of Interaction Tensors.

This figure visualizes the permutation-based null model results on the BALF COVID dataset:
A) Percentage of significant interactions (p < 0.05) across communication score percentiles.
B) Communication score distributions for significant (p < 0.05) vs non-significant (p >= 0.05) interactions.
C) Empirical p-value distribution across all interaction tensor entries.
"""

from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from parafac2.parafac2 import anndata_to_list, parafac2_nd
from scipy.sparse import spmatrix

from ..ccc_rise import calc_communication_score
from ..import_data import add_cond_idxs, import_balf_covid, import_ligand_receptor_pairs
from .common import getSetup, subplotLabel


def generate_null_tensors(
    projected_matrices: list[np.ndarray],
    gene_names: list[str],
    filtered_lr_pairs: pd.DataFrame,
    n_perms: int = 1000,
    random_state: int | None = None,
    complex_sep: str | None = None,
    complex_agg_method: str = "min",
) -> Iterator[np.ndarray]:
    """Generates permuted null 4D interaction tensors by shuffling receptor pairings."""
    rng = np.random.default_rng(random_state)

    for _ in range(n_perms):
        permuted_lr = filtered_lr_pairs.copy()
        permuted_lr["receptor"] = rng.permutation(permuted_lr["receptor"].values)

        null_tensor, _ = calc_communication_score(
            projected_matrices=projected_matrices,
            gene_names=gene_names,
            lr_pairs=permuted_lr,
            complex_sep=complex_sep,
            complex_agg_method=complex_agg_method,
        )
        yield null_tensor


def calculate_pvalues(
    observed_tensor: np.ndarray,
    null_tensors: Iterable[np.ndarray],
    n_perms: int,
) -> np.ndarray:
    """Computes empirical p-values by comparing the observed tensor to the null distribution."""
    ge_count = np.zeros_like(observed_tensor, dtype=np.int32)
    for T_null in null_tensors:
        ge_count += (T_null >= observed_tensor).astype(np.int32)
    return (ge_count + 1.0) / (n_perms + 1.0)


def compute_interaction_pvalues(
    projected_matrices: list[np.ndarray],
    gene_names: list[str] | None = None,
    lr_pairs: pd.DataFrame | None = None,
    n_perms: int = 1000,
    random_state: int | None = None,
    complex_sep: str | None = None,
    complex_agg_method: str = "min",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute empirical p-values for every interaction in the 4D communication tensor
    against a permutation null distribution.

    Parameters
    ----------
    projected_matrices : list[np.ndarray]
        List of matrices of shape (rank, genes) representing projected cell expressions
        across conditions.
    gene_names : list[str], optional
        List of gene names corresponding to columns in projected matrices.
    lr_pairs : pd.DataFrame, optional
        DataFrame with 'ligand' and 'receptor' columns.
    n_perms : int, default=1000
        Number of permutations for the null distribution.
    random_state : int, optional
        Random seed for reproducibility.
    complex_sep : str, optional
        Separator for multimeric complexes (e.g., '&').
    complex_agg_method : str, default='min'
        Method to aggregate complex expression.

    Returns
    -------
    interaction_tensor : np.ndarray
        Observed 4D interaction tensor of shape (conditions, rank, rank, n_lr_pairs).
    pvalue_tensor : np.ndarray
        Empirical p-values of shape (conditions, rank, rank, n_lr_pairs).
    filtered_lr_pairs : pd.DataFrame
        DataFrame of filtered ligand-receptor pairs matching the last axis.
    """
    interaction_tensor, filtered_lr_pairs = calc_communication_score(
        projected_matrices=projected_matrices,
        gene_names=gene_names,
        lr_pairs=lr_pairs,
        complex_sep=complex_sep,
        complex_agg_method=complex_agg_method,
    )
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(projected_matrices[0].shape[1])]

    null_tensors = generate_null_tensors(
        projected_matrices,
        gene_names,
        filtered_lr_pairs,
        n_perms=n_perms,
        random_state=random_state,
        complex_sep=complex_sep,
        complex_agg_method=complex_agg_method,
    )
    pvalue_tensor = calculate_pvalues(interaction_tensor, null_tensors, n_perms)

    return interaction_tensor, pvalue_tensor, filtered_lr_pairs


def filter_tensor_by_pvalue(
    interaction_tensor: np.ndarray,
    pvalue_tensor: np.ndarray,
    threshold: float = 0.05,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Mask elements in the interaction tensor where p-value >= threshold.

    Parameters
    ----------
    interaction_tensor : np.ndarray
        4D communication tensor.
    pvalue_tensor : np.ndarray
        4D tensor of p-values corresponding to each interaction.
    threshold : float, default=0.05
        Significance threshold (values with p < threshold are kept).
    fill_value : float, default=0.0
        Value used to replace non-significant scores.

    Returns
    -------
    np.ndarray
        Masked interaction tensor.
    """
    return np.where(pvalue_tensor < threshold, interaction_tensor, fill_value)


def compute_null_model_analysis(
    n_perms: int = 1000,
    rise_rank: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Run full null model workflow on BALF COVID data and compute plotting statistics.

    Returns
    -------
    df_quantiles : pd.DataFrame
        Significance rates across top score percentiles.
    df_scores : pd.DataFrame
        Sampled communication scores labeled by significance.
    df_components : pd.DataFrame
        Cosine similarities between raw and masked CP components.
    fms_score : float
        Overall Factor Match Score.
    """
    # 1. Import data
    adata = import_balf_covid(gene_threshold=0.001, normalize=True)
    adata = add_cond_idxs(adata, "sample")

    # 2. PARAFAC2 RISE factorization
    pf2_output, _ = parafac2_nd(
        adata, rank=rise_rank, n_iter_max=10, random_state=random_state
    )
    _, _, projections = pf2_output

    X_list = anndata_to_list(adata)
    projected_matrices = []
    for i, tensor in enumerate(X_list):
        if isinstance(tensor, spmatrix):
            tensor_np = tensor.toarray()
        elif hasattr(tensor, "get"):
            tensor_np = tensor.get()
        else:
            tensor_np = np.asarray(tensor)

        proj_np = (
            projections[i].get()
            if hasattr(projections[i], "get")
            else np.asarray(projections[i])
        )
        projected_matrices.append(proj_np.T @ tensor_np)

    gene_names = list(adata.var_names)

    # 3. Assemble observed 4D tensor
    observed_tensor, filtered_pairs = calc_communication_score(
        projected_matrices=projected_matrices,
        gene_names=gene_names,
        complex_sep="&",
    )

    # 4. Generate null permuted tensors & calculate p-values
    null_tensors_gen = generate_null_tensors(
        projected_matrices,
        gene_names,
        filtered_pairs,
        n_perms=n_perms,
        random_state=random_state,
        complex_sep="&",
    )
    pvalues = calculate_pvalues(observed_tensor, null_tensors_gen, n_perms=n_perms)

    # 5. Quantile Analysis
    scores_flat = observed_tensor.flatten()
    pvals_flat = pvalues.flatten()

    percentiles = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
    quantile_data = []
    for top_pct in percentiles:
        thresh = np.percentile(scores_flat, 100 - top_pct)
        top_mask = scores_flat >= thresh
        sig_in_top = float(np.mean(pvals_flat[top_mask] < 0.05) * 100)
        mean_pval = float(np.mean(pvals_flat[top_mask]))
        quantile_data.append(
            {
                "Top Percentile (%)": top_pct,
                "Percentile Label": f"Top {top_pct}%",
                "Score Threshold": thresh,
                "% Significant (p < 0.05)": sig_in_top,
                "Mean p-value": mean_pval,
            }
        )

    # Add overall baseline
    quantile_data.append(
        {
            "Top Percentile (%)": 100.0,
            "Percentile Label": "All Scores",
            "Score Threshold": float(np.min(scores_flat)),
            "% Significant (p < 0.05)": float(np.mean(pvals_flat < 0.05) * 100),
            "Mean p-value": float(np.mean(pvals_flat)),
        }
    )
    df_quantiles = pd.DataFrame(quantile_data)

    # Sample scores for distribution plot
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(scores_flat), size=min(50000, len(scores_flat)), replace=False)
    sample_scores = scores_flat[sample_idx]
    sample_pvals = pvals_flat[sample_idx]

    df_scores = pd.DataFrame(
        {
            "Score": sample_scores,
            "p-value": sample_pvals,
            "Status": np.where(sample_pvals < 0.05, "Significant (p < 0.05)", "Non-significant (p \u2265 0.05)"),
        }
    )

    return df_quantiles, df_scores


def makeFigure():
    """Generate Figure showing Null Modeling significance across communication score tiers."""
    ax, f = getSetup((10, 3.2), (1, 3))
    subplotLabel(ax)

    df_quantiles, df_scores = compute_null_model_analysis()

    # --- Panel A: Significance Rate vs. Score Percentiles ---
    plot_df = df_quantiles[df_quantiles["Top Percentile (%)"] <= 20.0].copy()
    sns.barplot(
        data=plot_df,
        x="Percentile Label",
        y="% Significant (p < 0.05)",
        hue="Percentile Label",
        legend=False,
        ax=ax[0],
        palette="Blues_r",
    )
    baseline_val = df_quantiles.loc[
        df_quantiles["Top Percentile (%)"] == 100.0, "% Significant (p < 0.05)"
    ].values[0]
    ax[0].axhline(
        baseline_val,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline ({baseline_val:.1f}%)",
    )
    ax[0].set_title("Significance Rate by Score Tier", fontsize=10, fontweight="bold")
    ax[0].set_ylabel("% Significant ($p < 0.05$)", fontsize=9)
    ax[0].set_xlabel("Score Quantile Tier", fontsize=9)
    ax[0].tick_params(axis="x", rotation=35)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(True, alpha=0.3)

    # Add text labels above each bar showing exact percentages
    for p in ax[0].patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax[0].annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=6.5,
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax[0].set_ylim(0, 115)

    # --- Panel B: Score Distribution by Significance Status ---
    sns.boxplot(
        data=df_scores,
        x="Status",
        y="Score",
        hue="Status",
        legend=False,
        ax=ax[1],
        palette=["#2b5c8f", "#d95f02"],
        showfliers=False,
    )
    ax[1].set_title("Communication Score by Status", fontsize=10, fontweight="bold")
    ax[1].set_ylabel("Communication Score Magnitude", fontsize=9)
    ax[1].set_xlabel("")
    ax[1].grid(True, alpha=0.3)

    # --- Panel C: Empirical P-value Distribution ---
    bins = np.linspace(0.0, 1.0, 21)  # Bins of width 0.05 so threshold 0.05 falls on the bin boundary
    sns.histplot(
        df_scores["p-value"],
        bins=bins,
        kde=False,
        ax=ax[2],
        color="#3182bd",
        stat="density",
    )
    ax[2].axvline(0.05, color="red", linestyle="--", linewidth=1.5, label="Threshold ($p = 0.05$)")
    ax[2].set_title("Empirical $p$-value Distribution", fontsize=10, fontweight="bold")
    ax[2].set_xlabel("Empirical $p$-value", fontsize=9)
    ax[2].set_ylabel("Density", fontsize=9)
    ax[2].set_xlim(0.0, 1.0)
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].grid(True, alpha=0.3)

    return f
