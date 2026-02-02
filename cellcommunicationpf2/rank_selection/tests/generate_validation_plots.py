"""Generate rank selection validation plots using synthetic data with known rank."""

import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_array
from tensorly.parafac2_tensor import parafac2_to_slices
from tensorly.random import random_parafac2

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cellcommunicationpf2.import_data import add_cond_idxs, import_balf_covid
from cellcommunicationpf2.rank_selection import run_rank_selection
from parafac2.parafac2 import parafac2_nd

USE_RANDOM_PARAFAC2 = False
TARGET_CELLS_PER_COND = 2500
TARGET_N_CONDITIONS = 8
TARGET_N_GENES = 200

print("Loading real scRNA-seq data...")
adata_full = import_balf_covid()
print(f"Full data: {adata_full.shape}")


def create_known_rank_data(adata_full, true_rank, cells_per_cond=100, n_genes=200,
                            random_state=42):
    """Factorize real data at known rank, then reconstruct to get data with known effective rank."""
    print(f"\n=== Creating known-rank-{true_rank} data (factorize-reconstruct) ===")
    
    np.random.seed(random_state)
    
    # Select conditions and sample cells from each
    all_conditions = adata_full.obs['sample'].unique()
    n_conditions = len(all_conditions)
    
    all_sampled_indices = []
    for condition in all_conditions:
        cell_indices_for_condition = np.where(adata_full.obs['sample'] == condition)[0]
        n_cells_to_sample = min(cells_per_cond, len(cell_indices_for_condition))
        sampled_cell_indices = np.random.choice(
            cell_indices_for_condition, size=n_cells_to_sample, replace=False
        )
        all_sampled_indices.extend(sampled_cell_indices)
    
    adata_subsampled = adata_full[all_sampled_indices].copy()
    
    # Keep only the top expressed genes
    gene_mean_expression = np.asarray(adata_subsampled.X.mean(axis=0)).flatten()
    top_gene_indices = np.argsort(gene_mean_expression)[-n_genes:]
    adata_subsampled = adata_subsampled[:, top_gene_indices].copy()
    
    # Add condition index column required by parafac2_nd
    adata_subsampled = add_cond_idxs(adata_subsampled, "sample")
    print(f"Subsampled: {adata_subsampled.shape}")
    
    # Factorize at the true rank
    print(f"Factorizing at rank {true_rank}...")
    pf2_output, r2x = parafac2_nd(
        adata_subsampled, rank=true_rank, n_iter_max=100, tol=1e-6, random_state=random_state
    )
    print(f"R2X at rank {true_rank}: {r2x:.4f}")
    
    # Reconstruct data from the factorization (this gives us exact-rank data)
    reconstructed_slices = parafac2_to_slices(pf2_output)
    condition_idxs = adata_subsampled.obs["condition_unique_idxs"].values
    
    reconstructed_X_per_condition = []
    for cond_idx in range(n_conditions):
        cells_in_condition = (condition_idxs == cond_idx).sum()
        reconstructed_slice = reconstructed_slices[cond_idx][:cells_in_condition, :]
        reconstructed_X_per_condition.append(reconstructed_slice)
    
    reconstructed_X = np.vstack(reconstructed_X_per_condition).astype(np.float32)
    
    # Create output AnnData with reconstructed matrix
    adata_reconstructed = adata_subsampled.copy()
    adata_reconstructed.X = csr_array(reconstructed_X)
    
    return adata_reconstructed, r2x


def create_random_parafac2_data(true_rank, n_conditions=8, cells_per_cond=100, n_genes=200,
                                 random_state=42):
    """Generate synthetic PARAFAC2 data with random factors."""
    print(f"\n=== Creating known-rank-{true_rank} data (random PARAFAC2) ===")
    
    shapes = [(cells_per_cond, n_genes) for _ in range(n_conditions)]
    pf2_tensor = random_parafac2(shapes, rank=true_rank, full=False, random_state=random_state)
    slices = parafac2_to_slices(pf2_tensor)
    
    print(f"Generated {n_conditions} conditions with {cells_per_cond} cells each, {n_genes} genes")
    
    X_list = []
    sample_labels = []
    condition_idxs = []
    
    for i, slice_data in enumerate(slices):
        X_list.append(np.asarray(slice_data))
        sample_labels.extend([f"condition_{i}"] * slice_data.shape[0])
        condition_idxs.extend([i] * slice_data.shape[0])
    
    X = np.vstack(X_list).astype(np.float32)

    np.random.seed(random_state)
    noise_scale = 0.01 * np.std(X)
    X += np.random.randn(*X.shape).astype(np.float32) * noise_scale

    adata = ad.AnnData(X=csr_array(X))
    adata.obs['sample'] = sample_labels
    adata.obs['condition_unique_idxs'] = condition_idxs
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    print(f"Synthetic data shape: {adata.shape}")
    
    return adata, 1.0


def get_test_ranks(true_rank, interval=5, overshoot=20):
    """Generate list of ranks to test, including the true rank."""
    ranks = list(range(1, true_rank + overshoot + 1, interval))
    if true_rank not in ranks:
        ranks.append(true_rank)
    return sorted(ranks)

true_ranks = [20, 30, 40, 50]
results_all = {}
method_name = "Random PARAFAC2" if USE_RANDOM_PARAFAC2 else "Factorize-Reconstruct"

for true_rank in true_ranks:
    if USE_RANDOM_PARAFAC2:
        adata, r2x = create_random_parafac2_data(
            true_rank,
            n_conditions=TARGET_N_CONDITIONS,
            cells_per_cond=TARGET_CELLS_PER_COND,
            n_genes=TARGET_N_GENES
        )
    else:
        adata, r2x = create_known_rank_data(
            adata_full, true_rank,
            cells_per_cond=TARGET_CELLS_PER_COND,
            n_genes=TARGET_N_GENES
        )
    
    ranks_to_test = get_test_ranks(true_rank)
    print(f"\nRunning rank selection on known-rank-{true_rank} data...")
    print(f"Data shape: {adata.shape}, Testing ranks: {ranks_to_test}")
    
    results = run_rank_selection(
        adata, ranks=ranks_to_test, condition_key="sample",
        n_folds=5, n_iter_max=100, tol=1e-6, ot_epsilon=0.05, random_state=42
    )
    
    results_all[true_rank] = {"results": results, "r2x_fit": r2x}
    print(f"True rank: {true_rank} validation complete")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, true_rank in zip(axes, true_ranks):
    results = results_all[true_rank]['results']
    ranks = results["rank"].values
    scores = results["ot_score_mean"].values
    stds = results["ot_score_std"].values
    
    ax.errorbar(ranks, scores, yerr=stds, marker='o', markersize=7, 
                linewidth=2, capsize=4, color='steelblue', label='OT Score')
    ax.axvline(true_rank, color='green', linestyle='-', linewidth=3, 
               label=f'True Rank ({true_rank})', alpha=0.8)
    
    ax.set_xlabel('Rank', fontsize=11, fontweight='bold')
    ax.set_ylabel('OT Score', fontsize=11, fontweight='bold')
    ax.set_title(f'True Rank = {true_rank}', fontsize=12, fontweight='bold')
    ax.set_xticks(ranks)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Rank Selection Validation ({method_name})', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = f"/home/nthomas/cellcommunication-Pf2/cellcommunicationpf2/rank_selection/output/validation_{method_name.lower().replace('-', '_').replace(' ', '_')}.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: {output_path}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for true_rank in true_ranks:
    data = results_all[true_rank]
    min_idx = data['results']["ot_score_mean"].idxmin()
    best_rank = data['results'].loc[min_idx, "rank"]
    print(f"True={true_rank}, Best OT Rank={best_rank}, R2X@fit={data['r2x_fit']:.4f}")

plt.show()
