"""Unit tests for cellcommunicationpf2 tensor and ccc modules."""

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from tensorly.cp_tensor import cp_to_tensor

from cellcommunicationpf2.ccc_rise import (
    calc_communication_score,
    calc_communication_score_pseudobulk,
    standardize_cc_pf2,
)
from cellcommunicationpf2.tensor import (
    calculate_fms_rise,
    run_ccc_rise_workflow,
)
from cellcommunicationpf2.utils import pseudobulk_X


def test_calc_communication_score_exact_values():
    n_cond = 2
    n_rank = 3
    gene_names = ["L1", "R1", "L2", "R2"]

    # Construct deterministic projected matrices (n_rank, n_genes) for each condition
    rng = np.random.default_rng(42)
    matrices = [rng.uniform(1.0, 5.0, size=(n_rank, len(gene_names))) for _ in range(n_cond)]

    lr_pairs = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "interaction_symbol": ["L1^R1", "L2^R2"],
        }
    )

    tensor, filtered_lr = calc_communication_score(
        matrices, gene_names=gene_names, lr_pairs=lr_pairs
    )

    assert tensor.shape == (n_cond, n_rank, n_rank, len(lr_pairs))
    assert len(filtered_lr) == len(lr_pairs)

    # Verify each slice T[c, i, j, p] == matrix_c[i, ligand_pos] * matrix_c[j, receptor_pos]
    for c in range(n_cond):
        mat = matrices[c]
        for p, (_, row) in enumerate(lr_pairs.iterrows()):
            lig_idx = gene_names.index(row["ligand"])
            rec_idx = gene_names.index(row["receptor"])
            expected = np.outer(mat[:, lig_idx], mat[:, rec_idx])
            np.testing.assert_allclose(tensor[c, :, :, p], expected, rtol=1e-5)


def test_calc_communication_score_pseudobulk():
    # 2 conditions, 2 cell types, 4 genes (genes as rows, cell types as columns)
    gene_names = ["L1", "R1", "L2", "R2"]
    ct_names = ["CT1", "CT2"]
    rng = np.random.default_rng(42)
    dfs = [
        pd.DataFrame(
            rng.uniform(1.0, 5.0, size=(len(gene_names), len(ct_names))),
            index=gene_names,
            columns=ct_names,
        )
        for _ in range(2)
    ]
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "interaction_symbol": ["L1^R1", "L2^R2"],
        }
    )

    tensor, filtered_lr = calc_communication_score_pseudobulk(
        dfs, gene_names=gene_names, lr_pairs=lr_pairs
    )
    assert tensor.shape == (2, len(ct_names), len(ct_names), len(lr_pairs))
    assert len(filtered_lr) == len(lr_pairs)


def test_pseudobulk_aggregation_mean_and_fraction():
    # 4 cells, 2 samples, 2 celltypes, 2 genes
    X_counts = np.array(
        [
            [10.0, 0.0],  # s0, ct_A
            [20.0, 4.0],  # s0, ct_B
            [0.0, 0.0],   # s1, ct_A
            [6.0, 8.0],   # s1, ct_B
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "sample": ["s0", "s0", "s1", "s1"],
            "celltype": ["ct_A", "ct_B", "ct_A", "ct_B"],
        },
        index=[f"cell_{i}" for i in range(4)],
    )
    var = pd.DataFrame(index=["gene1", "gene2"])
    adata = anndata.AnnData(X=csr_array(X_counts), obs=obs, var=var)

    # Test mean aggregation
    dfs_mean = pseudobulk_X(adata, condition_name="sample", groupby="celltype", type="mean")
    assert len(dfs_mean) == 2  # 2 samples
    # dfs_mean[0] is for s0 (genes x celltypes)
    np.testing.assert_allclose(dfs_mean[0].loc["gene1", "ct_A"], 10.0)
    np.testing.assert_allclose(dfs_mean[0].loc["gene2", "ct_A"], 0.0)
    np.testing.assert_allclose(dfs_mean[0].loc["gene1", "ct_B"], 20.0)
    np.testing.assert_allclose(dfs_mean[0].loc["gene2", "ct_B"], 4.0)

    # Test fraction aggregation
    dfs_frac = pseudobulk_X(adata, condition_name="sample", groupby="celltype", type="fraction")
    np.testing.assert_allclose(dfs_frac[0].loc["gene1", "ct_A"], 1.0)
    np.testing.assert_allclose(dfs_frac[0].loc["gene2", "ct_A"], 0.0)


def test_standardize_cc_pf2_preserves_tensor():
    rank = 3
    rng = np.random.default_rng(123)
    A = rng.uniform(0.5, 2.0, size=(4, rank))
    B = rng.uniform(0.5, 2.0, size=(5, rank))
    C = rng.uniform(0.5, 2.0, size=(5, rank))
    D = rng.uniform(0.5, 2.0, size=(6, rank))
    weights = np.ones(rank)

    orig_tensor = cp_to_tensor((weights, [A, B, C, D]))

    new_weights, new_factors = standardize_cc_pf2(weights, [A, B, C, D])
    std_tensor = cp_to_tensor((new_weights, new_factors))

    # Reconstructed tensor should be identical before and after standardization
    np.testing.assert_allclose(orig_tensor, std_tensor, rtol=1e-5, atol=1e-6)

    # Factors B, C, D must have unit column norms
    for factor in new_factors[1:]:
        norms = np.linalg.norm(factor, axis=0)
        np.testing.assert_allclose(norms, np.ones(rank), rtol=1e-5)


def test_full_ccc_rise_workflow_synthetic():
    n_cells = 30
    n_genes = 6
    rise_rank = 3
    cp_rank = 2
    n_cond = 2

    # Create synthetic AnnData with counts
    rng = np.random.default_rng(42)
    X_counts = rng.poisson(lam=3.0, size=(n_cells, n_genes)).astype(np.float32)
    gene_names = ["L1", "R1", "L2", "R2", "L3", "R3"]
    obs = pd.DataFrame(
        {
            "sample": [f"S{i % n_cond}" for i in range(n_cells)],
            "celltype": ["CT_A" if i % 2 == 0 else "CT_B" for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)
    adata = anndata.AnnData(X=csr_array(X_counts), obs=obs, var=var)

    lr_pairs = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "interaction_symbol": ["L1^R1", "L2^R2"],
        }
    )

    out_adata, r2x = run_ccc_rise_workflow(
        adata,
        rise_rank=rise_rank,
        cp_rank=cp_rank,
        lr_pairs=lr_pairs,
        n_iter_max=30,
        tol=1e-4,
        doEmbedding=False,
    )

    assert isinstance(r2x, float)
    assert 0.0 <= r2x <= 1.0
    assert out_adata.uns["A"].shape == (n_cond, cp_rank)
    assert out_adata.uns["B"].shape == (rise_rank, cp_rank)
    assert out_adata.uns["C"].shape == (rise_rank, cp_rank)
    assert out_adata.uns["D"].shape == (len(lr_pairs), cp_rank)
    assert out_adata.obsm["projections"].shape == (n_cells, rise_rank)
    assert out_adata.obsm["sc_B"].shape == (n_cells, cp_rank)
    assert out_adata.obsm["rc_C"].shape == (n_cells, cp_rank)


def test_calculate_fms_rise():
    n_cond = 3
    rank = 4
    n_genes = 8

    A = np.random.rand(n_cond, rank)
    B = np.eye(rank)
    C = np.random.rand(n_genes, rank)
    weights = np.ones(rank)

    adata1 = anndata.AnnData(shape=(n_cond, n_genes))
    adata1.uns["Pf2_A"] = A
    adata1.uns["Pf2_B"] = B
    adata1.varm["Pf2_C"] = C
    adata1.uns["Pf2_weights"] = weights

    adata2 = adata1.copy()

    # Self-comparison must have perfect FMS = 1.0
    score = calculate_fms_rise(adata1, adata2)
    np.testing.assert_allclose(score, 1.0, atol=1e-5)
