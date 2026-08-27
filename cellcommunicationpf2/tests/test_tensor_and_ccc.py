"""Unit tests for cellcommunicationpf2 tensor and ccc modules."""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
from tensorly.cp_tensor import cp_to_tensor

from cellcommunicationpf2.ccc import combine_communication_score, compute_ccc_matrix
from cellcommunicationpf2.ccc_rise import (
    calc_communication_score,
    calc_communication_score_pseudobulk,
    compute_communication_significance,
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
    matrices = [
        rng.uniform(1.0, 5.0, size=(n_rank, len(gene_names))) for _ in range(n_cond)
    ]

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


def test_calc_communication_score_gene_means_recenters_and_clips():
    # A single condition, single eigen-state, two LR genes. Loadings are
    # centered (signed) deviations from a per-gene mean, as produced by RISE.
    gene_names = ["L1", "R1"]
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["L1"],
            "receptor": ["R1"],
            "interaction_symbol": ["L1^R1"],
        }
    )

    # Both loadings negative: naive product on centered values is spuriously
    # positive and unrelated to actual (non-negative) expression.
    matrix = np.array([[-1.0, -2.0]])
    gene_means = np.array([0.5, 1.0])

    tensor, _ = calc_communication_score(
        [matrix], gene_names=gene_names, lr_pairs=lr_pairs, gene_means=gene_means
    )
    # actual_L = max(0.5 - 1.0, 0) = 0, actual_R = max(1.0 - 2.0, 0) = 0
    np.testing.assert_allclose(tensor[0, 0, 0, 0], 0.0)

    # Without gene_means, the sign-ambiguous artifact remains: (-1)*(-2) = 2.
    tensor_uncentered, _ = calc_communication_score(
        [matrix], gene_names=gene_names, lr_pairs=lr_pairs
    )
    np.testing.assert_allclose(tensor_uncentered[0, 0, 0, 0], 2.0)

    # A genuinely above-mean pairing recovers a proportional, non-negative score.
    matrix_pos = np.array([[2.0, 3.0]])
    tensor_pos, _ = calc_communication_score(
        [matrix_pos], gene_names=gene_names, lr_pairs=lr_pairs, gene_means=gene_means
    )
    # actual_L = 0.5 + 2.0 = 2.5, actual_R = 1.0 + 3.0 = 4.0
    np.testing.assert_allclose(tensor_pos[0, 0, 0, 0], 2.5 * 4.0)


@pytest.mark.parametrize(
    "communication_score,expected_fn",
    [
        ("expression_product", lambda a, b: a * b),
        ("expression_mean", lambda a, b: (a + b) / 2.0),
        ("expression_gmean", lambda a, b: np.sqrt(np.clip(a * b, 0.0, None))),
        (
            "expression_hill",
            lambda a, b: np.clip(a * b, 0.0, None) / (0.5 + np.clip(a * b, 0.0, None)),
        ),
    ],
)
def test_combine_communication_score_formulas(communication_score, expected_fn):
    a = np.array([[-1.0, 0.0, 2.0, 5.0]])
    b = np.array([[-1.0, 3.0, 4.0, 0.5]])
    raw_product = a * b
    raw_sum = a + b

    result = combine_communication_score(
        raw_product, raw_sum, communication_score=communication_score, hill_k=0.5
    )
    np.testing.assert_allclose(result, expected_fn(a, b))


def test_combine_communication_score_invalid_raises():
    with pytest.raises(ValueError, match="Not a valid communication_score"):
        combine_communication_score(
            np.array([1.0]), np.array([1.0]), communication_score="bogus"
        )


def test_compute_ccc_matrix_hill_matches_manual_and_is_bounded():
    prot_a_exp = np.array([-2.0, 0.0, 1.0, 10.0])
    prot_b_exp = np.array([3.0, 0.5, -1.0, 4.0])
    hill_k = 0.5

    result = compute_ccc_matrix(
        prot_a_exp, prot_b_exp, communication_score="expression_hill", hill_k=hill_k
    )

    raw_product = np.clip(np.outer(prot_a_exp, prot_b_exp), 0.0, None)
    expected = raw_product / (hill_k + raw_product)
    np.testing.assert_allclose(result, expected)

    # Hill scores are bounded in [0, 1) and non-negative, unlike the raw product.
    assert np.all(result >= 0.0)
    assert np.all(result < 1.0)
    # A large positive product saturates close to 1, a negative-product entry
    # (e.g. prot_a_exp[0]=-2 times prot_b_exp[0]=3) is clipped to a score of 0,
    # not a spurious negative or positive value from the unclipped product.
    np.testing.assert_allclose(result[0, 0], 0.0)


def test_calc_communication_score_hill_scoring_end_to_end():
    gene_names = ["L1", "R1"]
    lr_pairs = pd.DataFrame(
        {"ligand": ["L1"], "receptor": ["R1"], "interaction_symbol": ["L1^R1"]}
    )
    matrix = np.array([[2.0, 3.0]])

    tensor, _ = calc_communication_score(
        [matrix],
        gene_names=gene_names,
        lr_pairs=lr_pairs,
        communication_score="expression_hill",
        hill_k=0.5,
    )
    expected = (2.0 * 3.0) / (0.5 + 2.0 * 3.0)
    np.testing.assert_allclose(tensor[0, 0, 0, 0], expected)


@pytest.mark.parametrize(
    "communication_score",
    ["expression_product", "expression_mean", "expression_gmean", "expression_hill"],
)
def test_compute_communication_significance_works_across_scoring_strategies(
    communication_score,
):
    rng = np.random.default_rng(0)
    n_genes = 4
    gene_names = ["L1", "R1", "L2", "R2"]
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "interaction_symbol": ["L1^R1", "L2^R2"],
        }
    )

    n_cells_per_cond = 12
    rank = 3
    X_list = [rng.uniform(0.0, 5.0, size=(n_cells_per_cond, n_genes)) for _ in range(2)]
    projections = []
    for _ in range(2):
        raw = rng.normal(size=(n_cells_per_cond, rank))
        q, _ = np.linalg.qr(raw)
        projections.append(q[:, :rank])

    (
        interaction_tensor,
        pvalue_tensor,
        null_mean_tensor,
        null_std_tensor,
        filtered_lr_pairs,
    ) = compute_communication_significance(
        X_list=X_list,
        projections=projections,
        gene_names=gene_names,
        lr_pairs=lr_pairs,
        n_perms=25,
        random_state=0,
        communication_score=communication_score,
        validate=True,
    )

    n_lr = len(filtered_lr_pairs)
    expected_shape = (2, rank, rank, n_lr)
    assert interaction_tensor.shape == expected_shape
    assert pvalue_tensor.shape == expected_shape
    assert null_mean_tensor.shape == expected_shape
    assert null_std_tensor.shape == expected_shape

    assert np.all(pvalue_tensor >= 0.0) and np.all(pvalue_tensor <= 1.0)
    assert np.all(np.isfinite(null_mean_tensor))
    assert np.all(null_std_tensor >= 0.0)

    if communication_score in ("expression_gmean", "expression_hill"):
        assert np.all(interaction_tensor >= 0.0)
    if communication_score == "expression_hill":
        assert np.all(interaction_tensor < 1.0)


def test_compute_communication_significance_gene_means_keeps_tensor_nonnegative():
    # Raw (uncentered) expression projected through a signed projection can
    # itself be negative; gene_means should recenter and clip it, just like
    # calc_communication_score, so the null model stays sign-safe (#139).
    rng = np.random.default_rng(1)
    n_genes = 4
    gene_names = ["L1", "R1", "L2", "R2"]
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "interaction_symbol": ["L1^R1", "L2^R2"],
        }
    )
    n_cells_per_cond = 10
    rank = 2
    X_list = [rng.uniform(0.0, 5.0, size=(n_cells_per_cond, n_genes)) for _ in range(2)]
    gene_means = X_list[0].mean(axis=0)
    projections = []
    for _ in range(2):
        raw = rng.normal(size=(n_cells_per_cond, rank))
        q, _ = np.linalg.qr(raw)
        projections.append(q[:, :rank])

    interaction_tensor, pvalue_tensor, _, _, _ = compute_communication_significance(
        X_list=X_list,
        projections=projections,
        gene_names=gene_names,
        lr_pairs=lr_pairs,
        n_perms=10,
        random_state=1,
        gene_means=gene_means,
        validate=True,
    )

    assert np.all(interaction_tensor >= 0.0)
    assert np.all(pvalue_tensor >= 0.0) and np.all(pvalue_tensor <= 1.0)


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
            [0.0, 0.0],  # s1, ct_A
            [6.0, 8.0],  # s1, ct_B
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
    dfs_mean = pseudobulk_X(
        adata, condition_name="sample", groupby="celltype", type="mean"
    )
    assert len(dfs_mean) == 2  # 2 samples
    # dfs_mean[0] is for s0 (genes x celltypes)
    np.testing.assert_allclose(dfs_mean[0].loc["gene1", "ct_A"], 10.0)
    np.testing.assert_allclose(dfs_mean[0].loc["gene2", "ct_A"], 0.0)
    np.testing.assert_allclose(dfs_mean[0].loc["gene1", "ct_B"], 20.0)
    np.testing.assert_allclose(dfs_mean[0].loc["gene2", "ct_B"], 4.0)

    # Test fraction aggregation
    dfs_frac = pseudobulk_X(
        adata, condition_name="sample", groupby="celltype", type="fraction"
    )
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
