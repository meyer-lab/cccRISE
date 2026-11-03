"""Unit tests for cccrise tensor and ccc modules."""

from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
from tensorly.cp_tensor import cp_to_tensor

from cccrise.ccc import combine_communication_score, compute_ccc_matrix
from cccrise.ccc_rise import (
    calc_communication_score,
    calc_communication_score_pseudobulk,
    calculate_interaction_tensor,
    fit_rise,
    run_ccc_rise_workflow,
    standardize_cc_pf2,
)
from cccrise.factors import load_factors_to_anndata, save_ccc_rise_fit
from cccrise.import_data import add_cond_idxs
from cccrise.runners.datasets import COMMUNICATION_SCORES
from cccrise.tensor import calculate_fms_rise
from cccrise.utils import pseudobulk_X


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


def _synthetic_adata(n_cells=30, n_cond=2, n_pairs=3, seed=42):
    """Counts AnnData with interleaved conditions, centering means and LR pairs."""
    rng = np.random.default_rng(seed)
    gene_names = [f"{kind}{i}" for i in range(1, n_pairs + 1) for kind in ("L", "R")]
    X = rng.poisson(lam=3.0, size=(n_cells, len(gene_names))).astype(np.float32)
    obs = pd.DataFrame(
        {"sample": [f"S{i % n_cond}" for i in range(n_cells)]},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata = anndata.AnnData(X=csr_array(X), obs=obs, var=pd.DataFrame(index=gene_names))
    adata.var["means"] = X.mean(axis=0)
    lr_pairs = pd.DataFrame(
        {
            "ligand": [f"L{i}" for i in range(1, n_pairs + 1)],
            "receptor": [f"R{i}" for i in range(1, n_pairs + 1)],
            "interaction_symbol": [f"L{i}^R{i}" for i in range(1, n_pairs + 1)],
        }
    )
    return adata, lr_pairs


FIT_KW: dict[str, Any] = {
    "rise_n_iter_max": 30,
    "cp_n_iter_max": 30,
    "rise_tol": 1e-4,
    "cp_tol": 1e-4,
}


@pytest.mark.parametrize("communication_score", COMMUNICATION_SCORES)
def test_full_ccc_rise_workflow_synthetic(communication_score):
    rise_rank, cp_rank, n_cells, n_cond = 3, 2, 30, 2
    adata, lr_pairs = _synthetic_adata(n_cells=n_cells, n_cond=n_cond, n_pairs=2)

    fit = run_ccc_rise_workflow(
        adata,
        rise_rank=rise_rank,
        cp_rank=cp_rank,
        lr_pairs=lr_pairs,
        backend="cpu",
        keep_tensor=True,
        communication_score=communication_score,
        **FIT_KW,
    )
    assert set(fit) == {
        "adata",
        "tensor",
        "filtered_lr_pairs",
        "cp_weights",
        "cp_factors",
        "r2x_rise",
        "r2x_cp",
        "diagnostics",
    }
    out_adata, r2x = fit["adata"], fit["r2x_cp"]

    assert isinstance(r2x, float)
    assert 0.0 <= r2x <= 1.0
    assert 0.0 <= fit["r2x_rise"] <= 1.0
    assert out_adata.uns["r2x"] == r2x
    assert out_adata.uns["r2x_pf2"] == fit["r2x_rise"]
    assert fit["tensor"].shape == (n_cond, rise_rank, rise_rank, len(lr_pairs))
    assert set(fit["diagnostics"]) == {
        "MaxModesCollinear",
        "WorstPairCanceling",
    }
    for factor, stored in zip(fit["cp_factors"], "ABCD", strict=True):
        np.testing.assert_array_equal(out_adata.uns[stored], factor)
    assert out_adata.uns["A"].shape == (n_cond, cp_rank)
    assert out_adata.uns["B"].shape == (rise_rank, cp_rank)
    assert out_adata.uns["C"].shape == (rise_rank, cp_rank)
    assert out_adata.uns["D"].shape == (len(lr_pairs), cp_rank)
    assert out_adata.varm["Pf2_C"].shape == (adata.n_vars, rise_rank)
    assert out_adata.obsm["projections"].shape == (n_cells, rise_rank)
    assert out_adata.obsm["sc_B"].shape == (n_cells, cp_rank)
    assert out_adata.obsm["rc_C"].shape == (n_cells, cp_rank)
    assert "PaCMAP" not in out_adata.obsm

    assert (
        run_ccc_rise_workflow(adata, 3, lr_pairs, cp_rank=2, **FIT_KW)["tensor"] is None
    )


def test_calculate_interaction_tensor_matches_per_condition_reference():
    """Densifying every condition at once gives the same tensor as projecting
    one sparse condition slab at a time."""
    adata, lr_pairs = _synthetic_adata(n_cells=40, n_cond=3)
    adata = add_cond_idxs(adata, "sample")
    adata, _ = fit_rise(adata, rank=3, n_iter_max=30, tolerance=1e-4, backend="cpu")

    tensor, filtered = calculate_interaction_tensor(adata, lr_pairs)

    cond_idx = adata.obs["condition_unique_idxs"].to_numpy()
    means = adata.var["means"].to_numpy()
    projected = [
        adata.obsm["projections"][cond_idx == k].T
        @ (cast(Any, adata.X)[cond_idx == k].toarray() - means)
        for k in range(3)
    ]
    expected, expected_lr = calc_communication_score(
        projected,
        gene_names=list(adata.var_names),
        lr_pairs=lr_pairs,
        gene_means=means,
    )
    np.testing.assert_allclose(tensor, expected, rtol=1e-10)
    pd.testing.assert_frame_equal(filtered, expected_lr)


def test_fit_rise_compressed_matches_exact(monkeypatch):
    """Compressed RISE uses `RISE_COMPRESSION_OPTIONS`, returns full-size
    per-cell projections, and fits about as well as exact RISE."""
    import parafac2.parafac2 as pf2_module

    from cccrise import ccc_rise

    calls = []
    real = pf2_module.compress_dataset
    monkeypatch.setattr(
        pf2_module,
        "compress_dataset",
        lambda *a, **kw: calls.append(kw) or real(*a, **kw),
    )
    adata, _ = _synthetic_adata(n_cells=160, n_cond=4, n_pairs=30, seed=3)
    adata = add_cond_idxs(adata, "sample")
    fits = {
        compress: ccc_rise.fit_rise(
            adata.copy(),
            rank=2,
            n_iter_max=200,
            tolerance=1e-8,
            backend="cpu",
            compress=compress,
            random_state=0,
        )
        for compress in (True, False)
    }
    assert len(calls) == 1
    assert calls[0]["n_power_iter"] == ccc_rise.RISE_COMPRESSION_OPTIONS["n_power_iter"]
    assert calls[0]["normalize_slices"] is True
    compressed, exact = fits[True], fits[False]
    assert compressed[0].obsm["projections"].shape == (adata.n_obs, 2)
    assert compressed[1] == pytest.approx(exact[1], abs=0.01)


def test_workflow_compressed_and_exact_agree():
    """`compress` compresses RISE (CP is never compressed); the fits stay
    close to the exact ones."""
    adata, lr_pairs = _synthetic_adata(n_cells=160, n_cond=4, n_pairs=30, seed=3)
    kw: dict[str, Any] = FIT_KW | {
        "rise_n_iter_max": 200,
        "rise_tol": 1e-8,
        "cp_n_iter_max": 500,
        "cp_tol": 1e-10,
    }
    fits = {
        compress: run_ccc_rise_workflow(
            adata.copy(),
            rise_rank=3,
            lr_pairs=lr_pairs,
            cp_rank=2,
            random_state=0,
            backend="cpu",
            compress=compress,
            **kw,
        )
        for compress in (True, False)
    }
    for key in ("r2x_rise", "r2x_cp"):
        assert fits[True][key] == pytest.approx(fits[False][key], abs=0.01), key


def test_run_rise_rank_hands_bicv_the_shared_rise_settings(monkeypatch, tmp_path):
    import cccrise.runners.run_rise_rank as runner
    from cccrise.ccc_rise import RISE_COMPRESSION_OPTIONS, RISE_FIT_OPTIONS
    from cccrise.runners._cli import REPLICATE_COMPRESS
    from cccrise.runners.datasets import DATASETS

    adata, _ = _synthetic_adata()
    captured = {}
    monkeypatch.setattr(
        runner, "bicv", lambda *a, **kw: captured.update(kw) or pd.DataFrame()
    )
    monkeypatch.setattr(runner, "dataset_metadata", lambda spec: {})
    monkeypatch.setattr(runner, "resulting_dims", lambda spec, adata: {})
    spec = DATASETS["alad"]
    monkeypatch.setattr(type(spec), "load", lambda self: adata)
    runner.run_rise_rank_selection(spec, ranks=[2], output_path=tmp_path / "r.parquet")

    assert captured["compress"] is REPLICATE_COMPRESS
    assert captured["compression_kwarg"] == RISE_COMPRESSION_OPTIONS
    assert captured["parafac2_kwarg"] == RISE_FIT_OPTIONS | {"backend": spec.backend}
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--dataset", "alad", "--compress", "none"])


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


@pytest.mark.parametrize("compress", [False, True])
def test_saved_fit_recomputes_its_own_projections(tmp_path, compress):
    """Projections recomputed from a saved fit and the raw data match the
    ones the fit itself produced."""
    adata, lr_pairs = _synthetic_adata(n_cells=160, n_cond=4, n_pairs=30, seed=3)
    fit = run_ccc_rise_workflow(
        adata.copy(),
        rise_rank=3,
        lr_pairs=lr_pairs,
        cp_rank=2,
        random_state=0,
        backend="cpu",
        compress=compress,
        **FIT_KW,
    )
    fit["adata"].obsm["PaCMAP"] = np.random.default_rng(0).normal(size=(160, 2))
    path = tmp_path / "fit.nc"
    save_ccc_rise_fit(path, fit["adata"], fit, "sample")

    out = load_factors_to_anndata(path, adata=adata)
    recomputed, original = out.obsm, fit["adata"].obsm
    for key in ("projections", "sc_B", "rc_C"):
        np.testing.assert_allclose(
            np.asarray(recomputed[key]),
            np.asarray(original[key]),
            atol=1e-5,
            err_msg=key,
        )
    # PaCMAP is stored, at float16, rather than recomputed
    np.testing.assert_allclose(
        np.asarray(recomputed["PaCMAP"]), np.asarray(original["PaCMAP"]), atol=5e-3
    )


def test_projection_recompute_rejects_other_data(tmp_path):
    """The saved fingerprints catch reordered cells and changed centering."""
    adata, lr_pairs = _synthetic_adata(n_cells=60, n_cond=2, n_pairs=5)
    fit = run_ccc_rise_workflow(
        adata.copy(), 2, lr_pairs, cp_rank=1, backend="cpu", **FIT_KW
    )
    path = tmp_path / "fit.nc"
    save_ccc_rise_fit(path, fit["adata"], fit, "sample")

    shifted = adata.copy()
    shifted.var["means"] *= 1.01
    for other in (adata[::-1].copy(), shifted):
        with pytest.raises(ValueError, match="do not match|differ"):
            load_factors_to_anndata(path, adata=other)
