"""Unit tests for cccrise.utils."""

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cccrise.utils import (
    _gene_vector,
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    expression_product_matrix,
    resample,
)


def test_gene_vector_sparse_and_dense_equivalence():
    n_cells, n_genes = 20, 5
    raw_dense = np.random.uniform(0.0, 10.0, size=(n_cells, n_genes)).astype(np.float32)
    genes = [f"Gene_{i}" for i in range(n_genes)]
    obs = pd.DataFrame(index=[f"c_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=genes)

    adata_dense = anndata.AnnData(X=raw_dense, obs=obs, var=var)
    adata_csr = anndata.AnnData(X=sp.csr_matrix(raw_dense), obs=obs, var=var)
    adata_csc = anndata.AnnData(X=sp.csc_matrix(raw_dense), obs=obs, var=var)

    target_gene = "Gene_2"
    vec_dense = _gene_vector(adata_dense, target_gene)
    vec_csr = _gene_vector(adata_csr, target_gene)
    vec_csc = _gene_vector(adata_csc, target_gene)

    assert isinstance(vec_dense, np.ndarray)
    assert vec_dense.shape == (n_cells,)
    np.testing.assert_allclose(vec_dense, raw_dense[:, 2])
    np.testing.assert_allclose(vec_csr, raw_dense[:, 2])
    np.testing.assert_allclose(vec_csc, raw_dense[:, 2])


def test_resample_stratified():
    n_cells = 40
    samples = ["S1", "S2", "S3"]
    obs = pd.DataFrame(
        {"sample": [samples[i % len(samples)] for i in range(n_cells)]},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    X = np.random.rand(n_cells, 2)
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    resampled = resample(adata, condition_name="sample", random_seed=42)

    assert resampled.n_obs == n_cells
    assert resampled.n_vars == 2

    # Verify that the cell count per condition is preserved exactly
    for s in samples:
        orig_count = (adata.obs["sample"] == s).sum()
        res_count = (resampled.obs["sample"] == s).sum()
        assert orig_count == res_count


def test_expression_product_matrix():
    n_cells_1, n_cells_2 = 4, 3
    expr1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    expr2 = np.array([[2.0, 10.0], [0.0, 5.0], [3.0, 1.0]], dtype=np.float32)

    obs1 = pd.DataFrame(index=[f"sender_{i}" for i in range(n_cells_1)])
    obs2 = pd.DataFrame(index=[f"receiver_{j}" for j in range(n_cells_2)])
    var = pd.DataFrame(index=["LIG", "REC"])

    X1 = anndata.AnnData(X=expr1, obs=obs1, var=var)
    X2 = anndata.AnnData(X=expr2, obs=obs2, var=var)

    df = expression_product_matrix(X1, X2, ligand="LIG", receptor="REC")

    assert df.shape == (n_cells_1, n_cells_2)
    assert list(df.index) == list(obs1.index)
    assert list(df.columns) == list(obs2.index)

    # LIG in X1 is column 0: [1, 3, 5, 0]
    # REC in X2 is column 1: [10, 5, 1]
    expected = np.outer([1.0, 3.0, 5.0, 0.0], [10.0, 5.0, 1.0])
    np.testing.assert_allclose(df.values, expected)


def test_add_obs_cmp_label_and_unique_one():
    n_cells = 50
    adata = anndata.AnnData(
        X=np.zeros((n_cells, 2)),
        obs=pd.DataFrame(index=[f"c_{i}" for i in range(n_cells)]),
    )
    # 2 components
    rng = np.random.default_rng(0)
    adata.obsm["sc_B"] = rng.uniform(0, 1, size=(n_cells, 2))

    # Add label for top 20% of component 1
    adata = add_obs_cmp_label(adata, cmp=1, pos=True, top_perc=20, type="sender")
    assert "Cmp1" in adata.obs.columns
    assert adata.obs["Cmp1"].dtype == bool
    # ~20% of cells should be True (10 cells out of 50)
    assert adata.obs["Cmp1"].sum() == 10

    # Add unique label string
    adata = add_obs_cmp_unique_one(adata, cmp=1)
    assert "Label" in adata.obs.columns
    assert (adata.obs["Label"] == "Cmp1").sum() == 10
    assert (adata.obs["Label"] == "NoLabel").sum() == 40
