"""The pseudobulk cache reproduces `pseudobulk_X` / `pseudobulk_tensor` for
merged labellings."""

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

from cccrise.benchmark import pseudobulk_tensor
from cccrise.ccc_rise import calc_communication_score_pseudobulk
from cccrise.comparators.pseudobulk_cache import (
    group_sums,
    labelled_matrices,
    loading_sums,
    lr_genes,
    merge_groups,
)


def _adata(seed=0):
    rng = np.random.default_rng(seed)
    genes = ["L1", "R1", "R2", "L2", "X"]
    n = 300
    X = sparse.csr_matrix(rng.poisson(1.0, (n, len(genes))).astype(float))
    obs = pd.DataFrame(
        {
            "sample": rng.choice(["s1", "s2", "s3"], n),
            "fine": rng.choice(["a1", "a2", "b1", "c1"], n),
        },
        index=[f"c{i}" for i in range(n)],
    )
    obs.loc[obs.index[:5], "fine"] = "a1"
    obs = obs[~((obs["sample"] == "s3") & (obs["fine"] == "c1"))]  # an empty group
    adata = anndata.AnnData(X=X[: len(obs)], obs=obs, var=pd.DataFrame(index=genes))
    adata.obs["coarse"] = adata.obs["fine"].str[0]
    return adata


LR = pd.DataFrame(
    {
        "ligand": ["L1", "L2"],
        "receptor": ["R1&R2", "R1"],
        "interaction_symbol": ["L1^R1&R2", "L2^R1"],
    }
)


def test_lr_genes_splits_complexes():
    assert lr_genes(LR) == ["L1", "L2", "R1", "R2"]


def _cached_tensor(adata, mapping_of):
    """The tensor of `mapping_of(fine types)` built from the cache, and its
    group names."""
    genes = lr_genes(LR)
    samples, fines = np.unique(adata.obs["sample"]), np.unique(adata.obs["fine"])
    counts, sums = group_sums(
        adata[:, genes].X, adata.obs["sample"], adata.obs["fine"], samples, fines
    )
    matrices, groups = labelled_matrices(counts, sums, fines, genes, mapping_of(fines))
    return calc_communication_score_pseudobulk(matrices, lr_pairs=LR)[0], groups


def test_cache_reproduces_the_coarse_tensor():
    adata = _adata()
    tensor, groups = _cached_tensor(adata, lambda fines: {t: t[0] for t in fines})
    direct, types, _ = pseudobulk_tensor(adata, "sample", "coarse", LR)
    assert list(groups) == list(types)
    np.testing.assert_allclose(tensor, direct)


def test_identity_mapping_reproduces_the_fine_tensor():
    adata = _adata(1)
    tensor, _ = _cached_tensor(adata, lambda fines: {t: t for t in fines})
    direct, _, _ = pseudobulk_tensor(adata, "sample", "fine", LR)
    np.testing.assert_allclose(tensor, direct)


def test_loading_sums_and_merge():
    rng = np.random.default_rng(2)
    samples = np.array(["s1"] * 4 + ["s2"] * 2)
    types = np.array(["a", "b", "a", "b", "a", "a"])
    loadings = rng.random((6, 2))
    sums, active = loading_sums(loadings, samples, types, ["s1", "s2"], ["a", "b"])
    assert sums.shape == (2, 2, 2) and active.shape == (2, 2, 2)
    np.testing.assert_allclose(sums[0, 0], (loadings[[0, 2]] * np.sqrt(4)).sum(axis=0))
    merged, groups = merge_groups(sums, ["a", "b"], {"a": "x", "b": "x"})
    assert list(groups) == ["x"]
    np.testing.assert_allclose(merged[..., 0, :], sums.sum(axis=1))
