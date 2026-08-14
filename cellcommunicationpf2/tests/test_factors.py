"""Tests for the factors module (xarray NetCDF serialization & reconstruction)."""

import tempfile

import anndata
import numpy as np
import pandas as pd

from cellcommunicationpf2.factors import (
    factors_to_xarray,
    load_factors_netcdf,
    load_factors_to_anndata,
    save_factors_netcdf,
)


def test_factors_to_xarray_and_netcdf_roundtrip():
    n_cond = 5
    rise_rank = 6
    cp_rank = 4
    n_lr = 10
    n_genes = 20

    # Synthetic factor matrices
    A_cp = np.random.rand(n_cond, cp_rank)
    B_cp = np.random.rand(rise_rank, cp_rank)
    C_cp = np.random.rand(rise_rank, cp_rank)
    D_cp = np.random.rand(n_lr, cp_rank)
    weights_cp = np.ones(cp_rank)
    lr_symbols = [f"L{i}_R{i}" for i in range(n_lr)]
    cond_names = [f"Sample_{i}" for i in range(n_cond)]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]

    A_pf2 = np.random.rand(n_cond, rise_rank)
    B_pf2 = np.eye(rise_rank)
    C_pf2 = np.random.rand(n_genes, rise_rank)

    ds = factors_to_xarray(
        cpd_factors=[A_cp, B_cp, C_cp, D_cp],
        cpd_weights=weights_cp,
        filtered_lr_pairs=lr_symbols,
        condition_names=cond_names,
        gene_names=gene_names,
        pf2_factors=[A_pf2, B_pf2, C_pf2],
        r2x_cp=0.85,
        r2x_pf2=0.15,
        attrs={"condition_key": "sample_id", "study": "unit_test"},
    )

    assert ds.sizes["condition"] == n_cond
    assert ds.sizes["rise_component"] == rise_rank
    assert ds.sizes["cp_component"] == cp_rank
    assert ds.sizes["lr_pair"] == n_lr
    assert ds.sizes["gene"] == n_genes
    assert ds.attrs["r2x_cp"] == 0.85

    # Test NetCDF save & load
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        save_factors_netcdf(ds, tmp.name)
        loaded_ds = load_factors_netcdf(tmp.name)

        assert loaded_ds["cp_A"].dtype == np.float32
        assert loaded_ds["pf2_C"].dtype == np.float32
        np.testing.assert_allclose(loaded_ds["cp_A"].values, A_cp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(loaded_ds["cp_D"].values, D_cp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            loaded_ds["pf2_C"].values, C_pf2, rtol=1e-5, atol=1e-6
        )

        # Test conversion to lightweight AnnData
        adata_light = load_factors_to_anndata(loaded_ds)
        assert adata_light.shape == (n_cond, 0)
        assert "A" in adata_light.uns
        assert "D" in adata_light.uns
        assert len(adata_light.uns["lr_pairs"]) == n_lr


def test_load_factors_with_sc_projection():
    n_cells = 30
    n_genes = 15
    rise_rank = 4
    cp_rank = 3
    n_cond = 2
    n_lr = 5

    # Create dummy AnnData
    X_counts = np.random.poisson(lam=2.0, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame(
        {
            "sample": [f"S{i % n_cond}" for i in range(n_cells)],
            "celltype": ["T_cell" if i % 2 == 0 else "B_cell" for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"G_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X_counts, obs=obs, var=var)

    A_cp = np.random.rand(n_cond, cp_rank)
    B_cp = np.random.rand(rise_rank, cp_rank)
    C_cp = np.random.rand(rise_rank, cp_rank)
    D_cp = np.random.rand(n_lr, cp_rank)
    weights_cp = np.ones(cp_rank)

    A_pf2 = np.random.rand(n_cond, rise_rank)
    B_pf2 = np.eye(rise_rank)
    C_pf2 = np.random.rand(n_genes, rise_rank)

    ds = factors_to_xarray(
        cpd_factors=[A_cp, B_cp, C_cp, D_cp],
        cpd_weights=weights_cp,
        filtered_lr_pairs=[f"L{i}_R{i}" for i in range(n_lr)],
        condition_names=[f"S{i}" for i in range(n_cond)],
        gene_names=[f"G_{i}" for i in range(n_genes)],
        pf2_factors=[A_pf2, B_pf2, C_pf2],
        attrs={"condition_key": "sample"},
    )

    out = load_factors_to_anndata(ds, adata=adata, do_embedding=True)

    assert "projections" in out.obsm
    assert out.obsm["projections"].shape == (n_cells, rise_rank)
    assert "sc_B" in out.obsm
    assert out.obsm["sc_B"].shape == (n_cells, cp_rank)
    assert "rc_C" in out.obsm
    assert out.obsm["rc_C"].shape == (n_cells, cp_rank)
    assert "PaCMAP" in out.obsm
    assert out.obsm["PaCMAP"].shape == (n_cells, 2)
