"""
Unit tests for loading a previously-fit RISE/CP result instead of
refitting, used by `run_significance` / `run_anova_bootstrap` /
`run_component_stability` / `run_alt_comm_score`.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
from parafac2.parafac2 import parafac2_nd, store_pf2

from cccrise.factors import factors_to_xarray, save_factors_netcdf
from cccrise.import_data import add_cond_idxs
from cccrise.tensor import calculate_interaction_tensor
from cccrise.utils import load_rise_reference
from cccrise.validation.anova_stability import fit_anova_pipeline, load_anova_reference
from cccrise.validation.component_stability import (
    fit_stability_pipeline,
    load_stability_reference,
)


def _synthetic_cohort(
    n_conditions: int = 8, cells_per_cond: int = 12, n_genes: int = 8, seed: int = 7
) -> anndata.AnnData:
    rng = np.random.default_rng(seed)
    n_cells = n_conditions * cells_per_cond
    X = rng.exponential(scale=1.0, size=(n_cells, n_genes))
    samples = [f"S{i}" for i in range(n_conditions) for _ in range(cells_per_cond)]
    groups = [
        "Ctrl" if i % 2 == 0 else "Treat"
        for i in range(n_conditions)
        for _ in range(cells_per_cond)
    ]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"Sample": samples, "Group": groups}),
        var=pd.DataFrame(index=gene_names),
    )
    adata.var["means"] = np.asarray(X.mean(axis=0)).ravel()
    return adata


def _lr_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": ["Gene_0", "Gene_2"],
            "receptor": ["Gene_1", "Gene_3"],
            "interaction_symbol": ["Gene_0_Gene_1", "Gene_2_Gene_3"],
        }
    )


def _write_reference_nc(
    adata: anndata.AnnData,
    condition_column: str,
    rise_rank: int,
    cp_rank: int,
    path,
    random_state: int = 0,
) -> tuple[anndata.AnnData, tuple]:
    """A factors.nc with a real RISE fit and dummy rank-`cp_rank` CP
    placeholders, plus the fitted AnnData for comparison against a reload."""
    fit_adata = add_cond_idxs(adata.copy(), condition_column)
    pf2_out, r2x_pf2 = parafac2_nd(fit_adata, rank=rise_rank, random_state=random_state)
    fit_adata = store_pf2(fit_adata, pf2_out)

    n_cond = fit_adata.obs[condition_column].nunique()
    condition_names = np.unique(fit_adata.obs[condition_column])
    ds = factors_to_xarray(
        cpd_factors=[
            np.ones((n_cond, cp_rank)),
            np.ones((rise_rank, cp_rank)),
            np.ones((rise_rank, cp_rank)),
            np.ones((1, cp_rank)),
        ],
        cpd_weights=np.ones(cp_rank),
        filtered_lr_pairs=["placeholder"],
        condition_names=condition_names,
        gene_names=list(fit_adata.var_names),
        pf2_factors=[
            np.asarray(fit_adata.uns["Pf2_A"]),
            np.asarray(fit_adata.uns["Pf2_B"]),
            np.asarray(fit_adata.varm["Pf2_C"]),
        ],
        pf2_weights=fit_adata.uns["Pf2_weights"],
        pf2_projections=np.asarray(fit_adata.obsm["projections"]),
        cell_ids=fit_adata.obs_names.to_numpy(),
        r2x_pf2=r2x_pf2,
        attrs={"condition_key": condition_column, "complex_sep": "&"},
    )
    save_factors_netcdf(ds, path)
    return fit_adata, pf2_out


def test_load_rise_reference_reproduces_stored_projections_exactly(tmp_path):
    adata = _synthetic_cohort()
    path = tmp_path / "ref.nc"
    fit_adata, _ = _write_reference_nc(
        adata, "Sample", rise_rank=3, cp_rank=2, path=path
    )

    loaded = load_rise_reference(path, adata.copy(), "Sample")

    np.testing.assert_array_equal(
        np.asarray(loaded.obsm["projections"]),
        np.asarray(fit_adata.obsm["projections"]).astype(np.float32),
    )
    # Downstream tensor-building works unchanged off the loaded reference.
    tensor, filtered = calculate_interaction_tensor(
        loaded, _lr_pairs(), complex_sep="&", communication_score="expression_mean"
    )
    assert tensor.shape == (adata.obs["Sample"].nunique(), 3, 3, len(filtered))


def test_load_rise_reference_rejects_rank_mismatch(tmp_path):
    adata = _synthetic_cohort()
    path = tmp_path / "ref.nc"
    _write_reference_nc(adata, "Sample", rise_rank=3, cp_rank=2, path=path)

    with pytest.raises(ValueError, match="rise_rank"):
        load_rise_reference(path, adata.copy(), "Sample", rise_rank=5)


def test_load_rise_reference_rejects_cell_mismatch(tmp_path):
    adata = _synthetic_cohort()
    path = tmp_path / "ref.nc"
    _write_reference_nc(adata, "Sample", rise_rank=3, cp_rank=2, path=path)

    other = _synthetic_cohort(n_conditions=4)
    with pytest.raises(ValueError, match="cells"):
        load_rise_reference(path, other, "Sample")


def test_load_rise_reference_rejects_missing_projections(tmp_path):
    adata = _synthetic_cohort()
    fit_adata = add_cond_idxs(adata.copy(), "Sample")
    pf2_out, r2x_pf2 = parafac2_nd(fit_adata, rank=3, random_state=0)
    fit_adata = store_pf2(fit_adata, pf2_out)
    condition_names = np.unique(fit_adata.obs["Sample"])
    n_cond = fit_adata.obs["Sample"].nunique()
    ds = factors_to_xarray(
        cpd_factors=[
            np.ones((n_cond, 2)),
            np.ones((3, 2)),
            np.ones((3, 2)),
            np.ones((1, 2)),
        ],
        cpd_weights=np.ones(2),
        filtered_lr_pairs=["placeholder"],
        condition_names=condition_names,
        gene_names=list(fit_adata.var_names),
        pf2_factors=[
            np.asarray(fit_adata.uns["Pf2_A"]),
            np.asarray(fit_adata.uns["Pf2_B"]),
            np.asarray(fit_adata.varm["Pf2_C"]),
        ],
        pf2_weights=fit_adata.uns["Pf2_weights"],
        # No pf2_projections/cell_ids -- an old-style file.
        r2x_pf2=r2x_pf2,
        attrs={"condition_key": "Sample", "complex_sep": "&"},
    )
    path = tmp_path / "old.nc"
    save_factors_netcdf(ds, path)

    with pytest.raises(ValueError, match="pf2_projections"):
        load_rise_reference(path, adata.copy(), "Sample")


def test_load_anova_reference_matches_fit_anova_pipeline(tmp_path):
    adata = _synthetic_cohort()
    lr_pairs = _lr_pairs()
    fit = fit_anova_pipeline(
        adata,
        condition_column="Sample",
        group_column="Group",
        control_label="Ctrl",
        other_label="Treat",
        rise_rank=2,
        cp_rank=2,
        lr_pairs=lr_pairs,
        rise_n_iter_max=10,
        cp_n_iter_max=10,
        random_state=0,
    )

    condition_names = np.unique(add_cond_idxs(adata.copy(), "Sample").obs["Sample"])
    ds = factors_to_xarray(
        cpd_factors=fit["factors"],
        cpd_weights=fit["cp_weights"],
        filtered_lr_pairs=list(fit["lr_symbols"]),
        condition_names=condition_names,
        r2x_cp=fit["r2x_cp"],
        attrs={"condition_key": "Sample", "complex_sep": "&"},
    )
    path = tmp_path / "anova_ref.nc"
    save_factors_netcdf(ds, path)

    loaded = load_anova_reference(
        path,
        adata,
        condition_column="Sample",
        group_column="Group",
        control_label="Ctrl",
        other_label="Treat",
    )
    pd.testing.assert_frame_equal(
        loaded["anova"].reset_index(drop=True),
        fit["anova"].reset_index(drop=True),
        check_exact=False,
        rtol=1e-4,
    )


def test_load_anova_reference_rejects_condition_mismatch(tmp_path):
    adata = _synthetic_cohort()
    lr_pairs = _lr_pairs()
    fit = fit_anova_pipeline(
        adata,
        condition_column="Sample",
        group_column="Group",
        control_label="Ctrl",
        other_label="Treat",
        rise_rank=2,
        cp_rank=2,
        lr_pairs=lr_pairs,
        random_state=0,
    )
    condition_names = np.unique(add_cond_idxs(adata.copy(), "Sample").obs["Sample"])
    ds = factors_to_xarray(
        cpd_factors=fit["factors"],
        cpd_weights=fit["cp_weights"],
        filtered_lr_pairs=list(fit["lr_symbols"]),
        condition_names=condition_names,
        r2x_cp=fit["r2x_cp"],
        attrs={"condition_key": "Sample", "complex_sep": "&"},
    )
    path = tmp_path / "anova_ref.nc"
    save_factors_netcdf(ds, path)

    other = _synthetic_cohort(n_conditions=4)
    with pytest.raises(ValueError, match="conditions"):
        load_anova_reference(
            path,
            other,
            condition_column="Sample",
            group_column="Group",
            control_label="Ctrl",
            other_label="Treat",
        )


def test_load_stability_reference_smoke(tmp_path):
    adata = _synthetic_cohort()
    lr_pairs = _lr_pairs()
    reference = fit_stability_pipeline(
        adata.copy(),
        lr_pairs,
        condition_column="Sample",
        rise_rank=2,
        cp_rank=2,
        rise_n_iter_max=10,
        cp_n_iter_max=10,
        random_state=0,
        keep_tensor=True,
    )
    # Rebuild a RISE fit directly so we have real per-cell projections to store
    # (fit_stability_pipeline does not return the fitted AnnData itself).
    fit_adata = add_cond_idxs(adata.copy(), "Sample")
    pf2_out, r2x_pf2 = parafac2_nd(fit_adata, rank=2, random_state=0)
    fit_adata = store_pf2(fit_adata, pf2_out)

    condition_names = np.unique(fit_adata.obs["Sample"])
    ds = factors_to_xarray(
        cpd_factors=reference["cp_factors"],
        cpd_weights=reference["cp_weights"],
        filtered_lr_pairs=list(reference["lr_symbols"]),
        condition_names=condition_names,
        gene_names=list(fit_adata.var_names),
        pf2_factors=[
            np.asarray(fit_adata.uns["Pf2_A"]),
            np.asarray(fit_adata.uns["Pf2_B"]),
            np.asarray(fit_adata.varm["Pf2_C"]),
        ],
        pf2_weights=fit_adata.uns["Pf2_weights"],
        pf2_projections=np.asarray(fit_adata.obsm["projections"]),
        cell_ids=fit_adata.obs_names.to_numpy(),
        r2x_cp=reference["r2x_cp"],
        r2x_pf2=r2x_pf2,
        attrs={"condition_key": "Sample", "complex_sep": "&"},
    )
    path = tmp_path / "stability_ref.nc"
    save_factors_netcdf(ds, path)
    # `load_stability_reference` reads fitting params from the provenance
    # sidecar, not the netCDF attrs -- write a matching one.
    from cccrise.runners.provenance import write_metadata

    write_metadata(
        path,
        {
            "communication_score": "expression_product",
            "hill_k": 0.5,
            "complex_sep": "&",
            "rank_selection": {"rise": {"rank": 2}, "cpd": {"rank": 2}},
        },
    )

    loaded = load_stability_reference(path, adata.copy(), "Sample", lr_pairs)
    assert loaded["tensor"] is not None
    assert loaded["tensor"].shape[0] == adata.obs["Sample"].nunique()
    assert len(loaded["cp_factors"]) == 4
    assert "cpd_diagnostics" in loaded
