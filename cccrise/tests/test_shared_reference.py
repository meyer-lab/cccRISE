"""
Unit tests for loading a previously-fit RISE result instead of refitting
(`run_significance`), and for sharing one RISE fit across several CP fits
(`run_alt_comm_score`).
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from cccrise.ccc_rise import calculate_interaction_tensor, fit_ccc_cp, fit_rise
from cccrise.import_data import add_cond_idxs
from cccrise.utils import load_rise_reference
from cccrise.validation.component_stability import (
    fit_stability_pipeline,
    stability_fit,
)

from .helpers import write_rise_reference

FIT_KWARGS = {"rise_n_iter_max": 100, "rise_tol": 1e-6}
CP_FIT_KWARGS = {"cp_n_iter_max": 100, "cp_tol": 1e-6}


def _synthetic_cohort(
    n_conditions: int = 8, cells_per_cond: int = 12, n_genes: int = 8, seed: int = 7
) -> anndata.AnnData:
    rng = np.random.default_rng(seed)
    n_cells = n_conditions * cells_per_cond
    X = rng.exponential(scale=1.0, size=(n_cells, n_genes))
    samples = [f"S{i}" for i in range(n_conditions) for _ in range(cells_per_cond)]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"Sample": samples}),
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


def test_load_rise_reference_reproduces_stored_projections_exactly(tmp_path):
    adata = _synthetic_cohort()
    path = tmp_path / "ref.nc"
    fit_adata = write_rise_reference(adata, "Sample", 3, path)

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
    write_rise_reference(adata, "Sample", 3, path)

    with pytest.raises(ValueError, match="rise_rank"):
        load_rise_reference(path, adata.copy(), "Sample", rise_rank=5)


def test_load_rise_reference_rejects_cell_mismatch(tmp_path):
    adata = _synthetic_cohort()
    path = tmp_path / "ref.nc"
    write_rise_reference(adata, "Sample", 3, path)

    other = _synthetic_cohort(n_conditions=4)
    with pytest.raises(ValueError, match="cells"):
        load_rise_reference(path, other, "Sample")


def test_load_rise_reference_rejects_missing_projections(tmp_path):
    adata = _synthetic_cohort()
    path = tmp_path / "old.nc"
    write_rise_reference(adata, "Sample", 3, path, with_projections=False)

    with pytest.raises(ValueError, match="pf2_projections"):
        load_rise_reference(path, adata.copy(), "Sample")


def test_shared_rise_fit_reproduces_fit_stability_pipeline():
    adata = _synthetic_cohort()
    lr_pairs = _lr_pairs()
    whole = fit_stability_pipeline(
        adata.copy(),
        lr_pairs,
        "Sample",
        rise_rank=3,
        cp_rank=2,
        backend="cpu",
        **FIT_KWARGS,
        **CP_FIT_KWARGS,
    )
    rise_adata, _ = fit_rise(
        add_cond_idxs(adata.copy(), "Sample"),
        rank=3,
        n_iter_max=FIT_KWARGS["rise_n_iter_max"],
        tolerance=FIT_KWARGS["rise_tol"],
        random_state=0,
        backend="cpu",
    )
    split = stability_fit(
        rise_adata,
        fit_ccc_cp(rise_adata, lr_pairs, 2, random_state=0, **CP_FIT_KWARGS),
        "Sample",
    )

    assert split["r2x_rise"] == pytest.approx(whole["r2x_rise"])
    assert split["r2x_cp"] == pytest.approx(whole["r2x_cp"])
    np.testing.assert_array_equal(split["lr_symbols"], whole["lr_symbols"])
    for a, b in zip(split["cp_factors"], whole["cp_factors"], strict=True):
        np.testing.assert_allclose(a, b)


def test_fit_ccc_cp_leaves_shared_rise_fit_untouched():
    adata = _synthetic_cohort()
    rise_adata, _ = fit_rise(
        add_cond_idxs(adata.copy(), "Sample"),
        rank=3,
        n_iter_max=FIT_KWARGS["rise_n_iter_max"],
        tolerance=FIT_KWARGS["rise_tol"],
        random_state=0,
        backend="cpu",
    )
    before = (set(rise_adata.uns), set(rise_adata.obsm))
    for score in ("expression_product", "expression_hill"):
        fit_ccc_cp(
            rise_adata, _lr_pairs(), 2, communication_score=score, **CP_FIT_KWARGS
        )
    assert (set(rise_adata.uns), set(rise_adata.obsm)) == before
