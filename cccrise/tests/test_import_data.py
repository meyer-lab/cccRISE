"""Unit tests for cccrise.import_data."""

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from cccrise.import_data import (
    ALAD_PATIENT_OF,
    add_alad_patient_id,
    add_cond_idxs,
    cap_cells_per_condition,
    import_ligand_receptor_pairs,
    prepare_dataset,
    use_gene_symbols,
)


def test_add_cond_idxs():
    n_cells = 30
    conditions = ["cond_B", "cond_A", "cond_C"]
    obs = pd.DataFrame(
        {"sample": [conditions[i % len(conditions)] for i in range(n_cells)]},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=["gene1", "gene2"])
    X = np.ones((n_cells, 2), dtype=np.float32)
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    adata = add_cond_idxs(adata, "sample")
    assert "condition_unique_idxs" in adata.obs.columns

    unique_idxs = adata.obs["condition_unique_idxs"].to_numpy()
    assert unique_idxs.min() == 0
    assert unique_idxs.max() == 2
    assert len(np.unique(unique_idxs)) == 3

    # All cells of the same sample must share the exact same condition index
    for cond in conditions:
        mask = adata.obs["sample"] == cond
        cond_idx_set = set(unique_idxs[mask])
        assert len(cond_idx_set) == 1


def test_prepare_dataset_filtering_and_normalization():
    # Construct 10 cells x 6 genes sparse count matrix
    # Cell 0: all zeros (total counts <= 10, should be filtered out)
    # Cells 1-9: nonzero counts > 10
    # Gene 0: zero across all cells (mean = 0 <= geneThreshold, filtered out)
    # Genes 1-5: expression above threshold
    n_cells, n_genes = 10, 6
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(1, n_cells):
        counts[i, 1:] = np.arange(1, n_genes) * (i + 1)

    obs = pd.DataFrame(
        {"patient": [f"P{i % 2}" for i in range(n_cells)]},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    adata = anndata.AnnData(X=csr_array(counts), obs=obs, var=var)

    processed = prepare_dataset(
        adata, condition_name="patient", geneThreshold=0.05, normalize=True
    )

    # Cell 0 filtered (9 remaining cells)
    assert processed.n_obs == 9
    # Gene 0 filtered (5 remaining genes)
    assert processed.n_vars == 5
    assert "gene_0" not in processed.var_names

    # Check that condition_unique_idxs and means exist
    assert "condition_unique_idxs" in processed.obs.columns
    assert "means" in processed.var.columns
    assert len(processed.var["means"]) == 5
    assert np.all(processed.var["means"] > 0)


def test_import_ligand_receptor_pairs_structure():
    df = import_ligand_receptor_pairs()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "ligand" in df.columns
    assert "receptor" in df.columns
    assert "interaction_symbol" in df.columns

    # Verify formatting of complex components (standardized uppercase)
    ligands = df["ligand"].astype(str)
    receptors = df["receptor"].astype(str)
    assert not ligands.str.contains(" ").any()
    assert not receptors.str.contains(" ").any()


def test_import_alad():
    import os

    from cccrise.import_data import import_alad

    if os.path.exists("/opt/IVCSC/BAL-scRNAseq.raw.h5ad"):
        adata = import_alad(gene_threshold=0.001, normalize=True)
        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert "condition_unique_idxs" in adata.obs.columns
        assert "celltype" in adata.obs.columns
        assert "means" in adata.var.columns


def test_import_alad_brush():
    import os

    from cccrise.import_data import import_alad_brush

    if os.path.exists("/opt/IVCSC/alad.brush.sc.h5ad"):
        adata = import_alad_brush(gene_threshold=0.01, normalize=True)
        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert "condition_unique_idxs" in adata.obs.columns
        assert "means" in adata.var.columns


def _ensembl_indexed_adata() -> anndata.AnnData:
    """An AnnData shaped like IBDverse: Ensembl var_names, symbols in a column."""
    return anndata.AnnData(
        X=np.arange(12, dtype=float).reshape(3, 4),
        var=pd.DataFrame(
            # Two Ensembl IDs map to the same symbol, and one has no symbol.
            {"gene_symbols": ["TGFB1", "ACVR1", "TGFB1", "nan"]},
            index=["ENSG01", "ENSG02", "ENSG03", "ENSG04"],
        ),
    )


def test_use_gene_symbols_reindexes_and_keeps_the_first_symbol_matchable():
    """The LR table is keyed by symbol, so the bare symbol must survive."""
    adata = use_gene_symbols(_ensembl_indexed_adata())

    names = list(adata.var_names)
    # First occurrence keeps the bare symbol; the duplicate is suffixed.
    assert names[0] == "TGFB1"
    assert names[1] == "ACVR1"
    assert names[2].startswith("TGFB1") and names[2] != "TGFB1"
    # A gene with no symbol falls back to its own identifier.
    assert names[3] == "ENSG04"
    # The original identifiers stay recoverable.
    assert list(adata.var["ensembl_id"]) == ["ENSG01", "ENSG02", "ENSG03", "ENSG04"]


def test_use_gene_symbols_is_a_noop_without_the_column():
    """Datasets already keyed by symbol, like ALAD, must pass through untouched."""
    adata = anndata.AnnData(
        X=np.zeros((2, 2)), var=pd.DataFrame(index=["TGFB1", "ACVR1"])
    )
    result = use_gene_symbols(adata)
    assert list(result.var_names) == ["TGFB1", "ACVR1"]
    assert "ensembl_id" not in result.var


def test_add_alad_patient_id_pairs_the_two_known_multi_sample_patients():
    """HS63 and HS65 each contributed two dsco_id samples, and must share a
    `patient_id` so `validation`'s pairing can carry them together."""
    dsco_ids = [
        "ALAD1-HS1-E1",
        "ALAD1-HS63-E1",
        "ALAD1-HS63-E2",
        "ALAD1-HS65-E1",
        "ALAD1-HS65-E2",
    ]
    adata = anndata.AnnData(
        X=np.zeros((len(dsco_ids), 1)),
        obs=pd.DataFrame({"dsco_id": dsco_ids}),
    )

    result = add_alad_patient_id(adata)

    patient_id = dict(zip(result.obs["dsco_id"], result.obs["patient_id"], strict=True))
    assert patient_id["ALAD1-HS63-E1"] == patient_id["ALAD1-HS63-E2"]
    assert patient_id["ALAD1-HS65-E1"] == patient_id["ALAD1-HS65-E2"]
    assert patient_id["ALAD1-HS63-E1"] != patient_id["ALAD1-HS65-E1"]


def test_add_alad_patient_id_leaves_a_singleton_dsco_id_unpaired():
    """A dsco_id with no known sibling must map to a `patient_id` no other
    condition shares, so `_paired_conditions` never groups it with anything."""
    dsco_ids = ["ALAD1-HS1-E1", "ALAD1-HS2-E1"]
    adata = anndata.AnnData(
        X=np.zeros((len(dsco_ids), 1)),
        obs=pd.DataFrame({"dsco_id": dsco_ids}),
    )

    result = add_alad_patient_id(adata)

    assert list(result.obs["patient_id"]) == dsco_ids
    assert set(ALAD_PATIENT_OF) & set(dsco_ids) == set()


def _conditions(sizes: dict[str, int]) -> np.ndarray:
    return np.concatenate([[name] * n for name, n in sizes.items()])


def test_cap_cells_per_condition_caps_large_and_keeps_small_conditions():
    conditions = _conditions({"big": 50, "small": 5, "exact": 10})

    keep = cap_cells_per_condition(conditions, max_cells=10, seed=0)

    kept = pd.Series(conditions[keep]).value_counts()
    assert kept["big"] == 10
    assert kept["small"] == 5
    assert kept["exact"] == 10


def test_cap_cells_per_condition_is_deterministic_in_the_seed():
    conditions = _conditions({"a": 100, "b": 80})

    first = cap_cells_per_condition(conditions, max_cells=20, seed=3)
    again = cap_cells_per_condition(conditions, max_cells=20, seed=3)
    other = cap_cells_per_condition(conditions, max_cells=20, seed=4)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_cap_cells_per_condition_nests_smaller_caps_inside_larger_ones():
    """A smaller cap's cells are a subset of a larger cap's, same seed."""
    conditions = _conditions({"a": 100, "b": 60})

    small = cap_cells_per_condition(conditions, max_cells=10, seed=0)
    large = cap_cells_per_condition(conditions, max_cells=40, seed=0)

    assert np.all(large[small])


def test_cap_cells_per_condition_ignores_the_other_conditions():
    """A condition's draw depends on its own name and cells, not its neighbors'."""
    alone = _conditions({"a": 100})
    with_neighbor = _conditions({"z": 30, "a": 100})

    keep_alone = cap_cells_per_condition(alone, max_cells=10, seed=0)
    keep_with = cap_cells_per_condition(with_neighbor, max_cells=10, seed=0)

    np.testing.assert_array_equal(keep_alone, keep_with[30:])


def test_cap_cells_per_condition_draws_only_eligible_cells():
    conditions = _conditions({"a": 30})
    eligible = np.arange(30) % 2 == 0  # 15 eligible cells

    keep = cap_cells_per_condition(conditions, max_cells=10, eligible=eligible)

    assert keep.sum() == 10
    assert not np.any(keep & ~eligible)

    keep_all = cap_cells_per_condition(conditions, max_cells=20, eligible=eligible)
    np.testing.assert_array_equal(keep_all, eligible)
