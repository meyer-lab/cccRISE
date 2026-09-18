"""Unit tests for cccrise.nichenet (no network access)."""

import anndata
import numpy as np
import pandas as pd
import pytest

from cccrise.nichenet import (
    _receiver_population,
    component_accuracy_scores,
    ligand_activity_auroc,
    load_ligand_target_matrix,
    permutation_null_aurocs,
    receiver_celltypes_for_component,
    receiver_de_ranking,
    top_lr_pairs_for_component,
)


def test_top_lr_pairs_for_component_ranks_by_abs_loading():
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["L1", "L2", "L3", "L4"],
            "receptor": ["R1", "R2", "R3", "R4"],
        }
    )
    # cmp 1 loadings: strongly negative should still rank above a small positive.
    D = np.array(
        [
            [-5.0, 0.1],
            [1.0, 0.2],
            [0.5, 0.3],
            [3.0, 0.4],
        ]
    )

    top = top_lr_pairs_for_component(D, lr_pairs, cmp=1, top_m=2)

    assert list(top["ligand"]) == ["L1", "L4"]
    assert list(top["receptor"]) == ["R1", "R4"]
    np.testing.assert_allclose(top["loading"].to_numpy(), [-5.0, 3.0])


def test_top_lr_pairs_for_component_truncates_to_top_m():
    lr_pairs = pd.DataFrame({"ligand": ["L1", "L2"], "receptor": ["R1", "R2"]})
    D = np.array([[1.0], [2.0]])

    top = top_lr_pairs_for_component(D, lr_pairs, cmp=1, top_m=10)
    assert len(top) == 2


def test_ligand_activity_auroc_perfect_separation():
    # Targets rank at the very top of de_ranking -> AUROC should be 1.0.
    genes = [f"g{i}" for i in range(10)]
    scores = list(range(10, 0, -1))  # g0 highest score ... g9 lowest
    de_ranking = pd.Series(scores, index=genes)

    lt_df = pd.DataFrame(
        {
            "ligand": ["LIGA"] * 3,
            "target": ["g0", "g1", "g2"],
            "weight": [0.9, 0.8, 0.7],
        }
    )

    auroc = ligand_activity_auroc(de_ranking, "LIGA", lt_df)
    assert auroc == pytest.approx(1.0)


def test_ligand_activity_auroc_no_targets_returns_none():
    de_ranking = pd.Series([1.0, 2.0], index=["g0", "g1"])
    lt_df = pd.DataFrame({"ligand": [], "target": [], "weight": []})

    assert ligand_activity_auroc(de_ranking, "MISSING_LIGAND", lt_df) is None


def test_ligand_activity_auroc_all_positive_returns_none():
    de_ranking = pd.Series([1.0, 2.0], index=["g0", "g1"])
    lt_df = pd.DataFrame(
        {"ligand": ["LIGA", "LIGA"], "target": ["g0", "g1"], "weight": [0.9, 0.8]}
    )

    assert ligand_activity_auroc(de_ranking, "LIGA", lt_df) is None


def test_load_ligand_target_matrix_missing_cache_raises(tmp_path):
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        load_ligand_target_matrix(cache_path=missing)


def _make_synthetic_adata(
    n_cells: int = 200,
    n_genes: int = 30,
    n_high: int = 40,
    target_genes: list[str] | None = None,
    random_state: int = 0,
) -> anndata.AnnData:
    """Build a small AnnData where the first `n_high` cells have elevated
    expression of `target_genes` and a continuous receiver-eigenstate loading
    (`rc_C`) that's clearly separated between those cells and the rest, so a
    genuine top-vs-bottom percentile split (not just top-vs-rest) recovers
    them as component 1's "high" group."""
    rng = np.random.default_rng(random_state)
    gene_names = [f"g{i}" for i in range(n_genes)]
    if target_genes is None:
        target_genes = gene_names[:5]
    target_idx = [gene_names.index(g) for g in target_genes]

    X = rng.normal(loc=1.0, scale=0.3, size=(n_cells, n_genes))
    X[:n_high, target_idx] += 5.0
    X = np.clip(X, 0, None)

    obs = pd.DataFrame(
        {"broad_cell_type": ["CT1"] * n_cells},
        index=[f"c{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    adata = anndata.AnnData(X=X, obs=obs, var=var)

    rc_C = rng.normal(loc=0.0, scale=0.05, size=(n_cells, 1))
    rc_C[:n_high, 0] += 1.0
    adata.obsm["rc_C"] = rc_C

    return adata


def test_receiver_population_within_celltype_excludes_middle():
    adata = _make_synthetic_adata(n_cells=200, n_high=40, random_state=1)
    kept, pos_mask, n_pos, n_neg = _receiver_population(
        adata,
        cmp=1,
        receiver_mode="within_celltype",
        celltype_col="broad_cell_type",
        celltype="CT1",
        top_perc=20,
    )
    assert n_pos == pytest.approx(40, abs=2)
    assert n_neg == pytest.approx(40, abs=2)
    assert kept.n_obs == n_pos + n_neg
    assert kept.n_obs < adata.n_obs
    assert pos_mask.sum() == n_pos


def test_receiver_population_vs_rest_keeps_all_cells():
    adata = _make_synthetic_adata(n_cells=200, n_high=40, random_state=1)
    kept, _pos_mask, n_pos, n_neg = _receiver_population(
        adata,
        cmp=1,
        receiver_mode="vs_rest",
        celltype_col="broad_cell_type",
        celltype=None,
        top_perc=20,
    )
    assert kept.n_obs == adata.n_obs
    assert n_pos + n_neg == adata.n_obs


def test_receiver_de_ranking_within_celltype_recovers_target_genes():
    target_genes = ["g0", "g1", "g2"]
    adata = _make_synthetic_adata(target_genes=target_genes, n_high=40)

    de_ranking = receiver_de_ranking(
        adata,
        cmp=1,
        receiver_mode="within_celltype",
        celltype_col="broad_cell_type",
        celltype="CT1",
        top_perc=20,
    )

    top_genes = set(de_ranking.index[:5])
    assert set(target_genes).issubset(top_genes)


def test_receiver_de_ranking_within_celltype_accepts_pooled_celltypes():
    target_genes = ["g0", "g1", "g2"]
    adata = _make_synthetic_adata(target_genes=target_genes, n_high=40)

    de_ranking = receiver_de_ranking(
        adata,
        cmp=1,
        receiver_mode="within_celltype",
        celltype_col="broad_cell_type",
        celltype=["CT1"],
        top_perc=20,
    )

    top_genes = set(de_ranking.index[:5])
    assert set(target_genes).issubset(top_genes)


def test_receiver_de_ranking_vs_rest():
    target_genes = ["g0", "g1", "g2"]
    adata = _make_synthetic_adata(target_genes=target_genes, n_high=40)

    de_ranking = receiver_de_ranking(
        adata,
        cmp=1,
        receiver_mode="vs_rest",
        top_perc=20,
    )

    top_genes = set(de_ranking.index[:5])
    assert set(target_genes).issubset(top_genes)


def test_receiver_de_ranking_invalid_mode_raises():
    adata = _make_synthetic_adata()
    with pytest.raises(ValueError, match="Unknown receiver_mode"):
        receiver_de_ranking(adata, cmp=1, receiver_mode="bogus")  # ty: ignore[invalid-argument-type]


def test_receiver_de_ranking_within_celltype_requires_celltype():
    adata = _make_synthetic_adata()
    with pytest.raises(ValueError, match="celltype.*required"):
        receiver_de_ranking(adata, cmp=1, receiver_mode="within_celltype")


def test_receiver_celltypes_for_component_excludes_dilute_celltypes():
    n_each = 50
    obs = pd.DataFrame({"broad_cell_type": ["A"] * n_each + ["B"] * n_each})
    adata = anndata.AnnData(
        X=np.zeros((2 * n_each, 2)), obs=obs, var=pd.DataFrame(index=["g0", "g1"])
    )
    rc_C = np.zeros((2 * n_each, 1))
    rc_C[:n_each, 0] = 1.0  # A: strong signal
    rc_C[n_each:, 0] = 0.01  # B: negligible signal
    adata.obsm["rc_C"] = rc_C

    result = receiver_celltypes_for_component(
        adata, cmp=1, celltype_col="broad_cell_type", pool_threshold=0.3
    )
    assert result == ["A"]


def test_receiver_celltypes_for_component_pools_comparable_celltypes():
    n_each = 50
    obs = pd.DataFrame(
        {"broad_cell_type": ["A"] * n_each + ["B"] * n_each + ["C"] * n_each}
    )
    adata = anndata.AnnData(
        X=np.zeros((3 * n_each, 2)), obs=obs, var=pd.DataFrame(index=["g0", "g1"])
    )
    rc_C = np.zeros((3 * n_each, 1))
    rc_C[:n_each, 0] = 1.0
    rc_C[n_each : 2 * n_each, 0] = 0.9
    rc_C[2 * n_each :, 0] = 0.1
    adata.obsm["rc_C"] = rc_C

    result = receiver_celltypes_for_component(
        adata, cmp=1, celltype_col="broad_cell_type", pool_threshold=0.3
    )
    assert set(result) == {"A", "B"}


def test_permutation_null_aurocs_null_centered_near_chance():
    adata = _make_synthetic_adata(
        target_genes=["g0", "g1", "g2"], n_high=40, random_state=2
    )
    kept, pos_mask, _, _ = _receiver_population(
        adata,
        cmp=1,
        receiver_mode="within_celltype",
        celltype_col="broad_cell_type",
        celltype="CT1",
        top_perc=20,
    )
    lt_df = pd.DataFrame(
        {
            "ligand": ["LIGA"] * 3,
            "target": ["g0", "g1", "g2"],
            "weight": [0.9, 0.8, 0.7],
        }
    )

    null = permutation_null_aurocs(
        kept, pos_mask, lt_df, ["LIGA"], n_perm=200, random_state=0
    )

    assert "LIGA" in null
    assert null["LIGA"].shape == (200,)
    assert abs(null["LIGA"].mean() - 0.5) < 0.1


def test_permutation_null_aurocs_dedupes_repeated_ligands():
    adata = _make_synthetic_adata(
        target_genes=["g0", "g1", "g2"], n_high=40, random_state=2
    )
    kept, pos_mask, _, _ = _receiver_population(
        adata,
        cmp=1,
        receiver_mode="within_celltype",
        celltype_col="broad_cell_type",
        celltype="CT1",
        top_perc=20,
    )
    lt_df = pd.DataFrame(
        {
            "ligand": ["LIGA"] * 3,
            "target": ["g0", "g1", "g2"],
            "weight": [0.9, 0.8, 0.7],
        }
    )

    null = permutation_null_aurocs(
        kept, pos_mask, lt_df, ["LIGA", "LIGA"], n_perm=50, random_state=0
    )
    assert list(null.keys()) == ["LIGA"]


def test_component_accuracy_scores_end_to_end():
    target_genes = ["g0", "g1", "g2"]
    adata = _make_synthetic_adata(target_genes=target_genes, n_high=40)
    adata.uns["D"] = np.array([[1.0], [0.1]])  # LIGA loads high, LIGB loads low

    lr_pairs = pd.DataFrame({"ligand": ["LIGA", "LIGB"], "receptor": ["RECA", "RECB"]})
    lt_df = pd.DataFrame(
        {
            "ligand": ["LIGA"] * 3 + ["LIGB"] * 3,
            "target": target_genes + ["g20", "g21", "g22"],
            "weight": [0.9, 0.8, 0.7, 0.9, 0.8, 0.7],
        }
    )

    scores = component_accuracy_scores(
        adata,
        lr_pairs=lr_pairs,
        lt_df=lt_df,
        components=[1],
        top_m_lr=2,
        receiver_mode="within_celltype",
        top_perc=20,
        n_perm=200,
        random_state=0,
    )

    assert set(scores["ligand"]) == {"LIGA", "LIGB"}
    assert {"n_pos", "n_neg", "null_mean_auroc", "null_sd_auroc", "p_value"}.issubset(
        scores.columns
    )

    liga_row = scores.loc[scores["ligand"] == "LIGA"].iloc[0]
    ligb_row = scores.loc[scores["ligand"] == "LIGB"].iloc[0]
    assert liga_row["auroc"] > 0.9
    assert liga_row["p_value"] < 0.05
    # LIGB's targets aren't planted as DE, so its AUROC should be near chance.
    assert ligb_row["auroc"] < 0.75
