"""Unit tests for cccrise.validation.nichenet (no network access)."""

import anndata
import numpy as np
import pandas as pd
import pytest

from cccrise.anova import adjust_pvalues
from cccrise.validation.nichenet import (
    _receiver_population,
    component_ligand_scores,
    expressed_genes,
    ligand_activity_auroc,
    ligand_weights,
    load_ligand_target_matrix,
    receiver_ranking,
    top_ligand_tests,
    top_pair_ligands,
)


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
    (`rc_C`) that is higher for those cells than for the rest."""
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


def test_receiver_population_excludes_middle():
    adata = _make_synthetic_adata(n_cells=200, n_high=40, random_state=1)
    kept, pos_mask = _receiver_population(adata, cmp=1, top_perc=20)
    assert pos_mask.sum() == pytest.approx(40, abs=2)
    assert (~pos_mask).sum() == pytest.approx(40, abs=2)
    assert kept.n_obs < adata.n_obs


def test_receiver_ranking_recovers_target_genes():
    target_genes = ["g0", "g1", "g2"]
    adata = _make_synthetic_adata(target_genes=target_genes, n_high=40)
    result = receiver_ranking(adata, cmp=1, top_perc=20)
    assert result is not None
    ranking, n_pos, n_neg = result
    assert set(target_genes).issubset(set(ranking.index[:5]))
    assert n_pos == pytest.approx(40, abs=2) and n_neg == pytest.approx(40, abs=2)


def test_ligand_weights_combines_pairs_sharing_a_ligand():
    lr_pairs = pd.DataFrame({"ligand": ["A", "A", "B"], "receptor": ["R1", "R2", "R3"]})
    D = np.array([[3.0, 0.0], [4.0, 1.0], [1.0, 2.0]])
    weights = ligand_weights(D, lr_pairs)
    np.testing.assert_allclose(weights.loc["A"], [5.0, 1.0])
    np.testing.assert_allclose(weights.loc["B"], [1.0, 2.0])


def test_expressed_genes_keeps_genes_detected_in_either_group():
    pos = np.array([True] * 10 + [False] * 10)
    X = np.zeros((20, 3))
    X[:2, 0] = 1.0  # 20% of the high group only
    X[10:12, 1] = 1.0  # 20% of the low group only
    X[[0, 10], 2] = 1.0  # 10% pooled but only 10% of each group -> kept at 0.1
    adata = anndata.AnnData(
        X=X, var=pd.DataFrame(index=["hi_only", "lo_only", "sparse"])
    )
    assert list(expressed_genes(adata, pos, min_frac=0.15)) == ["hi_only", "lo_only"]
    assert list(expressed_genes(adata, pos, min_frac=0.1)) == [
        "hi_only",
        "lo_only",
        "sparse",
    ]


def test_top_pair_ligands_normalizes_to_scorable_pairs():
    lr_pairs = pd.DataFrame(
        {"ligand": ["U", "A", "B", "C"], "receptor": ["R1", "R2", "R3", "R4"]}
    )
    D = np.array([[1.0], [0.5], [0.3], [0.1]])  # U (unscorable) has the top pair
    # relative to A (the top scorable pair): A 1.0, B 0.6, C 0.2
    assert top_pair_ligands(D, lr_pairs, 1, {"A", "B", "C"}, rel_cutoff=0.4) == [
        "A",
        "B",
    ]
    assert top_pair_ligands(D, lr_pairs, 1, set()) == []


def test_top_ligand_tests_bh_across_components():
    rng = np.random.default_rng(0)
    rows = []
    for cmp, shift in ((1, 0.2), (2, 0.0), (3, -0.2)):
        auroc = rng.normal(0.55, 0.02, 40)
        auroc[:4] += shift
        rows.append(
            pd.DataFrame(
                {
                    "component": cmp,
                    "ligand": [f"L{i}" for i in range(40)],
                    "auroc": auroc,
                    "weight": np.linspace(1, 0.01, 40),
                    "is_top": np.arange(40) < 4,
                }
            )
        )
    tests = top_ligand_tests(pd.concat(rows, ignore_index=True))
    assert tests.loc[0, "direction"] == "up" and tests.loc[2, "direction"] == "down"
    assert tests.loc[0, "p"] < 0.01 and tests.loc[2, "p"] < 0.01
    np.testing.assert_allclose(
        tests["q"], adjust_pvalues(tests["p"].to_numpy(), "fdr_bh")
    )
    assert list(tests["n_top"]) == [4, 4, 4] and list(tests["n_comparison"]) == [
        36,
        36,
        36,
    ]


def test_component_ligand_scores_end_to_end():
    target_genes = ["g0", "g1", "g2"]
    adata = _make_synthetic_adata(target_genes=target_genes, n_high=40)
    # "GZMX" has the top pair but no prior targets, so it neither scores nor
    # sets the top-set scale; LIGA (planted) is the top scorable pair.
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["GZMX", "LIGA", "LIGB", "LIGC", "LIGD"],
            "receptor": ["R0", "RA", "RB", "RC", "RD"],
        }
    )
    adata.uns["D"] = np.array([[1.0], [0.9], [0.1], [0.05], [0.02]])
    lt_df = pd.DataFrame(
        {
            "ligand": ["LIGA"] * 3 + ["LIGB"] * 3 + ["LIGC"] * 3 + ["LIGD"] * 3,
            "target": target_genes
            + ["g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18"],
            "weight": 1.0,
        }
    )
    scores = component_ligand_scores(
        adata, lr_pairs, lt_df, components=[1], top_perc=20
    )
    s = scores.set_index("ligand")
    assert np.isnan(s.loc["GZMX", "auroc"]) and not s.loc["GZMX", "is_top"]
    assert list(s.index[s["is_top"]]) == ["LIGA"]
    assert s.loc["LIGA", "auroc"] > 0.9
    assert s.loc["LIGA", "rel_weight"] == pytest.approx(1.0)
    assert (s.loc[["LIGB", "LIGC", "LIGD"], "auroc"] < 0.8).all()
