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
    ligand_activities,
    ligand_weights,
    load_ligand_target_matrix,
    receiver_celltypes,
    receiver_ranking,
    receiver_rankings,
    score_percentiles,
    shuffle_prior,
    shuffled_percentiles,
    target_idf,
    top_pair_ligands,
    weighted_percentile,
    weighted_percentile_tests,
)


def test_ligand_activities_perfect_separation_and_undefined():
    # LIGA's targets rank at the very top -> AUROC 1.0; LIGB's target is not
    # ranked and LIGC lists every ranked gene -> undefined.
    genes = [f"g{i}" for i in range(10)]
    de_ranking = pd.Series(range(10, 0, -1), index=genes, dtype=float)
    lt_df = pd.DataFrame(
        {
            "ligand": ["LIGA"] * 3 + ["LIGB"] + ["LIGC"] * 10,
            "target": ["g0", "g1", "g2", "missing", *genes],
        }
    )
    act = ligand_activities(de_ranking, lt_df)
    assert act["LIGA"] == pytest.approx(1.0)
    assert np.isnan(act["LIGB"]) and np.isnan(act["LIGC"])


def test_ligand_activities_downweights_shared_targets():
    # LIGA's targets: g0 (top, also listed by LIGB) and g9 (bottom, its own).
    # Plain AUROC would be 0.5; weighting g0 by log(2/2) = 0 leaves g9 only.
    de_ranking = pd.Series(range(10, 0, -1), index=[f"g{i}" for i in range(10)])
    lt_df = pd.DataFrame(
        {"ligand": ["LIGA", "LIGA", "LIGB"], "target": ["g0", "g9", "g0"]}
    )
    np.testing.assert_allclose(target_idf(lt_df), [0.0, np.log(2), 0.0])
    assert ligand_activities(de_ranking, lt_df)["LIGA"] == pytest.approx(0.0)


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


def test_receiver_population_celltype_receivers():
    # Receivers are the top cells of the high-loading cell type; non-receivers
    # the bottom cells of every type.
    adata = _make_synthetic_adata(n_cells=200, n_high=40, random_state=2)
    adata.obs["broad_cell_type"] = ["REC"] * 40 + ["OTHER"] * 160
    assert receiver_celltypes(adata, 1, "broad_cell_type") == ["REC"]
    kept, pos_mask = _receiver_population(adata, 1, 20, "broad_cell_type")
    assert pos_mask.sum() == 8  # 20% of the 40 receiver-type cells
    assert (kept.obs["broad_cell_type"][pos_mask] == "REC").all()
    assert (kept.obs["broad_cell_type"][~pos_mask] == "OTHER").all()
    assert (~pos_mask).sum() == pytest.approx(40, abs=2)  # 20% of all cells


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


def test_shuffle_prior_keeps_link_count_and_vocabulary():
    # a hub target listed by every ligand loses its hub status
    lt_df = pd.DataFrame(
        {
            "ligand": np.repeat([f"L{i}" for i in range(20)], 15),
            "target": [
                t for i in range(20) for t in ["hub", *(f"g{i}_{j}" for j in range(14))]
            ],
        }
    )
    shuffled = shuffle_prior(lt_df, seed=1)
    assert len(shuffled) == len(lt_df) and not shuffled.duplicated().any()
    assert set(shuffled["ligand"]) <= set(lt_df["ligand"])
    assert set(shuffled["target"]) <= set(lt_df["target"])
    assert (shuffled["target"] == "hub").sum() < 20
    pd.testing.assert_frame_equal(shuffled, shuffle_prior(lt_df, seed=1))


def _random_null(scores: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Null percentiles unrelated to the weights: random score ranks."""
    rng = np.random.default_rng(seed)
    pct = score_percentiles(scores)[["component", "ligand"]]
    return pd.concat(
        [
            pct.assign(replicate=r, score_pct=rng.random(len(pct))).assign(
                score_pct=lambda d: d.groupby("component")["score_pct"].rank(pct=True)
            )
            for r in range(n)
        ],
        ignore_index=True,
    )


def test_weighted_percentile_tests_against_null():
    # Cmp 1: heavy ligands score highest; cmp 2: unrelated; cmp 3: lowest.
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
                    "weight": np.r_[np.ones(4), np.full(36, 0.01)],
                    "is_top": np.arange(40) < 4,
                }
            )
        )
    scores = pd.concat(rows, ignore_index=True)
    tests, null = weighted_percentile_tests(scores, _random_null(scores, 500))
    assert tests.loc[0, "weighted_pct"] > 0.9 and tests.loc[0, "p"] < 0.01
    assert tests.loc[2, "weighted_pct"] < 0.15 and tests.loc[2, "p"] > 0.99
    assert abs(tests.loc[1, "null_mean"] - 0.5) < 0.05
    assert tests.loc[0, "top_ligands"] == "L0, L1, L2, L3"
    np.testing.assert_allclose(
        tests["q"], adjust_pvalues(tests["p"].to_numpy(), "fdr_bh")
    )
    assert null.index.names == ["component", "replicate"] and len(null) == 1500
    # the observed statistic is the weight-averaged percentile
    pct = score_percentiles(scores)
    assert weighted_percentile(pct).loc[1] == pytest.approx(
        tests.loc[0, "weighted_pct"]
    )


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
    rankings = receiver_rankings(adata, components=[1], top_perc=20)
    scores = component_ligand_scores(adata.uns["D"], lr_pairs, lt_df, rankings)
    s = scores.set_index("ligand")
    assert np.isnan(s.loc["GZMX", "auroc"]) and not s.loc["GZMX", "is_top"]
    assert list(s.index[s["is_top"]]) == ["LIGA"]
    assert s.loc["LIGA", "auroc"] > 0.9
    assert s.loc["LIGA", "rel_weight"] == pytest.approx(1.0)
    assert (s.loc[["LIGB", "LIGC", "LIGD"], "auroc"] < 0.8).all()

    null = shuffled_percentiles(scores, rankings, lt_df, n_shuffle=3)
    assert set(null["replicate"]) == {0, 1, 2}
    assert set(null["ligand"]) <= {"LIGA", "LIGB", "LIGC", "LIGD"}
    assert null["score_pct"].between(0, 1).all()
