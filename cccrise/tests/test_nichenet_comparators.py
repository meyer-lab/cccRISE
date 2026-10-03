import anndata
import numpy as np
import pandas as pd
import pytest

from cccrise.validation.nichenet_comparators import (
    broadcast_receiver_loadings,
    comparator_ligand_scores,
    compare_methods,
    component_means,
    component_permutation_test,
    orient_receivers,
    standardized_top_scores,
)


def test_orient_receivers_gives_positive_column_means():
    out = orient_receivers(np.array([[-1.0, 2.0], [-3.0, 1.0]]))
    assert (out.mean(axis=0) > 0).all()
    np.testing.assert_array_equal(out[:, 1], [2.0, 1.0])


def test_broadcast_keeps_type_order_and_breaks_ties_within_types():
    types = np.array(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
    receiver = np.array([[0.1], [0.5], [0.3]])  # rows: A, B, C
    out = broadcast_receiver_loadings(
        types, ["A", "B", "C"], receiver, np.random.default_rng(0)
    )
    # every B cell above every C cell above every A cell
    assert out[types == "B"].min() > out[types == "C"].max()
    assert out[types == "C"].min() > out[types == "A"].max()
    # ties broken: values distinct within a type
    assert len(np.unique(out[types == "A"])) == 50


def test_broadcast_rejects_unknown_type():
    with pytest.raises(KeyError):
        broadcast_receiver_loadings(
            ["Z"], ["A"], np.ones((1, 1)), np.random.default_rng(0)
        )


def test_comparator_scores_use_broadcast_receivers_and_restore_adata():
    rng = np.random.default_rng(0)
    n, genes = 200, [f"g{i}" for i in range(30)]
    X = np.clip(rng.normal(1.0, 0.3, (n, 30)), 0, None)
    X[:40, :3] += 5.0  # the "hot" type up-regulates LIGA's targets
    types = np.array(["hot"] * 40 + ["cold"] * 160)
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"ct": types}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["rc_C"] = np.zeros((n, 1))
    lr_pairs = pd.DataFrame({"ligand": ["LIGA", "LIGB"], "receptor": ["RA", "RB"]})
    lt_df = pd.DataFrame(
        {
            "ligand": ["LIGA"] * 3 + ["LIGB"] * 3,
            "target": ["g0", "g1", "g2", "g10", "g11", "g12"],
            "weight": 1.0,
        }
    )
    scores = comparator_ligand_scores(
        adata,
        "ct",
        ["cold", "hot"],
        np.array([[-0.2], [-0.9]]),  # negative: oriented
        np.array([[1.0], [0.1]]),
        lr_pairs,
        lt_df,
        top_perc=20,
    )
    s = scores.set_index("ligand")
    assert s.loc["LIGA", "is_top"] and s.loc["LIGA", "auroc"] > 0.9
    np.testing.assert_array_equal(adata.obsm["rc_C"], np.zeros((n, 1)))
    assert "D" not in adata.uns


def _scores():
    rows = []
    for method, top_auc in (("cccRISE", 0.9), ("PB", 0.55)):
        for comp in (1, 2):
            rows += [
                {
                    "method": method,
                    "rank_mode": "own",
                    "component": comp,
                    "ligand": f"L{i}",
                    "auroc": a,
                    "is_top": i < 2,
                }
                for i, a in enumerate([top_auc, top_auc - 0.05, 0.5, 0.6, 0.4, 0.7])
            ]
    return pd.DataFrame(rows)


def test_standardized_top_scores_are_percentiles_among_non_top():
    std = standardized_top_scores(_scores())
    c = std[(std["method"] == "cccRISE") & (std["component"] == 1)].set_index("ligand")
    assert c.loc["L0", "percentile"] == 1.0  # 0.9 beats all four others
    p = std[(std["method"] == "PB") & (std["component"] == 1)].set_index("ligand")
    assert p.loc["L0", "percentile"] == pytest.approx(0.5)  # 0.55 beats 0.5, 0.4
    assert len(std) == 8


def test_compare_methods_one_sided_and_bh():
    out = compare_methods(standardized_top_scores(_scores()))
    assert list(out["method"]) == ["PB"]
    assert out.loc[0, "cles"] > 0.5 and out.loc[0, "p"] < 0.05
    assert out.loc[0, "q"] == pytest.approx(out.loc[0, "p"])


def test_component_means_and_permutation_test():
    means = component_means(standardized_top_scores(_scores()))
    assert set(means["n_top"]) == {2}
    res = component_permutation_test([1.0, 1.0, 0.9], [0.2, 0.3, 0.25], n_perm=2000)
    assert res["difference"] > 0 and res["p"] < 0.2
    same = component_permutation_test([0.5, 0.6], [0.5, 0.6], n_perm=500)
    assert same["p"] == pytest.approx(1.0)
